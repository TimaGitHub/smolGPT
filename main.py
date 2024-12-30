import sys
import os
current_dir = os.path.dirname(__file__)
candle_path = os.path.join(current_dir, "pycandle-2")
sys.path.append(candle_path)
import warnings
warnings.simplefilter("ignore", UserWarning)

try:
    import cupy
    device = 'gpu'
except:
    device = 'cpu'

import tiktoken
import candle
import candle.nn as nn
from candle import Tensor
from dataclasses import dataclass
import re
import argparse
from tqdm import tqdm

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GeLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class CausalSelfAttetion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = Tensor.split(qkv, 3, axis=2)
        n_head = self.config.n_head
        q = q.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3)
        att = q @ k.transpose(0, -3, -1, -2) * (k.shape[-1] ** -0.5)
        lg = att.local_gradients
        att = Tensor.tril(att)
        att[att == 0] = float('-inf')
        att.local_gradients = lg
        probs = Tensor.softmax(att, axis=-1)
        y = probs @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttetion(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(1024, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.config.block_size, f"Cannot forward the sequence of length {T}, block_size is smaller"
        pos = Tensor.arange(T).reshape(1, -1)
        pos_embd = self.wpe(pos)
        tok_embd = self.wte(x)
        x = pos_embd + tok_embd

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        model = GPT.get_params(model, model_type)

        return model

    @staticmethod
    def get_params(model, model_type):

        from transformers import GPT2LMHeadModel, logging

        logging.set_verbosity_error()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [key for key in sd_keys_hf if not key.endswith('.attn.bias')]
        sd_keys_hf = [key for key in sd_keys_hf if not key.endswith('.attn.masked_bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        model_map = {
            'wte': getattr(model, 'wte').w,
            'wpe': getattr(model, 'wpe').w,
            'ln_f': getattr(model, 'ln_f'),
            'h': getattr(model, 'h'),
            'lm_head': getattr(model, 'lm_head').w
        }

        for k in sd_keys_hf:
            attr = None
            if 'transformer' in k:
                if 'wte' in k:
                    attr = model_map['wte']

                elif 'wpe' in k:
                    attr = model_map['wpe']

                elif 'ln_f' in k:
                    if 'weight' in k:
                        attr = model_map['ln_f'].gamma
                    else:
                        attr = model_map['ln_f'].beta

                elif 'h' in k:
                    pos = int(re.search(r'\d+', k).group())
                    attr = model_map['h'][pos]
                    transformer_map = {
                        'ln_1': attr.ln_1,
                        'c_attn': attr.attn.c_attn,
                        'c_proj': attr.attn.c_proj,
                        'ln_2': attr.ln_2,
                        'c_fc': attr.mlp.c_fc,
                        'mlp_c_proj': attr.mlp.c_proj,

                    }
                    if 'ln_1' in k:
                        if 'weight' in k:
                            attr = transformer_map['ln_1'].gamma
                        else:
                            attr = transformer_map['ln_1'].beta

                    elif 'attn' in k:
                        if 'c_attn' in k:
                            if 'weight' in k:
                                attr = transformer_map['c_attn'].w
                            else:
                                attr = transformer_map['c_attn'].b

                        elif 'c_proj' in k:
                            if 'weight' in k:
                                attr = transformer_map['c_proj'].w
                            else:
                                attr = transformer_map['c_proj'].b

                    elif 'ln_2' in k:
                        if 'weight' in k:
                            attr = transformer_map['ln_2'].gamma
                        else:
                            attr = transformer_map['ln_2'].beta

                    elif 'mlp' in k:

                        if 'c_fc' in k:
                            if 'weight' in k:
                                attr = transformer_map['c_fc'].w
                            else:
                                attr = transformer_map['c_fc'].b

                        elif 'c_proj' in k:
                            if 'weight' in k:
                                attr = transformer_map['mlp_c_proj'].w
                            else:
                                attr = transformer_map['mlp_c_proj'].b
            elif 'lm_head' in k:
                attr = model_map['lm_head']

            if any(k.endswith(w) for w in transposed):
                hf_value = sd_hf[k].t()
            else:
                hf_value = sd_hf[k]

            if not any(substr in k for substr in ['wte', 'wpe', 'ln_1', 'ln_2', 'ln_f']):
                if 'weight' in k:
                    hf_value = hf_value.t()
            assert attr.shape == hf_value.shape
            attr.value = hf_value.data.cpu().numpy()
        return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pretrained gpt2')
    parser.add_argument('--prompt', type=str, default="Hello! I am a language model, ")
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--model', type=str, default='124M')  # 124M, 345M, 762M, 1542M
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--eos', type=bool, default=False)
    args = parser.parse_args()

    assert args.model in ('124M', '345M', '762M', '1542M')

    model_type = {'124M':'gpt2',
                  '345M':'gpt2-medium',
                  '762M':'gpt2-large',
                  '1542M':'gpt2-xl'}[args.model]

    model = GPT.from_pretrained(model_type=model_type)
    model = model.to(args.device)

    max_new_tokens = args.max_new_tokens
    number_of_examples = 1
    starting_sentence = args.prompt

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(starting_sentence)
    tokens = Tensor(tokens, dtype=int)
    tokens = Tensor.unsqueeze(tokens, 0)
    tokens = Tensor.repeat(tokens, number_of_examples, 0)
    tokens = tokens.to(args.device)

    topk = args.topk  # responsible for "creativity" or "adequacy"
    x = tokens

    import threading
    from time import sleep
    def user_prompt(text, delay=0.1):
        for start_token in text:  # printing user prompt
            output = start_token.value.astype(int).tolist()
            print(enc.decode([output]), end="", flush=True)
            sleep(delay)

    thread = threading.Thread(target=user_prompt, args=(tokens[0], 0.5))
    thread.start()

    for i in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :]
        probs = Tensor.safe_softmax(logits, axis=-1)
        topk_indices, topk_probs = Tensor.topk(probs, topk)
        new_token = Tensor.multinomial_from_array(topk_indices, topk_probs, num_samples=1).reshape(-1, 1)
        x = Tensor.cat([x, new_token], axis=1)
        output = x[0][-1].value.astype(int).tolist()
        if output == 50256 and args.eos: # <|endoftext|> token
            break
        thread.join()
        print(enc.decode([output]), end="", flush=True)

    # sample = x[0]
    # sample = sample.value.astype(int).tolist()
    # print(enc.decode(sample))