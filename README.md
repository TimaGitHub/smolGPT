# smolGPT 🐾

*smol but mighty.*


You've seen [openai/gpt-2](https://github.com/openai/gpt-2).

You've seen [karpathy/minGPT](https://github.com/karpathy/mingpt).

You've even seen [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)!

You might have even seen [jaymody/picoGPT](https://github.com/jaymody/picoGPT)!

But you have never seen smolGPT!!!

### What is **smolGPT**?

**smolGPT** is a small-scale version of the GPT architecture, built for research and learning purposes. This is a **research project** focused on understanding and experimenting with the inner workings of GPT-like architectures.

What makes **smolGPT** special? It’s built using **[pycandle-2](https://github.com/TimaGitHub/pycandle-2)**, a lightweight machine learning library that I’ve created from scratch. By using **NumPy** as the foundation, the goal was to implement a minimalist GPT model without relying on large, heavyweight frameworks like TensorFlow or PyTorch.
 This library allows you to run, train, and experiment with machine learning models while keeping things easy to understand.

---


picoGPT features:
* Fast? ❌ smolPGT is supaSLOW 🐌 We say 🚫 to CV-cache, Quantization and Distillation 
* Training code? ✅ Yes, but it may cause you 💢!
* top-p sampling? ❌ top-k? ✅ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Self-made??? ✅✅ YESS!!! I made it completely from scratch in numpy😲😲😲 
* Scalable? **(੭˃ᴗ˂)੭** You may build whatever architecture you want with **PyCandle**. 

**GPT2-3-4?**, **Llama 1-2-3?** 😎👌🔥 just provide model weights🤔



### 📦 Installation

To install **smolGPT** with **pycandle-2**, follow these simple steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/TimaGitHub/smolGPT.git
   cd smolGPT
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
   ```bash
   python main.py
   ```

And that’s it! You’re ready to start generating text with **smolGPT**. 😅

---

### 🚀 Example Usage

With **smolGPT**, you can quickly generate text. Here’s an example:

```bash
python main.py --prompt "Hello! i am a language model," --max_new_tokens 30 --model 124M --device gpu --topk 30
```
#### **output 😅**

```bach
Hello! i am a language model, I have no language background and this is not a problem

Anonymous 01/11/15 (Thu) 04:09:19 AM No.
```

The model’s lighthearted and playful nature makes it a fun tool for experimenting with GPT-like architectures (**124M**, **345M**, **762M**, **1542M**) 😆

### **Some fun generations**
---
#### *124M Parameters*
```python
prompt = "I will prove this mathematical theorem: "  
```
![145](https://github.com/user-attachments/assets/14456469-e0b6-466c-b5e8-b9fc111f7b31)

#### *345M Parameters*
```python
prompt = "I will prove this mathematical theorem: "  
```
![345](https://github.com/user-attachments/assets/1018b84d-f601-42c7-8bb3-a0a263f34c80)

#### *762M Parameters*
```python
prompt = "for i in range(5):"  
```
![762_1](https://github.com/user-attachments/assets/e32e4354-c03a-466e-8ad6-9eef54607c17)

#### *762M Parameters*
```python
prompt = "a = [10, 35, 20, -40]\nsorted(a)\n>>>"  
```
![762_2](https://github.com/user-attachments/assets/dbb3cb5d-eaff-42ce-a889-9dc705a76d06)

#### *762M Parameters*
```python
prompt = "one small step for a man"  
```
![762_3](https://github.com/user-attachments/assets/c14271da-a19c-4274-93f2-16677a9e85f5)


### 💡 Why **smolGPT**?

- **Learning-Focused**: Built to explore and experiment with GPT architecture, not necessarily to run on embedded or resource-limited systems.
- **Small Size**: Perfect for local experiments and educational purposes.
- **Built with **pycandle-2****: Leverages a custom **NumPy**-based machine learning library for simplicity and efficiency.
- **Lightweight**: While small, it retains the core principles of GPT models.

---

### 🤖 Contributing

If you’d like to improve **smolGPT** or contribute to **pycandle-2**, feel free to fork the repository, make your changes, and submit a pull request. New ideas and contributions are always welcome!
