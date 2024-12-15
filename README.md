# smolGPT 🐾

*smol but mighty.*


You've seen [openai/gpt-2](https://github.com/openai/gpt-2).

You've seen [karpathy/minGPT](https://github.com/karpathy/mingpt).

You've even seen [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)!

You might have seen [picoGPT](https://github.com/jaymody/picoGPT)!

But you have never seen smolGPT!!!

### What is **smolGPT**?

**smolGPT** is a small-scale version of the GPT architecture, built for research and learning purposes. This is a **research project** focused on understanding and experimenting with the inner workings of GPT-like architectures.

What makes **smolGPT** special? It’s built using **[pycandle-2](https://github.com/TimaGitHub/pycandle-2)**, a lightweight machine learning library that I’ve created from scratch. By using **NumPy** as the foundation, the goal was to implement a minimalist GPT model without relying on large, heavyweight frameworks like TensorFlow or PyTorch.
 This library allows you to run, train, and experiment with machine learning models while keeping things easy to understand.

---


picoGPT features:
* Fast? ❌ smolPGT is supaSLOW 🐌 We say 🚫 to CV-cache, Quantization and Distillation 
* Training code? ✅ Yes, but very it may cause you 💢!
* top-p sampling? ❌ top-k? ❌ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Self-made??? ✅✅ YESS!!! I made it completely from scratch in numpy😲😲😲 
* Scalable? **(੭˃ᴗ˂)੭** You may build whatever architecture you want with **PyCandle**. 

**GPT2-3-4?**, **Lamma 1-2-3?** 😎👌🔥 just provide model weights🤔



### 📦 Installation

To install **smolGPT** with **pycandle-2**, follow these simple steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/smolGPT.git
   cd smolGPT
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install **pycandle-2**:
   Since **pycandle-2** is your custom library, we’ll need to install it manually:
   ```bash
   git clone https://github.com/TimaGitHub/pycandle-2.git
   cd pycandle-2
   pip install .
   ```

4. Run the model:
   ```bash
   python smolgpt.py
   ```

And that’s it! You’re ready to start generating text with **smolGPT**. 😅

---

### 🚀 Example Usage

With **smolGPT**, you can quickly interact with the model and generate text or engage in a conversation. Here’s an example of an interaction:

```
You: Hey, smolGPT! How are you doing?
smolGPT: Hey! I'm here... small, but I can do a lot!
You: Tell me something funny.
smolGPT: Why do programmers hate training cats? Because they can never make them sit in a forever loop. 🐱
```

The model’s lighthearted and playful nature makes it a fun tool for experimenting with GPT-like architectures! 😆

---

### 🔧 How to Train **smolGPT**?

Want to train your own **smolGPT**? Training the model is simple with **pycandle-2**. Here’s how:

1. Download a dataset (e.g., **WikiText-2** or your own custom dataset).
2. Run the training script to fine-tune or train the model from scratch.

The process is designed to be straightforward, and you can find additional details in the [pycandle-2 README](https://github.com/TimaGitHub/pycandle-2) for training instructions.

---

### 💡 Why **smolGPT**?

- **Learning-Focused**: Built to explore and experiment with GPT architecture, not necessarily to run on embedded or resource-limited systems.
- **Small Size**: Perfect for local experiments and educational purposes.
- **Built with **pycandle-2****: Leverages a custom **NumPy**-based machine learning library for simplicity and efficiency.
- **Lightweight**: While small, it retains the core principles of GPT models.

---

### 🤖 Contributing

If you’d like to improve **smolGPT** or contribute to **pycandle-2**, feel free to fork the repository, make your changes, and submit a pull request. New ideas and contributions are always welcome!
