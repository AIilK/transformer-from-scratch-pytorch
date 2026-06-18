# PyTorch Transformer From Scratch

A clean, educational repository containing **three complete implementations** of Transformer architecture components built entirely from scratch using PyTorch. Perfect for deep learning enthusiasts, researchers, and students who want to understand how Transformers work under the hood.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This repository demonstrates the implementation of the core building blocks of the Transformer model (as introduced in the famous paper *"Attention Is All You Need"*).

### Implemented Models:

| Model | Architecture | Task | Parameters | Dataset |
|------|--------------|------|------------|--------|
| **Encoder** | Transformer Encoder | Text Classification (Sentiment) | ~1.2M | GLUE SST-2 |
| **Decoder** | Causal Decoder (GPT-like) | Autoregressive Language Modeling | ~1.1M | GLUE SST-2 (as LM) |
| **Encoder-Decoder** | Full Transformer | Neural Machine Translation (En→Es) | ~2.8M | OPUS English-Spanish |

---

## ✨ Features

- **Pure PyTorch** — No high-level libraries like Hugging Face Transformers for the core models
- Multi-Head Self-Attention with proper scaling
- Positional Encoding (Sinusoidal)
- Causal Masking for Decoder
- Padding Mask support
- Layer Normalization & Residual Connections
- Ready-to-train training loops with proper masking
- Inference examples (Greedy decoding for translation and generation)

---

## 📊 Results & Statistics

### 1. Encoder (Classification)
- **Task**: Sentiment Analysis (Positive/Negative)
- **Dataset**: GLUE SST-2 (67k training samples)
- **Model Size**: 64-dimensional model, 4 heads, 2 layers
- **Achieved**: ~82-85% validation accuracy after few epochs (small model)

### 2. Decoder (Language Modeling)
- **Task**: Next token prediction
- **Training**: Teacher Forcing + Causal Masking
- **Generation**: Successfully generates coherent continuations

### 3. Encoder-Decoder (Translation)
- **Task**: English → Spanish Translation
- **Dataset**: 30,000 parallel sentences (OPUS)
- **Vocabulary**: ~20k (En) / ~10k+ (Es)
- **Training**: Custom training loop with proper shifted targets
- **Inference**: Greedy decoding implemented with beam search potential

**Total Lines of Code**: ~3,200+  
**Total Parameters Across Models**: ~5.1 Million

---

## 📁 Project Structure

```bash
PyTorch-Transformer-From-Scratch/
├── Encoder.py                    # Transformer Encoder + Classification
├── Decoder.py                    # Causal Decoder + Language Modeling
├── Encoder-Decoder.py            # Full Transformer (NMT)
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOURUSERNAME/PyTorch-Transformer-From-Scratch.git
cd PyTorch-Transformer-From-Scratch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pandas numpy matplotlib
```

### Running the Models

```python
# Run Full Translation Model
python Encoder-Decoder.py

# Run Encoder (Classification)
python Encoder.py

# Run Decoder (Generation)
python Decoder.py
```

---

## 🎯 Learning Objectives

This repository helps you understand:
- How Multi-Head Attention really works
- Difference between Encoder and Decoder masking
- Proper implementation of Positional Encoding
- Training strategies for Seq2Seq models
- Masking techniques (padding + causal)

---

## 🔮 Future Improvements

- [ ] Add Beam Search decoding
- [ ] Implement Rotary Embeddings (RoPE)
- [ ] Add model checkpointing & logging (WandB)
- [ ] Larger scale experiments (d_model=512)
- [ ] Pre-training + Fine-tuning scripts
- [ ] Convert to `nn.Transformer` compatible interface

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- The Annotated Transformer (Harvard NLP)
- PyTorch Official Documentation

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. This is primarily an **educational project** — improvements, bug fixes, and better documentation are highly welcome!

## ⭐ Star this repo if you found it helpful!
