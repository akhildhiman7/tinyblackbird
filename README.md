# NanoGPT JAX - PyTorch-Free Implementation

A pure JAX/Flax implementation of NanoGPT for training character-level language models.

## Prerequisites

- Python 3.10+
- pip

## Installation

### CPU Only
```bash
pip install jax jaxlib flax optax numpy
```

### NVIDIA GPU (CUDA 12)
```bash
pip install jax[cuda12] flax optax numpy
```

## Quick Start

### 1. Prepare the Shakespeare Dataset

```bash
cd data/shakespeare_char
python prepare.py
cd ../..
```

### 2. Train the Model

**Quick test (20 iterations, ~3 min on CPU):**
```bash
python train_jax.py \
    --dataset=shakespeare_char \
    --batch_size=32 \
    --block_size=256 \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --max_iters=20 \
    --eval_interval=10 \
    --eval_iters=10 \
    --dropout=0.2 \
    --learning_rate=1e-3 \
    --gradient_accumulation_steps=1
```

**Full training (5000 iterations):**
```bash
python train_jax.py \
    --dataset=shakespeare_char \
    --batch_size=64 \
    --block_size=256 \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --max_iters=5000 \
    --eval_interval=500 \
    --dropout=0.2 \
    --learning_rate=1e-3 \
    --gradient_accumulation_steps=1
```

Checkpoints will be saved to `out_jax/`.

### 3. Generate Text Samples

```bash
python sample_jax.py \
    --out_dir=out_jax \
    --num_samples=5 \
    --max_new_tokens=200 \
    --temperature=0.8
```

### 4. Custom Prompts

```bash
python sample_jax.py \
    --out_dir=out_jax \
    --start="ROMEO:" \
    --num_samples=3 \
    --max_new_tokens=300
```

## Expected Output

| Training Progress | Loss | Output Quality |
|-------------------|------|----------------|
| 20 iterations | ~3.2 | Gibberish |
| 500 iterations | ~2.0 | Some structure |
| 5000 iterations | ~1.5 | Coherent Shakespeare-like text |

**Example output after full training:**
```
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun...
```

## Hardware Recommendations

| Hardware | Training Time (5000 iters) |
|----------|---------------------------|
| CPU (M3 Pro) | ~30-60 min |
| NVIDIA RTX 4090 | ~5-10 min |
| NVIDIA A100 | ~3-5 min |

## Project Structure

```
nanoGPT/
├── train_jax.py        # Training script (JAX/Flax)
├── sample_jax.py       # Text generation script (JAX/Flax)
├── jax_transformer.py  # GPT model implementation
├── jax_train_utils.py  # Training utilities
├── configurator.py     # Config override utility
├── data/
│   └── shakespeare_char/
│       ├── prepare.py  # Dataset preparation
│       ├── input.txt   # Raw Shakespeare text
│       └── meta.pkl    # Tokenizer metadata
└── out_jax/            # Saved checkpoints
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name | `openwebtext` |
| `--batch_size` | Batch size | `12` |
| `--block_size` | Context length | `1024` |
| `--n_layer` | Number of transformer layers | `12` |
| `--n_head` | Number of attention heads | `12` |
| `--n_embd` | Embedding dimension | `768` |
| `--max_iters` | Training iterations | `600000` |
| `--learning_rate` | Learning rate | `6e-4` |
| `--dropout` | Dropout rate | `0.0` |

## Sampling Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--out_dir` | Checkpoint directory | `out_jax` |
| `--start` | Prompt text | `"\n"` |
| `--num_samples` | Number of samples | `10` |
| `--max_new_tokens` | Tokens per sample | `500` |
| `--temperature` | Sampling temperature | `0.8` |
| `--top_k` | Top-k filtering | `200` |

## License

MIT
