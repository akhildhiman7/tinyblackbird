# JAX Transformer Quick Start Guide

Get up and running with the JAX version of nanoGPT in 5 minutes.

## Prerequisites

- Python 3.8+
- pip

## Installation

### Option 1: CPU Only (Quick Test)

```bash
pip install jax flax optax orbax-checkpoint numpy
```

### Option 2: GPU (CUDA 12)

```bash
pip install -U "jax[cuda12]" flax optax orbax-checkpoint numpy
```

### Option 3: GPU (CUDA 11)

```bash
pip install -U "jax[cuda11]" flax optax orbax-checkpoint numpy
```

### Option 4: Using requirements file

```bash
pip install -r requirements_jax.txt
```

## Quick Test

Verify the installation works:

```bash
python test_jax_transformer.py
```

You should see:

```
✓ ALL TESTS PASSED!
The JAX transformer library is working correctly.
```

## Files Created

```
nanoGPT/
├── jax_transformer.py          # Core GPT model in JAX/Flax
├── jax_train_utils.py          # Training utilities
├── train_jax.py                # Main training script (JAX version of train.py)
├── test_jax_transformer.py     # Test suite
├── requirements_jax.txt        # JAX dependencies
├── JAX_README.md              # Full documentation
└── TORCH_TO_JAX_MAPPING.md    # Complete mapping of torch→JAX
```

## Basic Usage

### 1. Train a Tiny Model (Quick Test)

First, prepare a small dataset (using Shakespeare):

```bash
cd data/shakespeare_char
python prepare.py
cd ../..
```

Then train a tiny model:

```bash
python train_jax.py \
    --dataset=shakespeare_char \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --batch_size=32 \
    --block_size=256 \
    --max_iters=5000 \
    --eval_interval=500 \
    --out_dir=out_jax_shakespeare
```

### 2. Train GPT-2 Scale Model

For a full GPT-2 (124M) training run, prepare OpenWebText:

```bash
cd data/openwebtext
python prepare.py
cd ../..
```

Then train:

```bash
python train_jax.py \
    --dataset=openwebtext \
    --n_layer=12 \
    --n_head=12 \
    --n_embd=768 \
    --batch_size=12 \
    --max_iters=600000 \
    --dtype=bfloat16
```

### 3. Resume Training

```bash
python train_jax.py --init_from=resume --out_dir=out_jax
```

### 4. Eval Only

```bash
python train_jax.py --init_from=resume --eval_only=True
```

## Command Line Options

All options from `train.py` are supported:

| Option                          | Description                          | Default  |
| ------------------------------- | ------------------------------------ | -------- |
| `--n_layer`                     | Number of transformer layers         | 12       |
| `--n_head`                      | Number of attention heads            | 12       |
| `--n_embd`                      | Embedding dimension                  | 768      |
| `--batch_size`                  | Batch size                           | 12       |
| `--block_size`                  | Context length                       | 1024     |
| `--learning_rate`               | Peak learning rate                   | 6e-4     |
| `--max_iters`                   | Total training iterations            | 600000   |
| `--dtype`                       | Data type (float32/bfloat16/float16) | bfloat16 |
| `--gradient_accumulation_steps` | Gradient accumulation                | 40       |
| `--eval_interval`               | Eval frequency                       | 2000     |
| `--out_dir`                     | Output directory                     | out_jax  |
| `--seed`                        | Random seed                          | 1337     |

## Key Differences from PyTorch Version

### What's the Same

- All model architectures (GPT-2 124M, 350M, etc.)
- Training hyperparameters
- Data format and preprocessing
- Command-line interface
- Logging and checkpointing

### What's Different

- **No `init_from='gpt2'`** - Can't load pretrained PyTorch weights (only train from scratch or resume)
- **Checkpoint format** - JAX checkpoints are not compatible with PyTorch
- **Different output directory** - Defaults to `out_jax` instead of `out`
- **No GradScaler** - Use bfloat16 instead of float16 for mixed precision

## Performance Tips

1. **Use bfloat16** - More stable than float16 and well-supported:

   ```bash
   python train_jax.py --dtype=bfloat16
   ```

2. **Multi-GPU automatic** - JAX uses all GPUs by default:

   ```bash
   # Will automatically use all available GPUs
   python train_jax.py
   ```

3. **Smaller batch for testing** - Reduce memory usage:

   ```bash
   python train_jax.py --batch_size=4 --gradient_accumulation_steps=10
   ```

4. **Check device usage**:
   ```python
   import jax
   print(jax.devices())  # List all devices
   ```

## Troubleshooting

### "Import jax could not be resolved"

Install JAX:

```bash
pip install jax flax optax
```

### "bfloat16 not supported"

Use float16 or float32:

```bash
python train_jax.py --dtype=float16
```

### Out of memory

Reduce batch size or use gradient accumulation:

```bash
python train_jax.py --batch_size=4 --gradient_accumulation_steps=10
```

### Slow first iteration

JAX compiles on first run (JIT compilation). Subsequent iterations are fast.

### Different results from PyTorch

Expected - JAX uses different PRNG and numerical implementations. Results should be similar but not identical.

## Comparing PyTorch vs JAX

Run both versions on the same dataset:

```bash
# PyTorch
python train.py --dataset=shakespeare_char --max_iters=1000 --out_dir=out_torch

# JAX
python train_jax.py --dataset=shakespeare_char --max_iters=1000 --out_dir=out_jax

# Compare logs
```

## Next Steps

- Read `JAX_README.md` for comprehensive documentation
- Check `TORCH_TO_JAX_MAPPING.md` for detailed API mappings
- Run `test_jax_transformer.py` to verify installation
- Experiment with different model sizes and hyperparameters

## Common Tasks

### Save checkpoint

Checkpoints are saved automatically at eval intervals. Force save:

```bash
python train_jax.py --eval_interval=100 --always_save_checkpoint=True
```

### Use config file

Create `config/my_config.py`:

```python
n_layer = 6
n_head = 6
n_embd = 384
batch_size = 32
```

Run:

```bash
python train_jax.py config/my_config.py
```

### Enable logging

```bash
python train_jax.py --wandb_log=True --wandb_project=my_project
```

## Getting Help

1. Check test suite: `python test_jax_transformer.py`
2. Read full docs: `JAX_README.md`
3. Check mappings: `TORCH_TO_JAX_MAPPING.md`
4. Compare with PyTorch version: `train.py`

## Example: Complete Tiny Model Training

```bash
# 1. Prepare data
cd data/shakespeare_char && python prepare.py && cd ../..

# 2. Train tiny model (2M params)
python train_jax.py \
    --dataset=shakespeare_char \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --batch_size=64 \
    --block_size=256 \
    --max_iters=5000 \
    --learning_rate=1e-3 \
    --eval_interval=500 \
    --out_dir=out_tiny

# Should take ~5-10 minutes on GPU
```

## Success!

You now have a working JAX implementation of nanoGPT that replaces all PyTorch functionality!

**Key achievement:** Every `torch` API has been replaced with pure JAX/Flax equivalents.
