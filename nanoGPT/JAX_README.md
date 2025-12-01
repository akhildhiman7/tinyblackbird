# JAX Transformer Library for nanoGPT

This directory contains a complete JAX/Flax reimplementation of nanoGPT's PyTorch code. All `torch` functionality has been replaced with pure JAX equivalents.

## Files Overview

### Core Library Files

1. **`jax_transformer.py`** - Pure JAX/Flax GPT model implementation

   - Replaces `model.py`
   - Implements: `GPT`, `GPTConfig`, `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`
   - Uses Flax for neural network modules
   - Functional JAX design with immutable parameters

2. **`jax_train_utils.py`** - Training utilities and helpers

   - Device management, data loading, checkpointing
   - Gradient clipping, learning rate schedules
   - Distributed training support (via pmap)
   - Replaces PyTorch utilities like `torch.save`, `torch.load`, `clip_grad_norm_`

3. **`train_jax.py`** - Main training script
   - Complete JAX reimplementation of `train.py`
   - Uses JIT compilation for performance
   - Supports single and multi-device training

## Installation

Install the required JAX dependencies:

```bash
# For CPU only
pip install jax flax optax

# For GPU (CUDA 12)
pip install -U "jax[cuda12]" flax optax

# For GPU (CUDA 11)
pip install -U "jax[cuda11]" flax optax

# For TPU
pip install -U "jax[tpu]" flax optax -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## PyTorch → JAX Migration Guide

### Key Differences

| PyTorch Concept            | JAX Equivalent                                   | Notes                              |
| -------------------------- | ------------------------------------------------ | ---------------------------------- |
| `torch.nn.Module`          | `flax.linen.Module`                              | Functional style, immutable        |
| `model.parameters()`       | `state.params` (PyTree)                          | Parameters are nested dicts/arrays |
| `loss.backward()`          | `jax.grad(loss_fn)`                              | Functional gradient computation    |
| `optimizer.step()`         | `state.apply_gradients()`                        | State updates return new state     |
| `torch.no_grad()`          | Just don't compute grads                         | No special context needed          |
| `torch.save()`             | `flax.training.checkpoints.save_checkpoint()`    | Different format                   |
| `torch.load()`             | `flax.training.checkpoints.restore_checkpoint()` | Different format                   |
| `torch.manual_seed()`      | `jax.random.PRNGKey()`                           | Functional PRNG with explicit keys |
| `torch.compile()`          | `@jax.jit`                                       | JIT compilation via XLA            |
| DDP                        | `jax.pmap`                                       | Data parallel via pmap             |
| `torch.cuda.synchronize()` | `array.block_until_ready()`                      | Explicit synchronization           |
| `torch.autocast()`         | Set dtype explicitly                             | No autocast context                |
| GradScaler                 | Manual loss scaling                              | No built-in scaler                 |

### Torch APIs Replaced

Here's a detailed mapping of every `torch` usage in `train.py` and its JAX equivalent:

#### Imports

```python
# PyTorch
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# JAX
import jax
import jax.numpy as jnp
from jax_train_utils import setup_distributed_training
# (DDP functionality via jax.pmap)
```

#### Device & Dtype Setup

```python
# PyTorch
torch.cuda.is_available()
torch.cuda.is_bf16_supported()
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = 'cuda'
ptdtype = torch.bfloat16

# JAX
device_info = check_device_availability()
rng = jax.random.PRNGKey(1337)
# (TF32 handled automatically by XLA)
device = jax.devices()[0]
compute_dtype = jnp.bfloat16
```

#### Autocast Context

```python
# PyTorch
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
with ctx:
    logits, loss = model(X, Y)

# JAX
# Set dtype in model directly, no autocast context needed
logits, loss = state.apply_fn({'params': state.params}, X, Y)
```

#### Data Loading

```python
# PyTorch
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy(...) for i in ix])
x = x.pin_memory().to(device, non_blocking=True)

# JAX
ix = jax.random.randint(rng, (batch_size,), 0, len(data) - block_size)
x = jnp.stack([jnp.array(...) for i in ix])
x = jax.device_put(x, device)
```

#### Model & Optimizer

```python
# PyTorch
model = GPT(gptconf)
model.to(device)
optimizer = model.configure_optimizers(...)

# JAX
state = create_train_state(rng, gptconf, lr, weight_decay, betas)
# state contains model, params, and optimizer
```

#### Checkpointing

```python
# PyTorch
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model'])
torch.save(checkpoint, path)

# JAX
state, metadata = load_checkpoint(checkpoint_dir, state)
save_checkpoint(checkpoint_dir, state, ...)
```

#### GradScaler (Mixed Precision)

```python
# PyTorch
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# JAX
# For bfloat16: just use it directly (preferred on TPU/modern GPUs)
# For float16: implement manual loss scaling if needed
loss, grads = jax.value_and_grad(loss_fn)(params)
state = state.apply_gradients(grads=grads)
```

#### Compilation

```python
# PyTorch
model = torch.compile(model)

# JAX
@jax.jit
def train_step(state, inputs, targets):
    ...
```

#### DDP (Distributed)

```python
# PyTorch
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# JAX
# Use pmap for data parallel training
train_step_pmap = jax.pmap(train_step, axis_name='device')
```

#### No-grad Evaluation

```python
# PyTorch
@torch.no_grad()
def estimate_loss():
    model.eval()
    ...

# JAX
# Just don't compute gradients - call apply_fn with train=False
def estimate_loss():
    loss = state.apply_fn({'params': state.params}, X, Y, train=False)
    ...
```

#### Gradient Clipping

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# JAX
grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
clip_factor = jnp.minimum(1.0, grad_clip / (grad_norm + 1e-6))
grads = jax.tree_map(lambda g: g * clip_factor, grads)
```

#### Training Loop

```python
# PyTorch
logits, loss = model(X, Y)
loss.backward()
optimizer.step()
optimizer.zero_grad()

# JAX
def loss_fn(params):
    _, loss = state.apply_fn({'params': params}, X, Y, train=True)
    return loss

loss, grads = jax.value_and_grad(loss_fn)(state.params)
state = state.apply_gradients(grads=grads)
```

## Usage

### Basic Training

Train from scratch (single device):

```bash
python train_jax.py --batch_size=32
```

### Configuration Options

All original `train.py` options are supported:

```bash
python train_jax.py \
    --dataset=openwebtext \
    --n_layer=12 \
    --n_head=12 \
    --n_embd=768 \
    --batch_size=12 \
    --learning_rate=6e-4 \
    --max_iters=600000 \
    --dtype=bfloat16
```

### Resume from Checkpoint

```bash
python train_jax.py --init_from=resume --out_dir=out_jax
```

### Multi-Device Training

JAX automatically uses all available devices. For explicit data parallel:

```bash
# Will automatically use all GPUs/TPUs
python train_jax.py --use_pmap=True
```

## Architecture Details

### Flax Module Design

The JAX implementation uses Flax's `linen` API, which is similar to PyTorch but with key differences:

1. **Immutable Parameters**: Parameters are stored in PyTrees (nested dicts), not as class attributes
2. **Functional Style**: Forward pass is pure function (no side effects)
3. **Explicit RNG**: Random number generation requires explicit PRNG keys
4. **Setup in `@nn.compact`**: Layers defined in the forward pass method

### Example Model Structure

```python
# PyTorch style
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# JAX/Flax style
class MyModule(nn.Module):
    @nn.compact
    def __call__(self, x):
        linear = nn.Dense(10)
        return linear(x)
```

### Training State

JAX uses `TrainState` to manage model, parameters, and optimizer together:

```python
state = train_state.TrainState.create(
    apply_fn=model.apply,  # Forward function
    params=params,         # Model parameters (PyTree)
    tx=optimizer          # Optax optimizer
)

# Training step
loss, grads = jax.value_and_grad(loss_fn)(state.params)
state = state.apply_gradients(grads=grads)
```

## Performance Tips

1. **Use bfloat16 on TPUs and modern GPUs** - Better numerical stability than float16
2. **JIT compile everything** - Use `@jax.jit` on all hot paths
3. **Use pmap for multi-device** - Automatic data parallelism
4. **Avoid host-device transfers** - Keep data on device between iterations
5. **Profile with JAX profiler** - Use `jax.profiler.trace()` for TensorBoard

## Limitations & Differences

1. **No pretrained GPT-2 loading**: The `init_from='gpt2*'` option is not implemented (would require converting HuggingFace weights to Flax format)
2. **Checkpoint format incompatible**: JAX checkpoints cannot be loaded by PyTorch version and vice versa
3. **Different PRNG behavior**: JAX uses functional PRNG, so results won't exactly match PyTorch even with same seed
4. **No automatic mixed precision**: Must explicitly choose dtype (though bfloat16 is preferred on supported hardware)
5. **Distributed training differs**: Uses pmap instead of DDP (conceptually similar but different API)

## Debugging

### Check device placement

```python
device_info = check_device_availability()
print(device_info)
```

### Inspect model structure

```python
print(jax.tree_util.tree_map(lambda x: x.shape, state.params))
```

### Profile performance

```python
with jax.profiler.trace("/tmp/tensorboard"):
    # Training code here
    pass
```

### Check for NaN/Inf

```python
has_nan = jnp.any(jnp.isnan(loss))
has_inf = jnp.any(jnp.isinf(loss))
```

## Contributing

When adding new features, maintain the mapping:

- Keep function signatures similar to PyTorch version
- Document all PyTorch → JAX replacements
- Add tests comparing outputs (where feasible)

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [JAX Transformers Examples](https://github.com/google/flax/tree/main/examples)
