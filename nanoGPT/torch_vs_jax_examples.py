"""
Side-by-side comparison of key torch vs JAX patterns.
This file demonstrates the mapping for educational purposes.
"""

# =============================================================================
# EXAMPLE 1: Imports
# =============================================================================

# PyTorch version (train.py)
"""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
"""

# JAX version (train_jax.py)
"""
import jax
import jax.numpy as jnp
from jax_transformer import GPTConfig, GPT, create_train_state
from jax_train_utils import setup_distributed_training
"""

# =============================================================================
# EXAMPLE 2: Device Setup
# =============================================================================

# PyTorch version
"""
device = 'cuda'
torch.cuda.set_device(device)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
"""

# JAX version
"""
from jax_train_utils import check_device_availability, get_dtype, setup_jax_environment

device_info = check_device_availability()
if dtype == 'bfloat16' and not device_info['bf16_supported']:
    dtype = 'float16'
compute_dtype = get_dtype(dtype)
rng = setup_jax_environment(seed=1337)
# TF32 handled automatically by XLA
"""

# =============================================================================
# EXAMPLE 3: Model Initialization
# =============================================================================

# PyTorch version
"""
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
"""

# JAX version
"""
from jax_transformer import create_train_state

gptconf = GPTConfig(**model_args)
rng, init_rng = jax.random.split(rng)
state = create_train_state(
    init_rng,
    gptconf,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    betas=(beta1, beta2)
)
# state contains model, params, and optimizer together
"""

# =============================================================================
# EXAMPLE 4: Data Loading
# =============================================================================

# PyTorch version
"""
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
if device_type == 'cuda':
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
"""

# JAX version
"""
from jax_train_utils import get_batch_jax

X, Y, rng = get_batch_jax(
    data_dir='data/openwebtext',
    split='train',
    batch_size=batch_size,
    block_size=block_size,
    rng=rng,
    device=jax.devices()[0]
)
"""

# =============================================================================
# EXAMPLE 5: Training Step
# =============================================================================

# PyTorch version
"""
# Forward
with torch.amp.autocast(device_type=device_type, dtype=ptdtype):
    logits, loss = model(X, Y)
    loss = loss / gradient_accumulation_steps

# Backward
scaler.scale(loss).backward()

# Clip gradients
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# Update
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
"""

# JAX version
"""
# Define loss function
def loss_fn(params):
    _, loss = state.apply_fn(
        {'params': params},
        X,
        targets=Y,
        train=True,
        rngs={'dropout': rng} if dropout > 0 else None
    )
    return loss / gradient_accumulation_steps

# Forward and backward (combined)
loss, grads = jax.value_and_grad(loss_fn)(state.params)

# Accumulate gradients (if using gradient accumulation)
accumulated_grads = jax.tree_map(
    lambda acc, g: acc + g,
    accumulated_grads,
    grads
)

# Clip gradients
if grad_clip > 0.0:
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(accumulated_grads)))
    clip_factor = jnp.minimum(1.0, grad_clip / (grad_norm + 1e-6))
    accumulated_grads = jax.tree_map(lambda g: g * clip_factor, accumulated_grads)

# Update (no zero_grad needed - functional)
state = state.apply_gradients(grads=accumulated_grads)
"""

# =============================================================================
# EXAMPLE 6: Evaluation
# =============================================================================

# PyTorch version
"""
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
"""

# JAX version
"""
def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y, _ = get_batch_jax(data_dir, split, batch_size, block_size, rng)
            _, loss = state.apply_fn(
                {'params': state.params},
                X,
                targets=Y,
                train=False  # Inference mode
            )
            losses.append(float(loss))
        out[split] = np.mean(losses)
    return out
"""

# =============================================================================
# EXAMPLE 7: Checkpointing
# =============================================================================

# PyTorch version
"""
# Save
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

# Load
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
"""

# JAX version
"""
from jax_train_utils import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    checkpoint_dir=out_dir,
    state=state,  # Contains params and optimizer state
    model_args=model_args,
    iter_num=iter_num,
    best_val_loss=best_val_loss,
    config=config
)

# Load
state, metadata = load_checkpoint(out_dir, state)
iter_num = metadata['iter_num']
best_val_loss = metadata['best_val_loss']
"""

# =============================================================================
# EXAMPLE 8: JIT Compilation
# =============================================================================

# PyTorch version
"""
model = torch.compile(model)  # Compile entire model
"""

# JAX version
"""
@jax.jit
def train_step(state, inputs, targets, rng):
    def loss_fn(params):
        _, loss = state.apply_fn({'params': params}, inputs, targets=targets, train=True)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, grads

# Use compiled function
loss, grads = train_step(state, X, Y, rng)
"""

# =============================================================================
# EXAMPLE 9: Distributed Training
# =============================================================================

# PyTorch version
"""
# Setup
from torch.distributed import init_process_group
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])

# Wrap model
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[ddp_local_rank])

# Training (gradients synced automatically)
loss.backward()
"""

# JAX version
"""
# Setup (automatic)
from jax_train_utils import setup_distributed_training
dist_info = setup_distributed_training()
rank = dist_info['rank']
world_size = dist_info['world_size']

# Use pmap for data parallel
from jax_train_utils import pmap_train_step
train_step_pmap = jax.pmap(pmap_train_step, axis_name='device')

# Replicate state across devices
replicated_state = jax.device_put_replicated(state, jax.local_devices())

# Training (gradients averaged via pmean)
replicated_state, loss = train_step_pmap(replicated_state, batch, dropout_rng)
"""

# =============================================================================
# EXAMPLE 10: Random Number Generation
# =============================================================================

# PyTorch version
"""
torch.manual_seed(42)  # Global state
x = torch.randint(0, 100, (10,))  # Uses global RNG
"""

# JAX version
"""
rng = jax.random.PRNGKey(42)  # Explicit key
rng, sample_rng = jax.random.split(rng)  # Split for new RNG
x = jax.random.randint(sample_rng, (10,), 0, 100)  # Use explicit key
"""

# =============================================================================
# KEY DIFFERENCES SUMMARY
# =============================================================================

"""
Paradigm:
  PyTorch: Imperative, object-oriented, mutable state
  JAX: Functional, immutable state, pure functions

Parameters:
  PyTorch: model.parameters() - list of Parameter objects
  JAX: state.params - nested dict (PyTree) of arrays

Gradients:
  PyTorch: loss.backward() - accumulates in .grad attributes
  JAX: jax.grad(loss_fn)(params) - returns gradient PyTree

Optimizer:
  PyTorch: Separate object with .step() and .zero_grad()
  JAX: Part of TrainState, .apply_gradients() returns new state

Random:
  PyTorch: Global RNG state (torch.manual_seed)
  JAX: Explicit PRNG keys (jax.random.PRNGKey, .split)

Device:
  PyTorch: .to(device), .cuda(), .cpu()
  JAX: jax.device_put(x, device), mostly automatic

Compilation:
  PyTorch: torch.compile() or TorchScript
  JAX: @jax.jit decorator, XLA-first

Distributed:
  PyTorch: DDP with process groups
  JAX: pmap with device axis

No-grad:
  PyTorch: @torch.no_grad() or torch.inference_mode()
  JAX: Just don't call grad functions (train=False)

Checkpoints:
  PyTorch: torch.save/load with state_dict
  JAX: flax.training.checkpoints (different format)
"""

print("This file demonstrates PyTorch → JAX pattern mappings.")
print("See the actual implementations in:")
print("  - jax_transformer.py (model)")
print("  - jax_train_utils.py (utilities)")
print("  - train_jax.py (training script)")
