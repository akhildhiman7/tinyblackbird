"""
JAX Training Script
This is a JAX/Flax reimplementation of train.py using the jax_transformer library.

To run on a single device:
$ python train_jax.py --batch_size=32

To run with data parallel across multiple devices:
$ python train_jax.py --use_pmap=True

Note: This script replaces all PyTorch (torch) functionality with JAX equivalents.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

# Import JAX transformer library (replaces: from model import GPTConfig, GPT)
from jax_transformer import GPTConfig, GPT, create_train_state, estimate_mfu
from jax_train_utils import (
    setup_jax_environment,
    get_dtype,
    check_device_availability,
    get_batch_jax,
    save_checkpoint,
    load_checkpoint,
    create_optimizer_with_schedule,
    estimate_loss_jax,
    setup_distributed_training,
)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = os.path.abspath('out_jax')
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2_jax'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# system
dtype = 'bfloat16'  # Will auto-detect if bfloat16 is supported
use_pmap = False  # Use pmap for data parallel training
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Setup JAX environment (replaces: torch.manual_seed, torch.backends.cuda settings)
print("Setting up JAX environment...")
rng = setup_jax_environment(seed=seed)

# Check device availability (replaces: torch.cuda.is_available())
device_info = check_device_availability()
print(f"JAX devices: {device_info['devices']}")
print(f"GPU available: {device_info['gpu_available']}")
print(f"Device count: {device_info['device_count']}")

# Auto-detect dtype (replaces: torch.cuda.is_bf16_supported())
if dtype == 'bfloat16' and not device_info['bf16_supported']:
    print("bfloat16 not supported, falling back to float16")
    dtype = 'float16'

compute_dtype = get_dtype(dtype)
print(f"Using dtype: {dtype}")

# Distributed training setup (replaces: DDP initialization)
dist_info = setup_distributed_training()
is_distributed = dist_info['is_distributed']
is_master = dist_info['is_master']
world_size = dist_info['world_size']
rank = dist_info['rank']

if is_distributed:
    print(f"Distributed training: rank {rank}/{world_size}")
    seed_offset = rank
    assert gradient_accumulation_steps % world_size == 0
    gradient_accumulation_steps //= world_size
else:
    print("Single device training")
    seed_offset = 0

tokens_per_iter = gradient_accumulation_steps * world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if is_master:
    os.makedirs(out_dir, exist_ok=True)

# Update RNG with seed offset
rng = jax.random.PRNGKey(seed + seed_offset)

# Data directory
data_dir = os.path.join('data', dataset)

# Init iteration counters
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

# Define make_eval_step before it's used
def make_eval_step(apply_fn):
    """Create a JIT-compiled eval step with the apply_fn baked in."""
    @jit
    def eval_step(params, inputs, targets):
        """Single evaluation step."""
        _, loss = apply_fn(
            {'params': params},
            inputs,
            targets=targets,
            train=False
        )
        return loss
    return eval_step

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    
    # Create training state (replaces: model = GPT(gptconf); optimizer = ...)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng,
        gptconf,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    
    # Print number of parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"number of parameters: {n_params/1e6:.2f}M")
    
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Load checkpoint (replaces: checkpoint = torch.load(...))
    
    # Create dummy state to restore into
    gptconf = GPTConfig(**model_args)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng,
        gptconf,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    
    # Restore from checkpoint
    state, metadata = load_checkpoint(out_dir, state)
    checkpoint_model_args = metadata['model_args']
    iter_num = metadata['iter_num']
    best_val_loss = metadata['best_val_loss']
    
    # Update model args from checkpoint
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    print(f"Resumed from iteration {iter_num}")

else:
    raise ValueError(f"init_from='{init_from}' not supported in JAX version (only 'scratch' and 'resume')")

# Initialize eval_step with the model's apply_fn
eval_step = make_eval_step(state.apply_fn)

# Create JIT-compiled training functions
# (replaces: model = torch.compile(model))
print("JIT compiling training functions...")

@jit
def train_step(state, inputs, targets, rng):
    """Single training step with gradient accumulation."""
    def loss_fn(params):
        _, loss = state.apply_fn(
            {'params': params},
            inputs,
            targets=targets,
            train=True,
            rngs={'dropout': rng} if dropout > 0 else None
        )
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, grads

@jit
def apply_gradients(state, grads):
    """Apply accumulated gradients."""
    return state.apply_gradients(grads=grads)

def estimate_loss():
    """Estimate loss on train and val sets."""
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y, rng_local = get_batch_jax(
                data_dir, split, batch_size, block_size, 
                jax.random.fold_in(rng, iter_num)
            )
            loss = eval_step(state.params, X, Y)
            losses.append(float(loss))
        out[split] = np.mean(losses)
    return out

# Learning rate schedule function
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if wandb_log and is_master:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
print("Starting training loop...")
rng, batch_rng = jax.random.split(rng)
X, Y, batch_rng = get_batch_jax(data_dir, 'train', batch_size, block_size, batch_rng)

t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    # Determine learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    
    # Evaluate and save checkpoint
    if iter_num % eval_interval == 0 and is_master:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                save_checkpoint(
                    out_dir,
                    state,
                    model_args,
                    iter_num,
                    best_val_loss,
                    config
                )
    
    if iter_num == 0 and eval_only:
        break
    
    # Forward backward update with gradient accumulation
    # (replaces: loss.backward(), scaler.scale(loss).backward())
    accumulated_grads = jax.tree.map(jnp.zeros_like, state.params)
    total_loss = 0.0
    
    for micro_step in range(gradient_accumulation_steps):
        # Forward and backward
        rng, step_rng = jax.random.split(rng)
        loss, grads = train_step(state, X, Y, step_rng)
        
        # Accumulate gradients
        accumulated_grads = jax.tree.map(
            lambda acc, g: acc + g / gradient_accumulation_steps,
            accumulated_grads,
            grads
        )
        total_loss += float(loss) / gradient_accumulation_steps
        
        # Fetch next batch asynchronously
        X, Y, batch_rng = get_batch_jax(data_dir, 'train', batch_size, block_size, batch_rng)
    
    # Clip gradients (replaces: torch.nn.utils.clip_grad_norm_)
    if grad_clip > 0.0:
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(accumulated_grads)))
        clip_factor = jnp.minimum(1.0, grad_clip / (grad_norm + 1e-6))
        accumulated_grads = jax.tree.map(lambda g: g * clip_factor, accumulated_grads)
    
    # Apply gradients (replaces: optimizer.step())
    state = apply_gradients(state, accumulated_grads)
    
    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0 and is_master:
        lossf = total_loss
        if local_iter_num >= 5:
            mfu = estimate_mfu(
                GPTConfig(**model_args),
                batch_size * gradient_accumulation_steps,
                dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    iter_num += 1
    local_iter_num += 1
    
    # Termination
    if iter_num > max_iters:
        break

print("Training complete!")
