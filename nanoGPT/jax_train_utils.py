"""
JAX Training Utilities
Provides helper functions for training, checkpointing, data loading, and distributed training.
This replaces PyTorch-specific utilities from train.py
"""

import os
import pickle
from typing import Tuple, Dict, Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import serialization
import optax


def setup_jax_environment(seed: int = 1337, enable_x64: bool = False):
    """
    Setup JAX environment (equivalent to torch seed and backend settings).
    
    Args:
        seed: Random seed
        enable_x64: Whether to enable 64-bit precision (default False for performance)
    """
    # Enable/disable 64-bit precision
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    
    # Set random seed (JAX uses functional PRNG, so we just return a key)
    rng = jax.random.PRNGKey(seed)
    return rng


def get_dtype(dtype_str: str):
    """
    Convert dtype string to JAX dtype.
    
    Args:
        dtype_str: One of 'float32', 'bfloat16', 'float16'
    
    Returns:
        JAX dtype
    """
    dtype_map = {
        'float32': jnp.float32,
        'bfloat16': jnp.bfloat16,
        'float16': jnp.float16,
    }
    return dtype_map.get(dtype_str, jnp.float32)


def check_device_availability():
    """
    Check available JAX devices (replaces torch.cuda.is_available()).
    
    Returns:
        Dict with device info
    """
    devices = jax.devices()
    gpu_available = any(d.platform == 'gpu' for d in devices)
    tpu_available = any(d.platform == 'tpu' for d in devices)
    
    # Check if bfloat16 is supported (TPUs and modern GPUs support it)
    bf16_supported = gpu_available or tpu_available
    
    return {
        'devices': devices,
        'gpu_available': gpu_available,
        'tpu_available': tpu_available,
        'bf16_supported': bf16_supported,
        'device_count': len(devices),
        'default_device': devices[0] if devices else None
    }


def get_batch_jax(
    data_dir: str,
    split: str,
    batch_size: int,
    block_size: int,
    rng: jax.random.PRNGKey,
    device: Optional[Any] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load a batch of data (replaces get_batch from train.py).
    
    Args:
        data_dir: Path to data directory
        split: 'train' or 'val'
        batch_size: Batch size
        block_size: Context length
        rng: JAX PRNG key for random sampling
        device: JAX device to put data on (optional)
    
    Returns:
        Tuple of (inputs, targets) as JAX arrays
    """
    # Load data from memmap
    filename = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, filename), dtype=np.uint16, mode='r')
    
    # Random sampling
    rng, sample_rng = jax.random.split(rng)
    ix = jax.random.randint(sample_rng, (batch_size,), 0, len(data) - block_size)
    ix = np.array(ix)  # Convert to numpy for indexing
    
    # Create batch
    x = np.stack([data[i:i+block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int64) for i in ix])
    
    # Convert to JAX arrays and move to device
    x = jnp.array(x)
    y = jnp.array(y)
    
    if device is not None:
        x = jax.device_put(x, device)
        y = jax.device_put(y, device)
    
    return x, y, rng


def clip_gradients(grads, max_norm: float):
    """
    Clip gradients by global norm (replaces torch.nn.utils.clip_grad_norm_).
    
    Args:
        grads: Gradient pytree
        max_norm: Maximum gradient norm
    
    Returns:
        Clipped gradients
    """
    if max_norm <= 0.0:
        return grads
    
    # Calculate global norm
    global_norm = optax.global_norm(grads)
    
    # Clip
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    clipped_grads = jax.tree_map(lambda g: g * clip_factor, grads)
    
    return clipped_grads


def save_checkpoint(
    checkpoint_dir: str,
    state,
    model_args: Dict[str, Any],
    iter_num: int,
    best_val_loss: float,
    config: Dict[str, Any]
):
    """
    Save training checkpoint (replaces torch.save).
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        state: Training state (Flax TrainState)
        model_args: Model configuration dict
        iter_num: Current iteration number
        best_val_loss: Best validation loss so far
        config: Training configuration
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model parameters and optimizer state
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=iter_num,
        overwrite=True,
        keep=3  # Keep last 3 checkpoints
    )
    
    # Save metadata
    metadata = {
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    metadata_path = os.path.join(checkpoint_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


def load_checkpoint(
    checkpoint_dir: str,
    state
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load training checkpoint (replaces torch.load).
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        state: Training state template to restore into
    
    Returns:
        Tuple of (restored_state, metadata)
    """
    # Restore state
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state
    )
    
    # Load metadata
    metadata_path = os.path.join(checkpoint_dir, 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return restored_state, metadata


def create_learning_rate_schedule(
    learning_rate: float,
    warmup_iters: int,
    lr_decay_iters: int,
    min_lr: float
):
    """
    Create learning rate schedule with warmup and cosine decay.
    
    Args:
        learning_rate: Peak learning rate
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Number of decay iterations
        min_lr: Minimum learning rate
    
    Returns:
        Learning rate schedule function
    """
    def schedule(step):
        # Linear warmup
        if step < warmup_iters:
            return learning_rate * (step + 1) / (warmup_iters + 1)
        # Constant at minimum after decay
        if step > lr_decay_iters:
            return min_lr
        # Cosine decay
        decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    
    return schedule


def create_optimizer_with_schedule(
    learning_rate: float,
    weight_decay: float,
    betas: Tuple[float, float],
    warmup_iters: int,
    lr_decay_iters: int,
    min_lr: float,
    grad_clip: float = 0.0
):
    """
    Create optimizer with learning rate schedule and gradient clipping.
    
    Args:
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        warmup_iters: Warmup iterations
        lr_decay_iters: Decay iterations
        min_lr: Minimum learning rate
        grad_clip: Gradient clipping value (0 to disable)
    
    Returns:
        Optax optimizer
    """
    # Create learning rate schedule
    schedule = create_learning_rate_schedule(
        learning_rate, warmup_iters, lr_decay_iters, min_lr
    )
    schedule_fn = optax.join_schedules(
        schedules=[schedule],
        boundaries=[]
    )
    
    # Build optimizer chain
    optimizer_chain = []
    
    # Gradient clipping
    if grad_clip > 0.0:
        optimizer_chain.append(optax.clip_by_global_norm(grad_clip))
    
    # AdamW optimizer with schedule
    optimizer_chain.append(
        optax.adamw(
            learning_rate=schedule_fn,
            b1=betas[0],
            b2=betas[1],
            weight_decay=weight_decay
        )
    )
    
    return optax.chain(*optimizer_chain)


def setup_distributed_training():
    """
    Setup distributed training environment (replaces DDP setup).
    
    Returns:
        Dict with distributed training info
    """
    # Check for distributed environment
    world_size = jax.process_count()
    rank = jax.process_index()
    local_rank = jax.local_device_count()
    
    is_distributed = world_size > 1
    is_master = rank == 0
    
    return {
        'is_distributed': is_distributed,
        'world_size': world_size,
        'rank': rank,
        'local_rank': local_rank,
        'is_master': is_master,
        'devices': jax.local_devices()
    }


def tree_zeros_like(tree):
    """Create a tree of zeros with the same structure."""
    return jax.tree_map(jnp.zeros_like, tree)


def pmap_train_step(state, batch, dropout_rng):
    """
    Training step for use with pmap (data parallel training).
    
    Args:
        state: Training state
        batch: Tuple of (inputs, targets)
        dropout_rng: PRNG key for dropout
    
    Returns:
        Tuple of (new_state, loss)
    """
    inputs, targets = batch
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)
    
    def loss_fn(params):
        logits, loss = state.apply_fn(
            {'params': params},
            inputs,
            targets=targets,
            train=True,
            rngs={'dropout': dropout_rng}
        )
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='device')
    loss = jax.lax.pmean(loss, axis_name='device')
    
    # Update state
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss


def estimate_loss_jax(
    state,
    data_dir: str,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    rng: jax.random.PRNGKey,
    device: Optional[Any] = None
) -> Dict[str, float]:
    """
    Estimate loss on train and val sets (replaces @torch.no_grad() estimate_loss).
    
    Args:
        state: Training state
        data_dir: Path to data directory
        batch_size: Batch size
        block_size: Context length
        eval_iters: Number of iterations to average over
        rng: PRNG key
        device: Device to run on
    
    Returns:
        Dict with 'train' and 'val' losses
    """
    out = {}
    
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y, rng = get_batch_jax(data_dir, split, batch_size, block_size, rng, device)
            _, loss = state.apply_fn({'params': state.params}, X, targets=Y, train=False)
            losses.append(float(loss))
        out[split] = np.mean(losses)
    
    return out
