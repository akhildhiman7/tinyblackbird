"""
JAX/Flax implementation of GPT Language Model.
This is a drop-in replacement for model.py using JAX instead of PyTorch.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""
    ndim: int
    bias: bool
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        
        weight = self.param('weight', nn.initializers.ones, (self.ndim,))
        x_norm = x_norm * weight
        
        if self.bias:
            bias = self.param('bias', nn.initializers.zeros, (self.ndim,))
            x_norm = x_norm + bias
        
        return x_norm


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train: bool = True):
        B, T, C = x.shape
        assert C == self.config.n_embd
        
        # QKV projections
        c_attn = nn.Dense(
            3 * self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name='c_attn'
        )
        qkv = c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        n_head = self.config.n_head
        head_dim = C // n_head
        q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        
        # Causal self-attention
        # Create causal mask
        mask = jnp.tril(jnp.ones((T, T))).reshape(1, 1, T, T)
        
        # Attention scores
        att = (q @ jnp.swapaxes(k, -2, -1)) * (1.0 / math.sqrt(head_dim))
        att = jnp.where(mask == 0, float('-inf'), att)
        att = jax.nn.softmax(att, axis=-1)
        
        if train and self.config.dropout > 0:
            att = nn.Dropout(rate=self.config.dropout, deterministic=not train)(att)
        
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        c_proj = nn.Dense(
            self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name='c_proj'
        )
        y = c_proj(y)
        
        if train and self.config.dropout > 0:
            y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)
        
        return y


class MLP(nn.Module):
    """MLP block."""
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train: bool = True):
        c_fc = nn.Dense(
            4 * self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name='c_fc'
        )
        x = c_fc(x)
        x = nn.gelu(x)
        
        c_proj = nn.Dense(
            self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name='c_proj'
        )
        x = c_proj(x)
        
        if train and self.config.dropout > 0:
            x = nn.Dropout(rate=self.config.dropout, deterministic=not train)(x)
        
        return x


class Block(nn.Module):
    """Transformer block."""
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train: bool = True):
        ln_1 = LayerNorm(ndim=self.config.n_embd, bias=self.config.bias, name='ln_1')
        attn = CausalSelfAttention(config=self.config, name='attn')
        ln_2 = LayerNorm(ndim=self.config.n_embd, bias=self.config.bias, name='ln_2')
        mlp = MLP(config=self.config, name='mlp')
        
        x = x + attn(ln_1(x), train=train)
        x = x + mlp(ln_2(x), train=train)
        return x


class GPT(nn.Module):
    """GPT Language Model."""
    config: GPTConfig

    @nn.compact
    def __call__(self, idx, targets=None, train: bool = True):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Token and position embeddings
        wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name='wte'
        )
        wpe = nn.Embed(
            num_embeddings=self.config.block_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name='wpe'
        )
        
        pos = jnp.arange(0, T)
        tok_emb = wte(idx)
        pos_emb = wpe(pos)
        x = tok_emb + pos_emb
        
        if train and self.config.dropout > 0:
            x = nn.Dropout(rate=self.config.dropout, deterministic=not train)(x)
        
        # Transformer blocks
        for i in range(self.config.n_layer):
            block = Block(config=self.config, name=f'h_{i}')
            x = block(x, train=train)
        
        # Final layer norm
        ln_f = LayerNorm(ndim=self.config.n_embd, bias=self.config.bias, name='ln_f')
        x = ln_f(x)
        
        # Language model head
        # Note: Weight tying - we'll handle this in the initialization
        lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            name='lm_head'
        )
        
        if targets is not None:
            # Training: compute logits for all positions and loss
            logits = lm_head(x)
            # Cross entropy loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()
        else:
            # Inference: only compute logits for last position
            logits = lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    def get_num_params(self, params):
        """Return the number of parameters in the model."""
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        # Subtract position embeddings for non-embedding count
        if 'wpe' in params:
            n_params -= params['wpe']['embedding'].size
        return n_params


def create_train_state(rng, config: GPTConfig, learning_rate: float, weight_decay: float, betas: Tuple[float, float]):
    """Create initial training state."""
    model = GPT(config=config)
    
    # Initialize with dummy input
    dummy_input = jnp.ones((1, config.block_size), dtype=jnp.int32)
    variables = model.init(rng, dummy_input, train=False)
    params = variables['params']
    
    # Apply scaled initialization to residual projections (c_proj weights)
    def init_c_proj_weights(params):
        """Apply GPT-2 style scaled initialization to c_proj layers."""
        def scale_c_proj(path, param):
            # Check if this is a c_proj weight
            if 'c_proj' in path and 'kernel' in path:
                std = 0.02 / math.sqrt(2 * config.n_layer)
                return jax.random.normal(jax.random.PRNGKey(0), param.shape) * std
            return param
        
        return jax.tree_util.tree_map_with_path(
            lambda path, x: scale_c_proj('/'.join(str(k.key) for k in path), x),
            params
        )
    
    params = init_c_proj_weights(params)
    
    # Create optimizer with weight decay
    # AdamW: separate weight decay from gradient-based optimization
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        b1=betas[0],
        b2=betas[1],
        weight_decay=weight_decay
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state


def generate(state, idx, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, rng=None):
    """
    Generate new tokens autoregressively.
    
    Args:
        state: Training state with model parameters
        idx: Input token indices of shape (B, T)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top k tokens
        rng: PRNG key for sampling
    
    Returns:
        Generated token indices of shape (B, T + max_new_tokens)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    config = state.apply_fn.__self__.config if hasattr(state.apply_fn, '__self__') else None
    
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.shape[1] <= config.block_size else idx[:, -config.block_size:]
        
        # Forward pass
        logits, _ = state.apply_fn({'params': state.params}, idx_cond, train=False)
        logits = logits[:, -1, :] / temperature
        
        # Optional top-k filtering
        if top_k is not None:
            top_k_actual = min(top_k, logits.shape[-1])
            # Get top-k values and zero out the rest
            top_vals, _ = jax.lax.top_k(logits, top_k_actual)
            min_top_k = top_vals[:, -1:]
            logits = jnp.where(logits < min_top_k, float('-inf'), logits)
        
        # Sample from distribution
        rng, sample_rng = jax.random.split(rng)
        idx_next = jax.random.categorical(sample_rng, logits, axis=-1)
        idx_next = idx_next[:, None]  # Add time dimension
        
        # Append to sequence
        idx = jnp.concatenate([idx, idx_next], axis=1)
    
    return idx


def estimate_mfu(config: GPTConfig, fwdbwd_per_iter: int, dt: float) -> float:
    """
    Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.
    
    Args:
        config: Model configuration
        fwdbwd_per_iter: Number of forward-backward passes per iteration
        dt: Time taken for iteration in seconds
    
    Returns:
        MFU as a ratio (0 to 1)
    """
    # Calculate number of parameters (approximate)
    N = config.n_layer * (
        12 * config.n_embd**2 +  # Attention
        13 * config.n_embd**2     # MLP
    )
    
    L, H, Q, T = config.n_layer, config.n_head, config.n_embd // config.n_head, config.block_size
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    
    flops_achieved = flops_per_iter * (1.0 / dt)
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    
    return mfu
