"""
Sample from a trained JAX/Flax model.
This is a JAX reimplementation of sample.py - no PyTorch dependencies.
"""
import os
import pickle

import jax
import jax.numpy as jnp
from flax.training import checkpoints

from jax_transformer import GPTConfig, GPT, create_train_state

# Try to import tiktoken (optional, only needed for GPT-2 style tokenization)
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# -----------------------------------------------------------------------------
# Configuration
init_from = 'resume'  # 'resume' from out_dir
out_dir = 'out_jax'  # checkpoint directory (will be converted to absolute path)
start = "KING RICHARD II: What do you think of this my lord?"  # prompt string, or "FILE:prompt.txt" to load from file
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens to generate per sample
temperature = 0.8  # sampling temperature (1.0 = no change, < 1.0 = less random)
top_k = 200  # retain only top_k most likely tokens (None to disable)
seed = 1337
# -----------------------------------------------------------------------------
# Override config from command line
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------
# Convert out_dir to absolute path AFTER config overrides
out_dir = os.path.abspath(out_dir)
# -----------------------------------------------------------------------------


def generate(params, apply_fn, idx, max_new_tokens, block_size, temperature=1.0, top_k=None, rng=None):
    """
    Generate new tokens autoregressively.
    
    Args:
        params: Model parameters
        apply_fn: Model apply function
        idx: Input token indices of shape (B, T)
        max_new_tokens: Number of tokens to generate
        block_size: Maximum context length
        temperature: Sampling temperature
        top_k: If set, only sample from top k tokens
        rng: PRNG key for sampling
    
    Returns:
        Generated token indices of shape (B, T + max_new_tokens)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    for _ in range(max_new_tokens):
        # Crop context if needed (keep last block_size tokens)
        idx_cond = idx if idx.shape[1] <= block_size else idx[:, -block_size:]
        
        # Forward pass (inference mode, no targets)
        logits, _ = apply_fn({'params': params}, idx_cond, targets=None, train=False)
        
        # Get logits for the last position and apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Optional top-k filtering
        if top_k is not None:
            top_k_actual = min(top_k, logits.shape[-1])
            # Get top-k values
            top_vals, _ = jax.lax.top_k(logits, top_k_actual)
            min_top_k = top_vals[:, -1:]
            # Zero out tokens below top-k threshold
            logits = jnp.where(logits < min_top_k, float('-inf'), logits)
        
        # Sample from distribution
        rng, sample_rng = jax.random.split(rng)
        idx_next = jax.random.categorical(sample_rng, logits, axis=-1)
        idx_next = idx_next[:, None]  # Add sequence dimension (B, 1)
        
        # Append to sequence
        idx = jnp.concatenate([idx, idx_next], axis=1)
    
    return idx


def main():
    print("Setting up JAX environment...")
    rng = jax.random.PRNGKey(seed)
    
    # Print device info
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    
    # Load model
    if init_from == 'resume':
        print(f"Loading model from {out_dir}...")
        
        # Load metadata to get model args
        metadata_path = os.path.join(out_dir, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata.pkl found in {out_dir}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        model_args = metadata['model_args']
        iter_num = metadata.get('iter_num', 0)
        print(f"Model args: {model_args}")
        print(f"Checkpoint from iteration: {iter_num}")
        
        # Create model config
        gptconf = GPTConfig(**model_args)
        
        # Create a dummy training state to restore into
        rng, init_rng = jax.random.split(rng)
        state = create_train_state(
            init_rng,
            gptconf,
            learning_rate=1e-4,  # Doesn't matter for inference
            weight_decay=0.0,
            betas=(0.9, 0.95)
        )
        
        # Restore checkpoint
        state = checkpoints.restore_checkpoint(
            ckpt_dir=out_dir,
            target=state
        )
        
        params = state.params
        apply_fn = state.apply_fn
        block_size = gptconf.block_size
        
        # Get dataset name for loading encoder
        config = metadata.get('config', {})
        dataset = config.get('dataset', None)
        
        print(f"Model loaded successfully!")
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Number of parameters: {n_params/1e6:.2f}M")
    else:
        raise ValueError(f"init_from='{init_from}' not supported. Use 'resume'.")
    
    # Setup encoder/decoder
    load_meta = False
    if dataset is not None:
        meta_path = os.path.join('data', dataset, 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # Default to GPT-2 encodings (requires tiktoken)
        if not HAS_TIKTOKEN:
            raise ImportError(
                "No meta.pkl found and tiktoken is not installed. "
                "Install with: pip install tiktoken"
            )
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    # Encode the prompt
    prompt = start
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    
    start_ids = encode(prompt)
    x = jnp.array(start_ids, dtype=jnp.int32)[None, ...]  # Add batch dimension (1, T)
    
    print(f"\nPrompt: {repr(prompt)}")
    print(f"Generating {num_samples} samples with {max_new_tokens} tokens each...")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print("=" * 50)
    
    # Generate samples
    for k in range(num_samples):
        rng, sample_rng = jax.random.split(rng)
        
        y = generate(
            params=params,
            apply_fn=apply_fn,
            idx=x,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature,
            top_k=top_k,
            rng=sample_rng
        )
        
        # Decode and print
        generated_tokens = y[0].tolist()
        generated_text = decode(generated_tokens)
        print(generated_text)
        print('---------------')


if __name__ == '__main__':
    main()
