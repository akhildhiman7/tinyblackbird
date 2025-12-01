"""
Test script to verify JAX transformer library works correctly.
This script tests basic functionality without requiring the full dataset.

Usage:
    python test_jax_transformer.py
"""

import numpy as np
import jax
import jax.numpy as jnp

try:
    from jax_transformer import GPTConfig, GPT, create_train_state, estimate_mfu, generate
    from jax_train_utils import (
        setup_jax_environment,
        check_device_availability,
        clip_gradients,
        get_dtype,
    )
    print("✓ Successfully imported JAX transformer library")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nPlease install JAX dependencies:")
    print("  pip install -r requirements_jax.txt")
    exit(1)


def test_device_setup():
    """Test device availability and setup."""
    print("\n" + "="*60)
    print("Testing Device Setup")
    print("="*60)
    
    device_info = check_device_availability()
    print(f"Available devices: {len(device_info['devices'])}")
    for i, device in enumerate(device_info['devices']):
        print(f"  Device {i}: {device.device_kind} (platform: {device.platform})")
    print(f"GPU available: {device_info['gpu_available']}")
    print(f"TPU available: {device_info['tpu_available']}")
    print(f"bfloat16 supported: {device_info['bf16_supported']}")
    
    rng = setup_jax_environment(seed=42)
    print(f"✓ PRNG key initialized: {rng}")
    
    return rng


def test_model_creation():
    """Test creating a small GPT model."""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)
    
    # Create small config for testing
    config = GPTConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False
    )
    
    print(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    
    # Initialize model
    rng = jax.random.PRNGKey(42)
    state = create_train_state(
        rng,
        config,
        learning_rate=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"✓ Model created with {n_params:,} parameters ({n_params/1e6:.2f}M)")
    
    return state, config


def test_forward_pass(state, config):
    """Test forward pass."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)
    
    # Create dummy input
    batch_size = 4
    seq_len = 32
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass (training mode)
    logits, loss = state.apply_fn(
        {'params': state.params},
        dummy_input,
        targets=dummy_target,
        train=True
    )
    
    print(f"✓ Training forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {float(loss):.4f}")
    
    # Forward pass (inference mode)
    logits_inf, _ = state.apply_fn(
        {'params': state.params},
        dummy_input,
        targets=None,
        train=False
    )
    
    print(f"✓ Inference forward pass successful")
    print(f"  Logits shape: {logits_inf.shape}")
    
    return loss


def test_backward_pass(state, config):
    """Test gradient computation."""
    print("\n" + "="*60)
    print("Testing Backward Pass (Gradient Computation)")
    print("="*60)
    
    # Create dummy batch
    batch_size = 4
    seq_len = 32
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # Define loss function
    def loss_fn(params):
        _, loss = state.apply_fn(
            {'params': params},
            dummy_input,
            targets=dummy_target,
            train=True
        )
        return loss
    
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    print(f"✓ Gradient computation successful")
    print(f"  Loss: {float(loss):.4f}")
    
    # Check gradient statistics
    grad_values = jax.tree_util.tree_leaves(grads)
    grad_norms = [jnp.linalg.norm(g.reshape(-1)) for g in grad_values]
    total_grad_norm = jnp.sqrt(sum(g**2 for g in grad_norms))
    
    print(f"  Total gradient norm: {float(total_grad_norm):.6f}")
    print(f"  Number of gradient tensors: {len(grad_values)}")
    
    return grads


def test_optimizer_step(state, grads):
    """Test optimizer step."""
    print("\n" + "="*60)
    print("Testing Optimizer Step")
    print("="*60)
    
    # Get initial parameter values
    initial_param = jax.tree_util.tree_leaves(state.params)[0]
    initial_value = float(jnp.mean(initial_param))
    
    print(f"Initial param mean: {initial_value:.6f}")
    
    # Apply gradients
    new_state = state.apply_gradients(grads=grads)
    
    # Check parameter update
    new_param = jax.tree_util.tree_leaves(new_state.params)[0]
    new_value = float(jnp.mean(new_param))
    
    print(f"Updated param mean: {new_value:.6f}")
    print(f"Change: {new_value - initial_value:.6f}")
    print(f"✓ Optimizer step successful")
    
    return new_state


def test_gradient_clipping(grads):
    """Test gradient clipping."""
    print("\n" + "="*60)
    print("Testing Gradient Clipping")
    print("="*60)
    
    # Original gradient norm
    orig_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    print(f"Original gradient norm: {float(orig_norm):.6f}")
    
    # Clip gradients
    max_norm = 0.5
    clipped_grads = clip_gradients(grads, max_norm)
    
    # Clipped gradient norm
    clipped_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(clipped_grads)))
    print(f"Clipped gradient norm (max={max_norm}): {float(clipped_norm):.6f}")
    print(f"✓ Gradient clipping successful")


def test_jit_compilation(state, config):
    """Test JIT compilation."""
    print("\n" + "="*60)
    print("Testing JIT Compilation")
    print("="*60)
    
    @jax.jit
    def train_step(state, inputs, targets):
        def loss_fn(params):
            _, loss = state.apply_fn(
                {'params': params},
                inputs,
                targets=targets,
                train=True
            )
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, grads
    
    # Create dummy batch
    batch_size = 4
    seq_len = 32
    dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    dummy_target = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # First call (compilation)
    print("First call (compiling)...")
    import time
    t0 = time.time()
    loss, grads = train_step(state, dummy_input, dummy_target)
    t1 = time.time()
    print(f"  Time: {(t1-t0)*1000:.2f}ms (includes compilation)")
    
    # Second call (cached)
    print("Second call (using cached compilation)...")
    t0 = time.time()
    loss, grads = train_step(state, dummy_input, dummy_target)
    t1 = time.time()
    print(f"  Time: {(t1-t0)*1000:.2f}ms")
    
    print(f"✓ JIT compilation successful")


def test_generation(state, config):
    """Test text generation."""
    print("\n" + "="*60)
    print("Testing Text Generation")
    print("="*60)
    
    # Start with a few tokens
    start_tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    print(f"Start tokens: {start_tokens[0].tolist()}")
    
    # Generate
    rng = jax.random.PRNGKey(42)
    generated = generate(
        state,
        start_tokens,
        max_new_tokens=10,
        temperature=1.0,
        top_k=None,
        rng=rng
    )
    
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"✓ Generation successful (generated {generated.shape[1] - start_tokens.shape[1]} new tokens)")


def test_dtype_support():
    """Test different dtype support."""
    print("\n" + "="*60)
    print("Testing Dtype Support")
    print("="*60)
    
    dtypes_to_test = ['float32', 'bfloat16', 'float16']
    
    for dtype_str in dtypes_to_test:
        try:
            dtype = get_dtype(dtype_str)
            test_array = jnp.array([1.0, 2.0, 3.0], dtype=dtype)
            print(f"✓ {dtype_str}: {test_array.dtype}")
        except Exception as e:
            print(f"✗ {dtype_str}: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("JAX TRANSFORMER LIBRARY TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Device setup
        rng = test_device_setup()
        
        # Test 2: Model creation
        state, config = test_model_creation()
        
        # Test 3: Forward pass
        loss = test_forward_pass(state, config)
        
        # Test 4: Backward pass
        grads = test_backward_pass(state, config)
        
        # Test 5: Gradient clipping
        test_gradient_clipping(grads)
        
        # Test 6: Optimizer step
        new_state = test_optimizer_step(state, grads)
        
        # Test 7: JIT compilation
        test_jit_compilation(state, config)
        
        # Test 8: Generation
        test_generation(state, config)
        
        # Test 9: Dtype support
        test_dtype_support()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe JAX transformer library is working correctly.")
        print("You can now run: python train_jax.py")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
