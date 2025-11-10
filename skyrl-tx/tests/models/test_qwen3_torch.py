import os
import tempfile

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from tx.models import Qwen3Config
from tx.torch.models.qwen3 import Qwen3ForCausalLM
from tx.torch.layers.lora import update_adapter_config

pytestmark = pytest.mark.torch  # Mark all tests in this file as torch tests


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_qwen3_torch_basic_shapes(device):
    """Test that the model initializes and produces correct output shapes."""
    base_config = PretrainedConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=False)
    
    model = Qwen3ForCausalLM(config).to(device)
    
    # Create dummy input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Check shapes
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert outputs.kv_cache is not None
    assert len(outputs.kv_cache.keys) == config.num_hidden_layers
    
    print("✓ Basic shapes test passed")


def test_qwen3_torch_vs_hf(device):
    """Test that our PyTorch implementation matches HuggingFace outputs."""
    model_name = "Qwen/Qwen3-0.6B"
    
    # Load HF model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", use_safetensors=True
    ).to(device)
    hf_model.eval()
    
    # Prepare input
    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    
    # Get HF outputs
    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
        )
    
    # Load our model
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=False)
    model = Qwen3ForCausalLM(config).to(device)
    
    # Copy weights from HF model
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(hf_model.model.embed_tokens.weight)
        for i, (our_layer, hf_layer) in enumerate(zip(model.model.layers, hf_model.model.layers)):
            # Copy attention weights
            our_layer.self_attn.q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
            our_layer.self_attn.k_proj.weight.copy_(hf_layer.self_attn.k_proj.weight)
            our_layer.self_attn.v_proj.weight.copy_(hf_layer.self_attn.v_proj.weight)
            our_layer.self_attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            our_layer.self_attn.q_norm.weight.copy_(hf_layer.self_attn.q_norm.weight)
            our_layer.self_attn.k_norm.weight.copy_(hf_layer.self_attn.k_norm.weight)
            
            # Copy MLP weights
            our_layer.mlp.gate_proj.weight.copy_(hf_layer.mlp.gate_proj.weight)
            our_layer.mlp.up_proj.weight.copy_(hf_layer.mlp.up_proj.weight)
            our_layer.mlp.down_proj.weight.copy_(hf_layer.mlp.down_proj.weight)
            
            # Copy layer norms
            our_layer.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)
            our_layer.post_attention_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
        
        model.model.norm.weight.copy_(hf_model.model.norm.weight)
        if not config.tie_word_embeddings:
            model.lm_head.weight.copy_(hf_model.lm_head.weight)
    
    model.eval()
    
    # Get our outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    # Compare outputs
    assert outputs.hidden_states is not None
    hf_hidden_states = hf_outputs.hidden_states
    our_hidden_states = outputs.hidden_states
    
    # Check first layer (after embedding)
    assert torch.allclose(hf_hidden_states[0], our_hidden_states[0], rtol=1e-4, atol=1e-4), \
        f"First hidden state mismatch: max diff = {(hf_hidden_states[0] - our_hidden_states[0]).abs().max()}"
    
    # Check middle layer
    mid_idx = len(hf_hidden_states) // 2
    assert torch.allclose(hf_hidden_states[mid_idx], our_hidden_states[mid_idx], rtol=1e-3, atol=1e-3), \
        f"Middle hidden state mismatch: max diff = {(hf_hidden_states[mid_idx] - our_hidden_states[mid_idx]).abs().max()}"
    
    # Check final layer
    assert torch.allclose(hf_hidden_states[-1], our_hidden_states[-1], rtol=1e-3, atol=1e-3), \
        f"Final hidden state mismatch: max diff = {(hf_hidden_states[-1] - our_hidden_states[-1]).abs().max()}"
    
    # Check logits
    assert torch.allclose(hf_outputs.logits, outputs.logits, rtol=1e-3, atol=1e-3), \
        f"Logits mismatch: max diff = {(hf_outputs.logits - outputs.logits).abs().max()}"
    
    print("✓ HuggingFace comparison test passed")


def test_qwen3_torch_lora_single_adapter(device):
    """Test single LoRA adapter by comparing with HuggingFace PEFT model."""
    base_model_name = "Qwen/Qwen3-0.6B"
    lora_adapter = "pcmoritz/qwen3-0.6b-lora-random"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    input_text = "The capital of France is"
    batch = tokenizer([input_text], return_tensors="pt", padding=True)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    
    # Create HF PEFT model
    lora_config = LoraConfig.from_pretrained(lora_adapter)
    lora_config.target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    hf_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, attn_implementation="eager", use_safetensors=True
    )
    hf_peft_model = get_peft_model(hf_base_model, lora_config)
    hf_peft_model.load_adapter(lora_adapter, adapter_name="default")
    hf_peft_model.to(device)
    hf_peft_model.eval()
    
    # Get HF PEFT output
    with torch.no_grad():
        hf_output = hf_peft_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
        )
    
    # Create our model with LoRA support
    base_config = PretrainedConfig.from_pretrained(base_model_name)
    config = Qwen3Config(
        base_config, 
        max_lora_adapters=1, 
        max_lora_rank=lora_config.r,
        shard_attention_heads=False
    )
    model = Qwen3ForCausalLM(config).to(device)
    
    # Copy base weights from HF model
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(hf_base_model.model.embed_tokens.weight)
        for i, (our_layer, hf_layer) in enumerate(zip(model.model.layers, hf_base_model.model.layers)):
            # Copy attention weights
            our_layer.self_attn.q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
            our_layer.self_attn.k_proj.weight.copy_(hf_layer.self_attn.k_proj.weight)
            our_layer.self_attn.v_proj.weight.copy_(hf_layer.self_attn.v_proj.weight)
            our_layer.self_attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            our_layer.self_attn.q_norm.weight.copy_(hf_layer.self_attn.q_norm.weight)
            our_layer.self_attn.k_norm.weight.copy_(hf_layer.self_attn.k_norm.weight)
            
            # Copy MLP weights
            our_layer.mlp.gate_proj.weight.copy_(hf_layer.mlp.gate_proj.weight)
            our_layer.mlp.up_proj.weight.copy_(hf_layer.mlp.up_proj.weight)
            our_layer.mlp.down_proj.weight.copy_(hf_layer.mlp.down_proj.weight)
            
            # Copy layer norms
            our_layer.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)
            our_layer.post_attention_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
        
        model.model.norm.weight.copy_(hf_base_model.model.norm.weight)
        if not config.tie_word_embeddings:
            model.lm_head.weight.copy_(hf_base_model.lm_head.weight)
    
    # Configure and load LoRA weights
    adapter_idx = 0
    update_adapter_config(model, adapter_idx, lora_config.r, lora_config.lora_alpha)
    
    with torch.no_grad():
        # Load LoRA weights for each layer
        for i, layer in enumerate(model.model.layers):
            hf_layer = hf_peft_model.base_model.model.model.layers[i]
            
            # Attention projections
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                hf_proj = getattr(hf_layer.self_attn, proj_name)
                our_proj = getattr(layer.self_attn, proj_name)
                our_proj.lora_A[adapter_idx].copy_(hf_proj.lora_A["default"].weight.T)
                our_proj.lora_B[adapter_idx].copy_(hf_proj.lora_B["default"].weight.T)
            
            # MLP projections
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                hf_proj = getattr(hf_layer.mlp, proj_name)
                our_proj = getattr(layer.mlp, proj_name)
                our_proj.lora_A[adapter_idx].copy_(hf_proj.lora_A["default"].weight.T)
                our_proj.lora_B[adapter_idx].copy_(hf_proj.lora_B["default"].weight.T)
    
    model.eval()
    
    # Get our output with LoRA adapter
    adapter_indices = torch.tensor([adapter_idx], device=device)
    with torch.no_grad():
        our_output = model(
            input_ids, 
            attention_mask=attention_mask, 
            adapter_indices=adapter_indices,
            output_hidden_states=True
        )
    
    # Compare outputs
    max_diff = (hf_output.logits - our_output.logits).abs().max().item()
    print(f"Max logits difference: {max_diff}")
    assert torch.allclose(hf_output.logits, our_output.logits, rtol=1e-3, atol=1e-3), \
        f"LoRA logits mismatch: max diff = {max_diff}"
    
    print("✓ Single LoRA adapter test passed")


def test_qwen3_torch_kv_cache(device):
    """Test that KV cache works correctly for generation."""
    model_name = "Qwen/Qwen3-0.6B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", use_safetensors=True
    ).to(device)
    hf_model.eval()
    
    # Prepare input
    input_text = "The capital of France is"
    batch = tokenizer([input_text], return_tensors="pt")
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    
    # Load our model
    base_config = PretrainedConfig.from_pretrained(model_name)
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=False)
    model = Qwen3ForCausalLM(config).to(device)
    
    # Copy weights
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(hf_model.model.embed_tokens.weight)
        for i, (our_layer, hf_layer) in enumerate(zip(model.model.layers, hf_model.model.layers)):
            our_layer.self_attn.q_proj.weight.copy_(hf_layer.self_attn.q_proj.weight)
            our_layer.self_attn.k_proj.weight.copy_(hf_layer.self_attn.k_proj.weight)
            our_layer.self_attn.v_proj.weight.copy_(hf_layer.self_attn.v_proj.weight)
            our_layer.self_attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            our_layer.self_attn.q_norm.weight.copy_(hf_layer.self_attn.q_norm.weight)
            our_layer.self_attn.k_norm.weight.copy_(hf_layer.self_attn.k_norm.weight)
            our_layer.mlp.gate_proj.weight.copy_(hf_layer.mlp.gate_proj.weight)
            our_layer.mlp.up_proj.weight.copy_(hf_layer.mlp.up_proj.weight)
            our_layer.mlp.down_proj.weight.copy_(hf_layer.mlp.down_proj.weight)
            our_layer.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)
            our_layer.post_attention_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
        model.model.norm.weight.copy_(hf_model.model.norm.weight)
        if not config.tie_word_embeddings:
            model.lm_head.weight.copy_(hf_model.lm_head.weight)
    
    model.eval()
    
    # Test 1: Prefill phase (no cache)
    with torch.no_grad():
        output_no_cache = model(input_ids, attention_mask=attention_mask)
    
    # Test 2: Using cache for next token
    from tx.torch.utils.generator import KVCache
    
    with torch.no_grad():
        # Prefill
        prefill_output = model(input_ids, attention_mask=attention_mask)
        kv_cache = prefill_output.kv_cache
        
        print(f"After prefill - cache_position: {kv_cache.cache_position}")
        print(f"After prefill - keys[0] shape: {kv_cache.keys[0].shape}")
        print(f"After prefill - values[0] shape: {kv_cache.values[0].shape}")
        
        # Pad cache to accommodate more tokens (e.g., 20 tokens total)
        max_length = 20
        kv_cache = kv_cache.pad_to_length(max_length)
        
        print(f"After padding - cache_position: {kv_cache.cache_position}")
        print(f"After padding - keys[0] shape: {kv_cache.keys[0].shape}")
        print(f"After padding - values[0] shape: {kv_cache.values[0].shape}")
        
        # Next token (simulate getting next token)
        next_token = output_no_cache.logits[:, -1:, :].argmax(dim=-1)
        print(f"next_token shape: {next_token.shape}")
        extended_attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=device)], dim=1)
        
        # Pad attention mask to match cache size
        mask_padding = max_length - extended_attention_mask.shape[1]
        if mask_padding > 0:
            extended_attention_mask = torch.cat([
                extended_attention_mask,
                torch.zeros(1, mask_padding, device=device)
            ], dim=1)
        
        print(f"extended_attention_mask shape: {extended_attention_mask.shape}")
        
        # Generate with cache
        cache_output = model(
            next_token,
            attention_mask=extended_attention_mask,
            kv_cache=kv_cache
        )
    
    # The cache output should be valid (no NaNs)
    assert not torch.isnan(cache_output.logits).any(), "KV cache produced NaN values"
    assert cache_output.kv_cache.cache_position == input_ids.shape[1] + 1, \
        f"Cache position should be {input_ids.shape[1] + 1}, got {cache_output.kv_cache.cache_position}"
    
    print("✓ KV cache test passed")


if __name__ == "__main__":
    # Run tests directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on device: {device}")
    
    print("\n1. Testing basic shapes...")
    test_qwen3_torch_basic_shapes(device)
    
    print("\n2. Testing vs HuggingFace...")
    test_qwen3_torch_vs_hf(device)
    
    print("\n3. Testing single LoRA adapter...")
    test_qwen3_torch_lora_single_adapter(device)
    
    print("\n4. Testing KV cache...")
    test_qwen3_torch_kv_cache(device)
    
    print("\n✓ All tests passed!")

