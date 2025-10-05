from flax import nnx
import jax
import jax.numpy as jnp
import optax
from transformers import AutoConfig
from huggingface_hub import snapshot_download

from tx.models import Qwen3ForCausalLM
from tx.utils.models import get_dtype, load_checkpoint


def test_lora_training():
    base_model = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(base_model)
    config.max_lora_adapters = 5
    config.max_lora_rank = 32

    checkpoint_path = snapshot_download(base_model, allow_patterns=["*.safetensors"])
    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=get_dtype(config.dtype), rngs=nnx.Rngs(0))
        load_checkpoint(checkpoint_path, config, model)

        # Create optimizer that only targets LoRA A and B parameters
        def is_lora_param(path, value):
            return any(name in path for name in ['lora_A', 'lora_B'])

        optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=is_lora_param)

        # Create dummy input batch
        batch_size = 2
        seq_len = 10
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        adapter_indices = jnp.array([0, 1], dtype=jnp.int32)

        # Define loss function
        def loss_fn(model):
            outputs = model(input_ids, adapter_indices=adapter_indices)
            logits = outputs["logits"]
            return jnp.sum(logits)

        # Compute gradients - we need to use nnx.split to separate parameters
        # that we want to compute gradients for
        graphdef, lora_params, non_lora_params = nnx.split(model, is_lora_param, ...)

        # Compute gradients only for LoRA parameters
        def loss_for_lora(lora_params):
            merged_model = nnx.merge(graphdef, lora_params, non_lora_params)
            return loss_fn(merged_model)

        grad_fn = nnx.grad(loss_for_lora)
        lora_grads = grad_fn(lora_params)

        # Update model with gradients
        optimizer.update(lora_params, lora_grads)