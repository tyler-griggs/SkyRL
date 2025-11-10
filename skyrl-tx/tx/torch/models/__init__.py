"""PyTorch model implementations."""

from tx.torch.models.outputs import CausalLMOutput, ModelOutput
from tx.torch.models.qwen3 import Qwen3ForCausalLM

Qwen3MoeForCausalLM = Qwen3ForCausalLM

__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "CausalLMOutput",
    "ModelOutput",
]
