from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_routing(
    x_flat: torch.Tensor,  # [N, in_features]
    adapter_indices_flat: torch.Tensor,  # [N], long
    max_lora_adapters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for adapter-specific routing:
      - Sort tokens by adapter id
      - Return contiguous group sizes per adapter
      - Provide inverse permutation to restore original order

    Returns:
      x_sorted:               [N, in_features]
      group_sizes:            [max_lora_adapters]
      unsort_indices:         [N]
      adapter_indices_sorted: [N]
    """
    if adapter_indices_flat.dtype != torch.long:
        adapter_indices_flat = adapter_indices_flat.long()
    if (adapter_indices_flat < 0).any() or (adapter_indices_flat >= max_lora_adapters).any():
        raise ValueError("adapter_indices contain out-of-range values.")

    sort_idx = torch.argsort(adapter_indices_flat)  # ascending by adapter id
    x_sorted = x_flat.index_select(0, sort_idx)
    adapter_indices_sorted = adapter_indices_flat.index_select(0, sort_idx)

    group_sizes = torch.bincount(adapter_indices_sorted, minlength=max_lora_adapters)
    if group_sizes.numel() != max_lora_adapters:
        group_sizes = F.pad(group_sizes, (0, max_lora_adapters - group_sizes.numel()))

    # inverse permutation to restore original token order
    unsort_indices = torch.empty_like(sort_idx)
    unsort_indices[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
    return x_sorted, group_sizes, unsort_indices, adapter_indices_sorted


class LoRAMixin(nn.Module):
    """A mixin for PyTorch modules to add multi-adapter LoRA support.

    This mixin adds LoRA parameters (lora_A, lora_B) and methods to apply
    the low-rank adaptation to a base module's output.

    Provides:
      - init_lora(...)  -> allocate lora_A, lora_B, lora_scaling, lora_ranks
      - apply_lora(x, base_output, adapter_indices)  -> apply LoRA adaptation

    Stored tensors (when enabled):
      lora_A:      [A, in_features, r_max]   (Param, He-uniform)
      lora_B:      [A, r_max, out_features]  (Param, zeros)
      lora_scaling:[A]                       (Buffer, alpha/rank per adapter)
      lora_ranks:  [A]                       (Buffer, int rank per adapter)
    """

    lora_scaling: torch.Tensor | None
    lora_ranks: torch.Tensor | None
    lora_A: nn.Parameter | None
    lora_B: nn.Parameter | None

    def init_lora(
        self,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shape_A: tuple[int, ...],
        shape_B: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        self.max_lora_adapters = int(max_lora_adapters)
        self.max_lora_rank = int(max_lora_rank)

        if self.max_lora_adapters == 0:
            self.lora_scaling = None
            self.lora_ranks = None
            self.lora_A = None
            self.lora_B = None
            return

        if device is None:
            device = torch.device("cpu")

        self.register_buffer(
            "lora_scaling",
            torch.full((self.max_lora_adapters,), 1.0, dtype=dtype, device=device),
            persistent=True,
        )
        self.register_buffer(
            "lora_ranks",
            torch.full((self.max_lora_adapters,), self.max_lora_rank, dtype=torch.int32, device=device),
            persistent=True,
        )

        A = torch.empty(*shape_A, dtype=dtype, device=device)
        B = torch.zeros(*shape_B, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))  # He-uniform A
        self.lora_A = nn.Parameter(A, requires_grad=True)
        self.lora_B = nn.Parameter(B, requires_grad=True)

    def apply_lora(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        adapter_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply multi-adapter LoRA to base module output.

        Args:
          x: Input tensor [B, T, in_features]
          base_output: Base module output [B, T, out_features]
          adapter_indices: Adapter index per batch element [B], broadcasted over sequence length

        Returns:
          base_output + lora_output with per-adapter routing and scaling
        """
        if self.max_lora_adapters == 0 or adapter_indices is None:
            return base_output

        if x.dim() != 3:
            raise ValueError("x must be [B, T, in_features].")
        B, T, in_features = x.shape
        if adapter_indices.dim() != 1 or adapter_indices.size(0) != B:
            raise ValueError("adapter_indices must be shape [B].")

        # Flatten tokens to [N, in]
        x_flat = x.reshape(-1, in_features)  # [B*T, in]
        # Broadcast adapter ids across sequence length
        adapters_flat = adapter_indices.repeat_interleave(T)  # [B*T]

        # Route by adapter (ragged groups)
        x_sorted, group_sizes, unsort_idx, adapters_sorted = _prepare_routing(
            x_flat, adapters_flat, self.max_lora_adapters
        )

        # Compute LoRA: (x @ A) @ B per-adapter group
        N = x_sorted.size(0)
        out_features = base_output.size(-1)
        y_sorted = torch.empty(N, out_features, dtype=base_output.dtype, device=base_output.device)

        offset = 0
        for a, n in enumerate(group_sizes.tolist()):
            if n == 0:
                continue
            s, e = offset, offset + n
            xa = x_sorted[s:e]  # [n, in]
            r = int(self.lora_ranks[a].item())
            if r > 0:
                Aa = self.lora_A[a, :, :r]  # [in, r]
                Ba = self.lora_B[a, :r, :]  # [r, out]
                inter = xa.matmul(Aa)  # [n, r]
                ya = inter.matmul(Ba)  # [n, out]
            else:
                ya = torch.zeros(n, out_features, dtype=y_sorted.dtype, device=y_sorted.device)
            y_sorted[s:e] = ya
            offset = e

        # Unsort back to original token order -> [B*T, out]
        y_flat = y_sorted.index_select(0, unsort_idx)

        # Reshape and scale: lora_output * self.lora_scaling[adapter_indices, None, None]
        y = y_flat.view(B, T, out_features)
        y = y * self.lora_scaling[adapter_indices].view(B, 1, 1)

        return base_output + y


class LoRALinear(LoRAMixin, nn.Linear):
    """An nn.Linear layer with multi-adapter LoRA support.

    Combines base linear transformation with optional per-adapter low-rank updates.

    Forward pass:
      base_out = F.linear(x, weight, bias)
      return self.apply_lora(x, base_out, adapter_indices)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: torch.dtype = torch.float32,
        use_bias: bool = True,
        device: torch.device | str | None = None,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=use_bias, device=device, dtype=dtype)
        LoRAMixin.init_lora(
            self,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, in_features, max_lora_rank),
            shape_B=(max_lora_adapters, max_lora_rank, out_features),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )

    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        return self.apply_lora(x, base_out, adapter_indices)


def update_adapter_config(model: nn.Module, adapter_index: int, lora_rank: int, lora_alpha: float):
    """Update lora_ranks and lora_scaling for a specific adapter across all LoRA layers.

    Note: This method needs to be called BEFORE any training happens, you should not update
    the config for the same adapter index multiple times throughout training (e.g. it will
    invalidate your current training progress and also violate the assumption that lora_B
    is zero).

    Args:
        model: The model containing LoRA layers
        adapter_index: Index of the adapter to update
        lora_rank: Rank to set for this adapter
        lora_alpha: Alpha value to use for computing scaling (alpha / rank)
    """
    scaling = float(lora_alpha) / float(lora_rank)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRAMixin) and m.max_lora_adapters > 0:
                if 0 <= adapter_index < m.max_lora_adapters:
                    m.lora_ranks[adapter_index] = int(lora_rank)
                    m.lora_scaling[adapter_index] = scaling
                    if lora_rank < m.max_lora_rank:
                        m.lora_A.data[adapter_index, :, lora_rank:] = 0.0
