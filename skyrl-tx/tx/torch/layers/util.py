from __future__ import annotations
import torch


def prepare_routing(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    num_groups: int,
    adapter_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepare inputs for ragged operations by sorting tokens by group.

    Args:
        tokens: Tensor of shape (num_tokens, ...) to be sorted by group
        indices: Tensor of shape (num_tokens,) indicating group assignment for each token
        num_groups: Total number of groups
        adapter_indices: Optional tensor of shape (num_tokens,) to be sorted together with tokens

    Returns:
        sorted_tokens: Tokens sorted by group index
        group_sizes: Number of tokens in each group
        unsort_indices: Indices to restore original order after ragged operations
        sorted_adapter_indices: Adapter indices sorted with tokens (or None if not provided)
    """
    # Sort by group index
    sort_idx = torch.argsort(indices)
    sorted_tokens = tokens[sort_idx]
    sorted_adapter_indices = None if adapter_indices is None else adapter_indices[sort_idx]
    
    # Compute group sizes (minlength guarantees output length)
    sorted_indices = indices[sort_idx]
    group_sizes = torch.bincount(sorted_indices, minlength=num_groups)
    
    # Inverse permutation to restore original order
    unsort_indices = torch.argsort(sort_idx)
    
    return sorted_tokens, group_sizes, unsort_indices, sorted_adapter_indices
