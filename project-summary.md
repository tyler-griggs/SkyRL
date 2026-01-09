# Branch: `tgriggs/rm_ppo_train` - Full Refactoring Summary

## Overview

This branch contains a significant architectural refactoring of the SkyRL training stack. The primary goals are:

1. **Clean separation of concerns** between algorithm (trainer) and infrastructure (workers, dispatch)
2. **Introduce WorkerDispatch** as a unified coordination layer for multi-model training
3. **Remove redundant training methods** (`ppo_train`, `training_step`) from workers
4. **Move micro-batching logic** from trainer to worker

---

## Motivation

### Problem: Algorithm and Infrastructure Were Entangled

The original codebase had several architectural issues:

1. **Trainer knew too much about infrastructure**
   - Computed micro batch sizes, accumulation steps
   - Looped over micro batches explicitly
   - Managed when to call `optim_step` based on accumulation

2. **Workers had redundant entry points**
   - `ppo_train()` - full training loop (used by Megatron)
   - `training_step()` - single step with forward/backward/optim
   - `forward_backward()` + `optim_step()` - decomposed operations
   - Confusing which to use when

3. **No unified dispatch layer**
   - Trainer managed actor groups directly
   - Offload/backload logic scattered across trainer
   - Hard to reason about GPU state with colocated models

### Solution: Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINER (Algorithm)                       │
│  - PPO algorithm implementation                              │
│  - Knows only mini batches                                   │
│  - Calls dispatch.forward_backward() + dispatch.optim_step() │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 WORKER DISPATCH (Coordination)               │
│  - Manages all actor groups (policy, critic, ref)           │
│  - Handles GPU state (offload/backload) automatically       │
│  - Routes calls to appropriate workers                      │
│  - Handles DP sharding via MeshDispatch                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    WORKERS (Execution)                       │
│  - Execute forward/backward passes                          │
│  - Handle micro-batching internally                         │
│  - Scale gradients at optim_step                            │
│  - Model-specific implementations (FSDP, Megatron)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Changes Made

### 1. New File: `worker_dispatch.py`

Created `WorkerDispatch` class (~315 lines) that:

**Manages Multiple Actor Groups**
```python
class WorkerDispatch:
    def __init__(self, cfg, policy_actor_group, critic_actor_group=None, ref_actor_group=None):
        self._actor_groups = {"policy": policy_actor_group}
        if critic_actor_group: self._actor_groups["critic"] = critic_actor_group
        if ref_actor_group: self._actor_groups["ref"] = ref_actor_group
```

**Provides Unified Training API**
- `forward(model, data)` - inference forward pass
- `forward_backward(model, data)` - training forward/backward
- `optim_step(model)` - optimizer step with gradient scaling
- `save_checkpoint(model, path)` / `load_checkpoint(model, path)`
- `init_model(model, path)` - model initialization

**Automatic GPU State Management**
```python
def _ensure_on_gpu(self, model, need_optimizer=True, need_model=True):
    """Ensure model is on GPU, offloading others in colocation group if needed."""
    if self.colocate_all:
        # Offload other models first
        for other in self._get_colocation_group(model):
            if other != model:
                self._actor_groups[other].offload_to_cpu()
        # Backload requested model
        self._actor_groups[model].backload_to_gpu(...)
```

**Weight Sync Coordination**
- `prepare_for_weight_sync()` - ensures policy on GPU, optimizer offloaded
- `finish_weight_sync()` - offloads policy model after sync
- `broadcast_to_inference_engines()` - sends weights to inference

### 2. Worker Changes (`worker.py`)

**Removed Methods**
- `ppo_train()` - redundant full training loop (kept only for Megatron compatibility via dispatch)
- `training_step()` - redundant combined step

**Refactored Methods**

`_normalize_mini_batch_size()` - Simplified
```python
# OLD: Computed batch sizes
def _normalize_mini_batch_size(self):
    self.policy_mini_batch_size_per_gpu = (mini_batch_size * n_samples) // dp_size
    self.accumulation_steps = self.policy_mini_batch_size_per_gpu // micro_batch_size

# NEW: Just initializes tracking
def _normalize_mini_batch_size(self):
    self._micro_batches_accumulated = 0
```

`forward_backward()` - Now handles micro-batching internally
```python
# OLD: Expected single micro batch, scaled loss
def forward_backward(self, experience, accumulation_steps):
    loss = loss / accumulation_steps  # Scale during backward
    self.strategy.backward(loss, ...)

# NEW: Handles any batch size, no loss scaling
def forward_backward(self, data):
    micro_batch_size = self.cfg.trainer.micro_train_batch_size_per_gpu
    for micro_batch in BatchIterator(data, micro_batch_size):
        metrics = self._forward_backward_micro(micro_batch)  # No scaling
        self._micro_batches_accumulated += 1
    return reduce_metrics(all_metrics)
```

`optim_step()` - Now scales gradients
```python
# OLD: Just stepped optimizer
def optim_step(self):
    return self.strategy.optimizer_step(...)

# NEW: Scales gradients first
def optim_step(self):
    if self._micro_batches_accumulated > 0:
        scale = 1.0 / self._micro_batches_accumulated
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
    grad_norm = self.strategy.optimizer_step(...)
    self._micro_batches_accumulated = 0
    return grad_norm
```

### 3. Trainer Changes (`trainer.py`)

**Removed from Trainer**
- Batch info computation (moved to worker)
- Micro batch looping (moved to worker)
- Direct actor group management (now via dispatch)

**Added to Trainer**
- `has_critic` / `has_ref` properties (algorithm-level knowledge)
- Uses `self.dispatch` for all model operations

**Simplified Training Loop**
```python
# OLD: Trainer managed micro batches
def train_critic_and_policy(self, data):
    batch_info = self.dispatch.get_batch_info("policy")
    for epoch in range(batch_info["num_ppo_epochs"]):
        for step in range(num_micro_batches):
            micro_batch = data[start:end]
            status = self.dispatch.forward_backward("policy", micro_batch)
            if (step + 1) % accumulation_steps == 0:
                self.dispatch.optim_step("policy")

# NEW: Trainer only knows mini batches
def _execute_training_step(self, model, data, metric_prefix):
    mini_batch_size = self.cfg.trainer.policy_mini_batch_size * n_samples
    for epoch in range(self.cfg.trainer.update_epochs_per_batch):
        for step in range(num_mini_batches):
            mini_batch = data[start:end]
            status = self.dispatch.forward_backward(model, mini_batch)
            grad_norm = self.dispatch.optim_step(model)
```

### 4. Test Changes

**Deleted Tests**
- `test_ppo_train.py` - tested removed `ppo_train` method

**Updated Tests**
- `test_training_step.py` - uses `WorkerDispatch` for policy tests
- `test_worker_offload.py` - updated to work with new interfaces
- `test_save_load_checkpoint.py` - updated imports
- `test_trainer.py` - rewrote `test_normalize_mini_batch_size`

---

## Mathematical Equivalence

The gradient scaling approach is mathematically equivalent to loss scaling:

**Old (scale loss during backward)**
```
for i in 1..N:
    grad += (1/N) * ∂loss_i/∂param
optimizer.step(grad)
```

**New (scale gradients at optim_step)**
```
for i in 1..N:
    grad += ∂loss_i/∂param
grad *= 1/N
optimizer.step(grad)
```

Both produce: `grad = (1/N) * Σ ∂loss_i/∂param`

Gradient clipping sees identical values (scaling happens before clipping).

---

## Key Design Decisions

### 1. Worker handles micro-batching (not dispatch)
- Single RPC per batch (better performance)
- Worker has access to model and gradients
- Mirrors Megatron's pattern

### 2. No 1:1 contract between forward_backward and optim_step
- Trainer can call multiple forward_backward before optim_step
- Worker tracks accumulated micro batches
- Enables flexible training patterns

### 3. WorkerDispatch requires policy_actor_group
- Policy is always present in PPO training
- Critic and ref are optional
- Simplifies API without loss of generality

### 4. GPU state managed by dispatch, not trainer
- Dispatch knows about colocate_all, colocate_policy_ref
- Automatically offloads/backloads as needed
- Trainer just calls methods, dispatch handles placement

---

## Files Changed Summary

| File | Lines | Changes |
|------|-------|---------|
| `worker_dispatch.py` | +314 | New file - unified dispatch layer |
| `worker.py` | ~±350 | Refactored forward_backward, optim_step; removed ppo_train, training_step |
| `trainer.py` | ~±370 | Simplified training loop; uses dispatch |
| `test_ppo_train.py` | -220 | Deleted |
| `test_trainer.py` | ~-200 | Simplified test_normalize_mini_batch_size |
| `test_training_step.py` | ~±60 | Uses WorkerDispatch |
| Other tests | ~±100 | Updated imports/interfaces |

**Net change**: ~750 additions, ~1000 deletions (simplified overall)

---

## Verification

```bash
# CPU tests
uv run --isolated --extra dev pytest tests/cpu/test_trainer.py -v

# GPU tests (requires GPU)
uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_training_step.py -v
uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_worker_offload.py -v
uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_save_load_checkpoint.py -v
```

---

## Invariants Preserved

1. **Training correctness** - Same gradients, same updates
2. **Gradient clipping** - Scale before clip (order preserved)
3. **Metrics aggregation** - Same reduce_metrics pattern
4. **DP sharding** - Unchanged, handled by MeshDispatch
5. **Checkpointing** - Same save/load behavior
6. **Megatron compatibility** - `ppo_train` still available via dispatch

---

## Future Work

1. **Make policy_actor_group optional** - Enable critic-only or ref-only testing
2. **Remove role-specific identifiers** - Move to generic model IDs
3. **Add WorkerDispatch integration tests** - Test multi-model coordination
4. **Unify FSDP and Megatron paths** - Both should use forward_backward + optim_step

---

## Commits

```
68e1ade pulling apart infra and alg logic
4e79626 x
6909e3b fix tests
1959fed removing training step
e855d41 Merge remote-tracking branch 'real/main' into tgriggs/rm_training_step
```

## Branch

`tgriggs/rm_ppo_train`

Base: `main` (after deepspeed removal #827)
