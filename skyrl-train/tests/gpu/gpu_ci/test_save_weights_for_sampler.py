"""
Test save_weights_for_sampler() - the Tinker API method for syncing weights before sampling.

This test validates the full flow:
1. Initialize policy model and inference engine
2. Do a training step (forward_backward + optim_step)
3. Call save_weights_for_sampler() to sync weights
4. Sample using the inference engine
5. Verify sampling works with the updated weights

GPU Requirements: 1 GPU (multi-GPU aspects are tested elsewhere)

Run with:
# vllm backend:
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_save_weights_for_sampler.py -m "vllm" -v

# sglang backend:
uv run --isolated --extra dev --extra sglang pytest tests/gpu/gpu_ci/test_save_weights_for_sampler.py -m "sglang" -v
"""

import pytest
import asyncio
import ray
import hydra
from omegaconf import DictConfig

from tests.gpu.utils import (
    init_worker_with_type,
    get_test_prompts,
    init_inference_engines,
    run_inference,
    make_dummy_training_batch,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl_train.utils.utils import validate_cfg

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.critic.model.path = ""
        # Use 1 GPU - multi-GPU aspects are tested elsewhere
        cfg.trainer.placement.policy_num_gpus_per_node = 1
        cfg.generator.inference_engine_tensor_parallel_size = 1
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.run_engines_locally = True
        cfg.trainer.use_sample_packing = False
        cfg.trainer.logger = "console"

        # Validate config (sets max_seq_len and other derived fields)
        validate_cfg(cfg)

        return cfg


@pytest.mark.parametrize(
    ("colocate_all", "strategy", "backend"),
    [
        pytest.param(False, "fsdp2", "vllm", marks=pytest.mark.vllm),
        pytest.param(True, "fsdp2", "vllm", marks=pytest.mark.vllm),
    ],
    ids=[
        "no_colocate_fsdp2_vllm",
        "colocate_fsdp2_vllm",
    ],
)
def test_save_weights_for_sampler_then_inference(ray_init_fixture, colocate_all, strategy, backend):
    """
    Test that save_weights_for_sampler() correctly syncs weights before sampling.

    This test validates the Tinker API pattern:
    1. Train: forward_backward + optim_step
    2. Sync: save_weights_for_sampler()
    3. Sample: inference engine generates with updated weights
    """
    try:
        cfg = get_test_config()
        cfg.trainer.placement.colocate_all = colocate_all
        cfg.trainer.strategy = strategy
        cfg.generator.backend = backend

        # Initialize inference engine (uses 1 GPU, set in get_test_config)
        client, pg = init_inference_engines(
            model=MODEL,
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend=backend,
            sleep_level=2,  # Full sleep since we explicitly sync weights
        )

        # Initialize policy worker (uses 1 GPU)
        policy_group = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Initialize weight sync state (required for broadcast)
        ray.get(policy_group.async_run_ray_method("pass_through", "init_weight_sync_state", client))

        # Create WorkerDispatch with inference_engine_client
        dispatch = WorkerDispatch(
            cfg=cfg,
            policy_actor_group=policy_group,
            inference_engine_client=client,
        )

        # If colocate_all, sleep inference engine to free GPU memory for training
        if colocate_all:
            asyncio.run(client.sleep())
            dispatch.mark_all_offloaded()

        # === Step 1: Do a training step ===
        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        # Training: forward_backward + optim_step
        dispatch.forward_backward("policy", dummy_batch)
        grad_norm = dispatch.optim_step("policy")
        assert grad_norm is not None, "optim_step should return gradient norm"

        # Verify weights are dirty after training
        assert dispatch._weights_dirty, "Weights should be dirty after optim_step"

        # === Step 2: Call save_weights_for_sampler ===
        result = dispatch.save_weights_for_sampler()

        # Verify result structure
        assert "type" in result, "Result should have 'type' field"
        assert result["type"] == "save_weights_for_sampler"
        assert "sampling_session_id" in result or "path" in result

        # Verify weights are clean after sync
        assert not dispatch._weights_dirty, "Weights should be clean after save_weights_for_sampler"

        # === Step 3: Sample using inference engine ===
        asyncio.run(client.reset_prefix_cache())
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL, num_samples=5), sampling_params))

        # Verify we got responses
        assert "responses" in outputs, "Inference should return responses"
        assert len(outputs["responses"]) == 5, f"Expected 5 responses, got {len(outputs['responses'])}"

        # Verify responses are non-empty
        for i, response in enumerate(outputs["responses"]):
            assert len(response) > 0, f"Response {i} should not be empty"

        print(f"Example output: {outputs['responses'][0][:100]}...")

    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("vllm", marks=pytest.mark.vllm),
    ],
    ids=["vllm"],
)
def test_save_weights_for_sampler_skips_when_clean(ray_init_fixture, backend):
    """
    Test that save_weights_for_sampler() skips sync when weights haven't changed.

    The _weights_dirty flag should prevent unnecessary syncs.
    """
    try:
        cfg = get_test_config()
        cfg.trainer.placement.colocate_all = False
        cfg.trainer.strategy = "fsdp2"
        cfg.generator.backend = backend

        # Initialize inference engine (uses 1 GPU)
        client, pg = init_inference_engines(
            model=MODEL,
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=False,
            backend=backend,
            sleep_level=2,
        )

        # Initialize policy worker (uses 1 GPU)
        policy_group = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Initialize weight sync state
        ray.get(policy_group.async_run_ray_method("pass_through", "init_weight_sync_state", client))

        # Create WorkerDispatch
        dispatch = WorkerDispatch(
            cfg=cfg,
            policy_actor_group=policy_group,
            inference_engine_client=client,
        )

        # Initial weights are dirty (set in constructor)
        assert dispatch._weights_dirty, "Weights should be dirty initially"

        # First sync - should actually sync
        result1 = dispatch.save_weights_for_sampler()
        assert not dispatch._weights_dirty, "Weights should be clean after sync"
        assert "unchanged" not in result1.get("sampling_session_id", ""), "First sync should not be 'unchanged'"

        # Second sync without training - should skip
        result2 = dispatch.save_weights_for_sampler()
        assert not dispatch._weights_dirty, "Weights should still be clean"
        assert "unchanged" in result2.get("sampling_session_id", ""), "Second sync should be 'unchanged'"

        # Do training
        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)
        dispatch.forward_backward("policy", dummy_batch)
        dispatch.optim_step("policy")

        # Weights should be dirty again
        assert dispatch._weights_dirty, "Weights should be dirty after training"

        # Third sync - should actually sync
        result3 = dispatch.save_weights_for_sampler()
        assert not dispatch._weights_dirty, "Weights should be clean after sync"
        assert "unchanged" not in result3.get("sampling_session_id", ""), "Third sync should not be 'unchanged'"

    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("vllm", marks=pytest.mark.vllm),
    ],
    ids=["vllm"],
)
def test_save_weights_for_sampler_multiple_training_steps(ray_init_fixture, backend):
    """
    Test that multiple training steps followed by one save_weights_for_sampler works correctly.

    This validates the key behavior: users can do multiple optim_steps before syncing.
    """
    try:
        cfg = get_test_config()
        cfg.trainer.placement.colocate_all = False
        cfg.trainer.strategy = "fsdp2"
        cfg.generator.backend = backend

        # Initialize inference engine (uses 1 GPU)
        client, pg = init_inference_engines(
            model=MODEL,
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=False,
            backend=backend,
            sleep_level=2,
        )

        # Initialize policy worker (uses 1 GPU)
        policy_group = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Initialize weight sync state
        ray.get(policy_group.async_run_ray_method("pass_through", "init_weight_sync_state", client))

        # Create WorkerDispatch
        dispatch = WorkerDispatch(
            cfg=cfg,
            policy_actor_group=policy_group,
            inference_engine_client=client,
        )

        # Initial sync to clear dirty flag
        dispatch.save_weights_for_sampler()
        assert not dispatch._weights_dirty

        # Do multiple training steps WITHOUT syncing
        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        for step in range(3):
            dispatch.forward_backward("policy", dummy_batch)
            dispatch.optim_step("policy")
            # Weights should be dirty after each optim_step
            assert dispatch._weights_dirty, f"Weights should be dirty after step {step}"

        # Now sync once - should sync all accumulated changes
        result = dispatch.save_weights_for_sampler()
        assert not dispatch._weights_dirty, "Weights should be clean after sync"
        assert "unchanged" not in result.get("sampling_session_id", ""), "Should have synced"

        # Verify inference works
        asyncio.run(client.reset_prefix_cache())
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL, num_samples=2), sampling_params))
        assert len(outputs["responses"]) == 2, "Should get 2 responses"

    finally:
        ray.shutdown()
