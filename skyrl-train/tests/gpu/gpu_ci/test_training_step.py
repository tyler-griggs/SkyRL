"""
Run with:
uv run --isolated --extra dev -- pytest tests/gpu/gpu_ci/test_training_step.py
"""

import ray
import pytest
import hydra
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, make_dummy_training_batch, validate_cfg, get_rank_0_memory
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.workers.worker_dispatch import WorkerDispatch


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine_tensor_parallel_size = 2

    return cfg


@pytest.fixture
def cfg() -> DictConfig:
    return get_test_actor_config()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packed", "strategy"),
    [
        # (True, "fsdp"), (False, "fsdp"),
        (True, "fsdp2"),
        (False, "fsdp2"),
    ],
    ids=[
        # "packed-fsdp",
        # "unpacked-fsdp",
        "packed-fsdp2",
        "unpacked-fsdp2",
    ],
)
async def test_policy_forward_backward_and_optim_step(ray_init_fixture, cfg, packed, strategy):
    """
    Full test: initialize actor group via WorkerDispatch, send dummy experience to forward_backward + optim_step, validate output.
    """
    cfg.trainer.use_sample_packing = packed
    cfg.trainer.strategy = strategy
    validate_cfg(cfg)

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Wrap actor group in WorkerDispatch
        dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)

        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        # Use dispatch for training operations
        result = dispatch.forward_backward("policy", dummy_batch)
        grad_norm = dispatch.optim_step("policy")

        # Use actor group directly for inspection
        get_rank_0_memory(actor_group, "memory after forward_backward + optim_step")

        assert isinstance(result, dict), "Result should be a dictionary of training stats"
        assert "policy_loss" in result
        assert "ppo_clip_ratio" in result
        assert "policy_entropy" in result
        for k, v in result.items():
            assert isinstance(v, (int, float)), f"{k} should be an int or float"

        assert grad_norm is None or isinstance(grad_norm, float), "grad_norm should be None or float"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packed", "strategy"),
    [
        # (True, "fsdp"), (False, "fsdp"),
        (True, "fsdp2"),
        (False, "fsdp2"),
    ],
    ids=[
        # "packed-fsdp",
        # "unpacked-fsdp",
        "packed-fsdp2",
        "unpacked-fsdp2",
    ],
)
async def test_critic_forward_backward_and_optim_step(ray_init_fixture, cfg, packed, strategy):
    """
    Full test: initialize critic actor group, send dummy experience to forward_backward + optim_step, validate output.

    Note: This test uses direct actor_group calls instead of WorkerDispatch because WorkerDispatch
    requires a policy_actor_group. The policy test validates dispatch routing; this test validates
    critic worker functionality directly.
    """
    cfg.trainer.use_sample_packing = packed
    cfg.trainer.strategy = strategy
    validate_cfg(cfg)
    try:
        actor_group = init_worker_with_type(
            "critic",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", dummy_batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        for result in results:
            assert isinstance(result, dict), "Result should be a dictionary of training stats"
            assert "critic_loss" in result
            assert "values_mean" in result
            for k, v in result.items():
                assert isinstance(v, float), f"{k} should be a float"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_forward_backward_with_loss_fn_param(ray_init_fixture):
    """Test forward_backward with Tinker-style loss_fn parameter."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.use_sample_packing = False
    validate_cfg(cfg)

    try:
        actor_group = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False,
            num_gpus_per_node=1, cfg=cfg,
        )
        dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        # Test 1: loss_fn="ppo" (should map to "regular")
        result = dispatch.forward_backward("policy", dummy_batch, loss_fn="ppo")
        assert "policy_loss" in result

        # Test 2: loss_fn_config override
        result = dispatch.forward_backward(
            "policy", dummy_batch, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.1, "clip_high_threshold": 0.3},
        )
        assert "policy_loss" in result

        dispatch.optim_step("policy")
    finally:
        ray.shutdown()
