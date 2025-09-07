import os
import time

import ray
import torch
import sys
import logging
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy, PlacementGroup


class Timer:
    def __init__(self, message, update_dict=None):
        self.message = message
        self.update_dict = update_dict

    def __enter__(self):
        self.start_time = time.time()
        logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"Finished: '{self.message}', time cost: {time.time() - self.start_time:.2f}s")
        if self.update_dict is not None:
            self.update_dict[self.message] = self.update_dict.get(self.message, 0.0) + time.time() - self.start_time

    async def __aenter__(self):
        self.start_time = time.time()
        logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"Finished: '{self.message}', time cost: {time.time() - self.start_time:.2f}s")
        if self.update_dict is not None:
            self.update_dict[self.message] = self.update_dict.get(self.message, 0.0) + time.time() - self.start_time


def validate_batch_sizes(cfg: DictConfig):
    """
    Validate configured batch sizes.

    Explanation of how batching operates:
    1. Each prompt in train_batch_size creates `n_samples_per_prompt` total samples.
    2. During training, these samples are split across data parallel (DP) workers, making the effective per-GPU batch size: `train_batch_size * n_samples_per_prompt / dp_size`.
    3. Mini batches are similarly normalized to per-gpu mini batches with size: `mini_batch_size * n_samples_per_prompt / dp_size`.
    4. Per-gpu train batch size must be divisble by per-gpu mini batch size, otherwise the last mini batch will be incomplete.
    5. Per-gpu mini batch size must be divisible by per-gpu micro batch size, otherwise the last micro batch will be incomplete.
    """
    assert cfg.trainer.train_batch_size >= cfg.trainer.policy_mini_batch_size
    assert cfg.trainer.policy_mini_batch_size > 0, "policy_mini_batch_size must be greater than 0"
    if cfg.trainer.critic.model.path is not None:
        assert cfg.trainer.train_batch_size >= cfg.trainer.critic_mini_batch_size
        assert cfg.trainer.critic_mini_batch_size > 0, "critic_mini_batch_size must be greater than 0"
    assert cfg.trainer.micro_train_batch_size_per_gpu > 0, "micro_train_batch_size_per_gpu must be greater than 0"
    assert cfg.trainer.micro_forward_batch_size_per_gpu > 0, "micro_forward_batch_size_per_gpu must be greater than 0"

    # Validate policy mini batch size
    policy_world_size = cfg.trainer.placement.policy_num_nodes * cfg.trainer.placement.policy_num_gpus_per_node
    policy_dp_size = policy_world_size // cfg.trainer.policy.sequence_parallel_size
    assert (
        cfg.trainer.train_batch_size % cfg.trainer.policy_mini_batch_size == 0
    ), f"train_batch_size {cfg.trainer.train_batch_size} should be divisible by policy_mini_batch_size {cfg.trainer.policy_mini_batch_size}"
    policy_mini_batch_size_per_gpu = (
        cfg.trainer.policy_mini_batch_size * cfg.generator.n_samples_per_prompt // policy_dp_size
    )
    assert policy_mini_batch_size_per_gpu > 0, (
        f"Invalid policy_mini_batch_size_per_gpu: {policy_mini_batch_size_per_gpu}. "
        f"mini_batch_size={cfg.trainer.policy_mini_batch_size}, "
        f"n_samples_per_prompt={cfg.generator.n_samples_per_prompt}, "
        f"dp_size={policy_dp_size}"
    )
    assert (
        policy_mini_batch_size_per_gpu % cfg.trainer.micro_train_batch_size_per_gpu == 0
    ), f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be divisible by micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    assert (
        policy_mini_batch_size_per_gpu // cfg.trainer.micro_train_batch_size_per_gpu > 0
    ), f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be larger than micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    policy_train_batch_size_per_gpu = (
        cfg.trainer.train_batch_size * cfg.generator.n_samples_per_prompt // policy_dp_size
    )

    # `train_batch_size_per_gpu` should be divisible by `policy_mini_batch_size_per_gpu`
    assert (
        policy_train_batch_size_per_gpu % policy_mini_batch_size_per_gpu == 0
    ), f"normalized policy_train_batch_size_per_gpu (train_batch_size * n_samples_per_prompt // policy_dp_size) {policy_train_batch_size_per_gpu} should be divisible by policy_mini_batch_size_per_gpu (policy_mini_batch_size * n_samples_per_prompt // policy_dp_size) {policy_mini_batch_size_per_gpu}"

    # Validate critic mini batch size
    critic_world_size = cfg.trainer.placement.critic_num_nodes * cfg.trainer.placement.critic_num_gpus_per_node
    critic_dp_size = critic_world_size // cfg.trainer.critic.sequence_parallel_size

    if cfg.trainer.critic.model.path is not None:
        assert (
            cfg.trainer.train_batch_size % cfg.trainer.critic_mini_batch_size == 0
        ), f"train_batch_size {cfg.trainer.train_batch_size} should be divisible by critic_mini_batch_size {cfg.trainer.critic_mini_batch_size}"
        critic_mini_batch_size_per_gpu = (
            cfg.trainer.critic_mini_batch_size * cfg.generator.n_samples_per_prompt // critic_dp_size
        )
        assert critic_mini_batch_size_per_gpu > 0, (
            f"Invalid critic_mini_batch_size_per_gpu: {critic_mini_batch_size_per_gpu}. "
            f"mini_batch_size={cfg.trainer.critic_mini_batch_size}, "
            f"n_samples_per_prompt={cfg.generator.n_samples_per_prompt}, "
            f"dp_size={critic_dp_size}"
        )
        assert (
            critic_mini_batch_size_per_gpu % cfg.trainer.micro_train_batch_size_per_gpu == 0
        ), f"normalized critic_mini_batch_size_per_gpu {critic_mini_batch_size_per_gpu} should be divisible by micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
        assert (
            critic_mini_batch_size_per_gpu // cfg.trainer.micro_train_batch_size_per_gpu > 0
        ), f"normalized critic_mini_batch_size_per_gpu {critic_mini_batch_size_per_gpu} should be larger than micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
        critic_train_batch_size_per_gpu = (
            cfg.trainer.train_batch_size * cfg.generator.n_samples_per_prompt // critic_dp_size
        )
        assert (
            critic_train_batch_size_per_gpu % critic_mini_batch_size_per_gpu == 0
        ), f"normalized critic_train_batch_size_per_gpu (train_batch_size * n_samples_per_prompt // critic_dp_size) {critic_train_batch_size_per_gpu} should be divisible by critic_mini_batch_size_per_gpu (critic_mini_batch_size * n_samples_per_prompt // critic_dp_size) {critic_mini_batch_size_per_gpu}"


def validate_megatron_cfg(cfg: DictConfig):
    # not yet supported + tested features
    assert cfg.generator.weight_sync_backend == "nccl", "only nccl is supported for megatron weight sync"
    assert cfg.generator.backend == "vllm", "only vllm is supported for with megatron"
    assert cfg.trainer.placement.colocate_all, "only colocate_all=True is supported for megatron training"
    assert cfg.trainer.critic.model.path is None, "only GRPO training is currently supported for megatron"
    assert not cfg.trainer.use_sample_packing, "sample packing is not yet supported for megatron"

    worker_configs = [(cfg.trainer.policy, "policy"), (cfg.trainer.ref, "ref")]
    for config, worker_type in worker_configs:
        # context, expert, and export tensor parallel are not yet supported for megatron
        assert (
            config.megatron_config.context_parallel_size == 1
        ), f"found {worker_type}.context_parallel_size > 1, context parallel is not yet supported for megatron"
        assert (
            config.megatron_config.expert_model_parallel_size == 1
        ), f"found {worker_type}.expert_model_parallel_size > 1, expert model parallel is not yet supported for megatron"
        assert (
            config.megatron_config.expert_tensor_parallel_size == 1
        ), f"found {worker_type}.expert_tensor_parallel_size > 1, expert tensor parallel is not yet supported for megatron"
        # check that sequence parallel is not configured outside of megatron
        assert (
            config.sequence_parallel_size == 1
        ), f"found {worker_type}.sequence_parallel_size={config.sequence_parallel_size}, ulysses style sequence parallel is not supported for megatron"

        # check that DP size is 1 - there are some convergence/grad norm issues that need to be resolved for DP > 1
        num_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
        assert (
            config.megatron_config.tensor_model_parallel_size * config.megatron_config.pipeline_model_parallel_size
            == num_gpus
        ), f"DP size should be 1, but found {worker_type}.megatron_config.tensor_model_parallel_size={config.megatron_config.tensor_model_parallel_size} * {config.megatron_config.pipeline_model_parallel_size}={config.megatron_config.tensor_model_parallel_size * config.megatron_config.pipeline_model_parallel_size} != num_gpus={num_gpus}"


def validate_cfg(cfg: DictConfig):
    from .ppo_utils import AdvantageEstimatorRegistry, PolicyLossRegistry

    if cfg.generator.max_turns == 1:
        assert (
            cfg.generator.max_input_length == cfg.trainer.max_prompt_length
        ), "generator.max_input_length should be set equal to trainer.max_prompt_length for single-turn generation"
    else:
        assert (
            cfg.generator.max_input_length >= cfg.trainer.max_prompt_length
        ), "generator.max_input_length should be set greater than or equal to trainer.max_prompt_length for multi-turn generation"

    if not cfg.generator.run_engines_locally:
        assert cfg.generator.num_inference_engines == len(
            cfg.generator.remote_inference_engine_urls
        ), "num_inference_engines should be equal to the number of remote_inference_engine_urls"

    if not cfg.generator.async_engine and cfg.generator.backend == "vllm":
        assert (
            cfg.generator.batched
        ), "if we are using the offline vLLM engine, we need to put generator in batched mode for faster generation"

    assert (
        cfg.trainer.sequence_parallel_backend == "ulysses"
    ), f"only ulysses is supported as of now, got {cfg.trainer.sequence_parallel_backend}"

    # if advantage estimator is GAE, then critic path should be provided
    if cfg.trainer.algorithm.advantage_estimator == "gae":
        assert (
            cfg.trainer.critic.model.path
        ), "`trainer.critic.model.path` should be provided for PPO training, got `None`"

    assert not (
        cfg.trainer.algorithm.use_kl_in_reward and cfg.trainer.algorithm.use_kl_loss
    ), "use_kl_in_reward and use_kl_loss should be mutually exclusive"

    if cfg.trainer.strategy in ("fsdp", "fsdp2"):
        assert not (
            cfg.trainer.policy.fsdp_config.cpu_offload and cfg.trainer.strategy == "fsdp"
        ), "fwd pass cpu offloading is not supported for FSDP1 policy worker, use FSDP2 instead"
        assert not (
            cfg.trainer.critic.fsdp_config.cpu_offload and cfg.trainer.strategy == "fsdp"
        ), "fwd pass cpu offloading is not supported for FSDP1 critic worker, use FSDP2 instead"

    if cfg.trainer.strategy == "deepspeed":
        assert (
            cfg.trainer.policy.deepspeed_config.zero_optimization.stage == 3
        ), "only deepspeed stage 3 is currently supported!"

    validate_batch_sizes(cfg)

    # tracker
    if cfg.trainer.logger == "wandb":
        assert os.environ.get("WANDB_API_KEY"), "`WANDB_API_KEY` is required for `wandb` logger"

    if cfg.trainer.max_ckpts_to_keep == 0:
        raise ValueError(
            "`max_ckpts_to_keep` must be greater than 0 to keep the last N checkpoints or negative to keep all checkpoints"
        )

    # resolve override_existing_update_group
    if cfg.generator.override_existing_update_group == "auto":
        if cfg.generator.backend == "vllm" and not cfg.generator.run_engines_locally:
            # remote engines can be launched separately so we `enable` by default
            cfg.generator.override_existing_update_group = "enable"
        else:
            # for local engines or sglang, we disable
            cfg.generator.override_existing_update_group = "disable"

    assert (
        cfg.trainer.algorithm.policy_loss_type in PolicyLossRegistry.list_available()
    ), f"invalid policy_loss_type: {cfg.trainer.algorithm.policy_loss_type}. Must be one of {PolicyLossRegistry.list_available()}"

    assert (
        cfg.trainer.algorithm.advantage_estimator in AdvantageEstimatorRegistry.list_available()
    ), f"invalid advantage_estimator: {cfg.trainer.algorithm.advantage_estimator}. Must be one of {AdvantageEstimatorRegistry.list_available()}"

    assert cfg.trainer.algorithm.loss_reduction in (
        "token_mean",
        "sequence_mean",
        "seq_mean_token_sum_norm",
    ), f"invalid loss_reduction: {cfg.trainer.algorithm.loss_reduction}. Must be one of `['token_mean', 'sequence_mean', 'seq_mean_token_sum_norm']`"

    # add field to algorithm config needed for loss functions
    # create a new config to make it modifiable
    algorithm_config = OmegaConf.create(cfg.trainer.algorithm)
    # NOTE (erictang000): this is the max sequence length including the prompt, since max response length
    # per batch can be variable based on the prompt length. This is used to normalize the loss for
    # seq_mean_token_sum_norm loss reduction. Potentially revisit this if we update to use a
    # fixed max response budget.
    algorithm_config.max_seq_len = cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length

    # TODO (erictang000): remove these after deprecation period
    if algorithm_config.use_abs_kl:
        logger.warning("`use_abs_kl` will be deprecated, overriding to use `kl_estimator_type='abs'` instead")
        algorithm_config.kl_estimator_type = "abs"
    elif algorithm_config.use_kl_estimator_k3:
        logger.warning("`use_kl_estimator_k3` will be deprecated, overriding to use `kl_estimator_type='k3'` instead")
        algorithm_config.kl_estimator_type = "k3"
    cfg.trainer.algorithm = algorithm_config

    # TODO: fix once we support these features with SGLang
    if cfg.generator.backend == "sglang" and cfg.generator.run_engines_locally:
        assert cfg.generator.inference_engine_tensor_parallel_size == 1, (
            "As of now, We do not support tensor parallel inference engine with SGLang when running engines locally. "
            "Please set `inference_engine_tensor_parallel_size` to 1."
        )

    if cfg.trainer.strategy == "deepspeed" and not (
        cfg.trainer.policy.optimizer_config.offload_after_step
        and cfg.trainer.critic.optimizer_config.offload_after_step
    ):
        raise ValueError(
            "`offload_after_step=False` is not supported for DeepSpeed, please set `offload_after_step` to `true` for both policy and critic"
        )

    if cfg.generator.backend == "sglang" and not cfg.generator.use_conversation_multi_turn:
        raise NotImplementedError("`use_conversation_multi_turn=False` is not supported for SGLang backend")

    if cfg.trainer.algorithm.use_tis:
        if cfg.trainer.algorithm.tis_imp_ratio_cap <= 0:
            raise ValueError(
                f"If `trainer.algorithm.use_tis` is `True` then `cfg.trainer.algorithm.tis_imp_ratio_cap` should be > 0, got {cfg.trainer.algorithm.tis_imp_ratio_cap }"
            )
        if cfg.generator.sampling_params.logprobs is None:
            logger.warning(
                "`generator.sampling_params.logprobs` is `None` but `trainer.algorithm.use_tis` is `True`. Setting `logprobs` to `True`."
            )
            # just set to 0 for better user exp
            cfg.generator.sampling_params.logprobs = 0

        if cfg.generator.backend == "sglang":
            raise NotImplementedError("`trainer.algorithm.use_tis` doesn't support Sglang backend, please use vLLM")

        if not cfg.generator.batched:
            raise ValueError(
                "Gneration with `trainer.algorithm.use_tis` needs to be batched with only single turn generation"
            )

    if cfg.generator.sampling_params.logprobs is not None:
        assert isinstance(cfg.generator.sampling_params.logprobs, int)

        if cfg.generator.sampling_params.logprobs > 0:
            raise ValueError(
                f"`logprobs` if set should be 0 i.e only for the chosen token, got {cfg.generator.sampling_params.logprobs}"
            )
        if not cfg.generator.batched:
            raise NotImplementedError(
                "Async generation with `generator.batched=false` doesn't support `sampling_params.logprobs`"
            )

        if not cfg.generator.run_engines_locally:
            raise NotImplementedError("Remote inference mode doesn't support `sampling_params.logprobs`")

    if cfg.trainer.strategy == "megatron":
        validate_megatron_cfg(cfg)
    if cfg.generator.backend == "sglang":
        # Some sampling parameters are not supported in SGLang when `skip_tokenizer_init` is True.
        if cfg.generator.sampling_params.stop is not None or cfg.generator.eval_sampling_params.stop is not None:
            raise ValueError(
                "`sampling_params.stop` and `eval_sampling_params.stop` are not supported for SGLang backend "
                "since we always set `skip_tokenizer_init` to True. If you have to use these parameters, you can switch to vLLM. "
                "See this issue for more: https://github.com/sgl-project/sglang/issues/9039#issuecomment-3218331087"
            )
        if "min_new_tokens" in cfg.generator.sampling_params or "min_new_tokens" in cfg.generator.eval_sampling_params:
            raise ValueError(
                "`sampling_params.min_new_tokens` and `eval_sampling_params.min_new_tokens` are not "
                "supported for SGLang backend since we always set `skip_tokenizer_init` to True. "
                "If you have to use these parameters, you can switch to vLLM. "
                "See this issue for more: https://github.com/sgl-project/sglang/issues/9039#issuecomment-3218331087"
            )

    if cfg.generator.use_conversation_multi_turn:
        if (
            cfg.generator.sampling_params.stop is not None or cfg.generator.eval_sampling_params.stop is not None
        ) and not cfg.generator.append_eos_token_after_stop_str_in_multi_turn:
            logger.warning(
                "WARNING: `sampling_params.stop` and `eval_sampling_params.stop` are specified and we "
                "are using multi-turn generation. You might want to set `append_eos_token_after_stop_str_in_multi_turn` "
                "to `True` to append tokenizer.eos_token_id to the assistant-generated response to match the chat template."
            )

    if cfg.generator.enable_http_endpoint and not cfg.generator.async_engine:
        raise ValueError("generator.async_engine must be True when generator.enable_http_endpoint==True.")


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def prepare_runtime_environment(cfg: DictConfig) -> dict[str, str]:
    """
    Prepare environment variables for Ray runtime environment.

    Args:
        cfg: Training config

    Returns:
        Dict[str, str]: Environment variables to be used in Ray runtime environment
    """
    # TODO(sumanthrh): introduce a debug mode and add debugging flags like `CUDA_LAUNCH_BLOCKING` here
    env_vars = {}

    # NOTE (charlie): See https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
    # and https://docs.vllm.ai/en/v0.9.2/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
    # Same for SGLang as we set `NCCL_CUMEM_ENABLE` to 0 in `sglang_engine.py`'s _patched_set_envs_and_config
    if cfg.generator.weight_sync_backend == "nccl":
        env_vars["NCCL_CUMEM_ENABLE"] = "0"

    if cfg.generator.backend == "vllm":
        # NOTE (sumanthrh): In vllm >= 0.9.0, we need to explicitly allow for serialization via pickle for collective RPCs.
        # During weight transfer, we use IPC handles, which contains a `function` object and requires pickling.
        env_vars["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # NOTE (sumanthrh): In vLLM >= 0.9.0, we've observed compilatiion failures with torch compile. removing the compilation directory and trying
        # again does not fix the issue. Temporarily we disable compilation cache, which seems to fix the issue.
        # This should not have any effect on performance - compilation will still happen, it's just not cached
        # TODO (sumanthrh): remove this once vLLM fixes the issue
        env_vars["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        if not os.environ.get("VLLM_USE_V1", False):
            logger.info(
                "`VLLM_USE_V1` is not specified, setting `VLLM_USE_V1` to 1. To override, set `VLLM_USE_V1` explicitly"
            )
            env_vars["VLLM_USE_V1"] = "1"
            env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Use max of available GPU counts, defaulting to 1 if none found
    gpu_counts = []
    if hasattr(cfg.generator, "inference_engine_tensor_parallel_size"):
        gpu_counts.append(cfg.generator.inference_engine_tensor_parallel_size)
    if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "placement"):
        placement = cfg.trainer.placement
        gpu_counts.extend(
            [
                placement.policy_num_gpus_per_node,
                placement.critic_num_gpus_per_node,
                placement.ref_num_gpus_per_node,
                placement.reward_num_gpus_per_node,
            ]
        )
    max_num_gpus_per_node = max(gpu_counts) if gpu_counts else 1
    if not peer_access_supported(max_num_gpus_per_node=max_num_gpus_per_node):
        logger.info("Peer access is not supported on this node type, disabling NCCL P2P and SHM")
        env_vars["NCCL_P2P_DISABLE"] = "1"
        env_vars["NCCL_SHM_DISABLE"] = "1"

    if cfg.trainer.strategy == "megatron":
        # useful when tp > 1 (and thus megatron sequence_parallel is enabled)
        # see: https://github.com/NVIDIA/Megatron-LM/issues/533#issuecomment-1760193239
        env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # TODO: this can be removed if we standardize on env files.
    # But it's helpful for a quickstart
    if os.environ.get("WANDB_API_KEY"):
        logger.info("Exporting wandb api key to ray runtime env")
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    if os.environ.get("MLFLOW_TRACKING_URI"):
        logger.info("Exporting mlflow tracking uri to ray runtime env")
        env_vars["MLFLOW_TRACKING_URI"] = os.environ["MLFLOW_TRACKING_URI"]

    if os.environ.get("MLFLOW_TRACKING_TOKEN"):
        logger.info("Exporting mlflow tracking token to ray runtime env")
        env_vars["MLFLOW_TRACKING_TOKEN"] = os.environ["MLFLOW_TRACKING_TOKEN"]

    if os.environ.get("SKYRL_LD_LIBRARY_PATH_EXPORT"):
        # export `LD_LIBRARY_PATH` to ray runtime env.
        # For some reason the `LD_LIBRARY_PATH` is not exported to the worker with .env file.
        logger.info(f"Exporting `LD_LIBRARY_PATH` to ray runtime env: {os.environ['LD_LIBRARY_PATH']}")
        env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

    return env_vars


def configure_worker_logging() -> None:
    """
    In Ray workers, stderr/stdout are not TTYs, so Loguru disables color.
    Force color, restore the old format (green timestamp, bold INFO),
    and route stdlib `logging` through Loguru so third‑party logs match.
    """
    # 1) Loguru formatting (force colors)
    logger.remove()
    # Make INFO green + bold so "Started"/"Finished" stand out again
    logger.level("INFO", color="<bold><green>")
    logger.add(
        sys.stderr,
        colorize=True,  # keep ANSI even without a TTY
        enqueue=True,  # safer with threads/process handoffs
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # 2) Route stdlib logging -> Loguru (so vLLM/transformers/etc. are colored)
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(logging.INFO)
    # Optionally quiet Ray’s own internal logs if they get noisy:
    # logging.getLogger("ray").setLevel(logging.WARN)


def ray_worker_init():
    configure_worker_logging()


def initialize_ray(cfg: DictConfig):
    """
    Initialize Ray cluster with prepared runtime environment.

    Args:
        cfg: Training config
    """
    from .ppo_utils import (
        sync_registries,
    )

    env_vars = prepare_runtime_environment(cfg)
    ray.init(
        runtime_env={
            "env_vars": env_vars,
            "worker_process_setup_hook": ray_worker_init,
        },
        # job_config=JobConfig(
        #     worker_process_setup_hook="skyrl_train.entrypoints.main_base:_ray_worker_init"
        # ),
    )

    # create the named ray actors for the registries to make available to all workers
    sync_registries()


def get_ray_pg_ready_with_timeout(pg: PlacementGroup, timeout: int = 60):
    try:
        ray.get(pg.ready(), timeout=timeout)
    except Exception as e:
        # Extract resource demands from the placement group
        bundles = pg.bundle_specs
        total_gpus = sum(bundle.get("GPU", 0) for bundle in bundles)
        total_cpus = sum(bundle.get("CPU", 0) for bundle in bundles)

        raise RuntimeError(
            f"Failed to create placement group with {len(bundles)} bundles "
            f"(requiring {total_gpus} GPUs, {total_cpus} CPUs total) in {timeout} seconds. "
            f"This might indicate insufficient GPU resources.\n"
            f"Error: {e}"
        )


# NOTE (sumanthrh): For SGLang, the string representations here should also match those used by (and supported by) SGLang.
# This is because we do not control the update weight implementation with SGLang backend.
# With VLLM, we use a custom Worker extension to have a custom update weight implementation.
def torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float16:
        return "float16"
    elif dtype == torch.float32:
        return "float32"
    else:
        return str(dtype)


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    else:
        return torch.dtype(dtype)


def format_gib(mem_bytes: int) -> str:
    return f"{mem_bytes / (1024 ** 3):.2f} GiB"


def print_mem(tag: str, mem: dict):
    print(
        f"{tag} - Allocated: {format_gib(mem['allocated'])}, "
        f"Reserved: {format_gib(mem['reserved'])}, "
        f"Free: {format_gib(mem['free'])}, "
        f"Total: {format_gib(mem['total'])}"
    )


def run_p2p_access_check():
    device_count = torch.cuda.device_count()
    if device_count < 2:
        return False

    # Check P2P access between all GPU pairs
    for i in range(device_count):
        for j in range(device_count):
            if i != j:
                # This checks if device i can access device j's memory
                can_access = torch.cuda.can_device_access_peer(i, j)
                if not can_access:
                    return False

    return True


def peer_access_supported(max_num_gpus_per_node: int):
    # whatever the max num gpus per node is, we can check p2p access if there are at least 2 GPUs
    # if max is 1, p2p access is not supported
    if max_num_gpus_per_node <= 1:
        return False

    if not torch.cuda.is_available():
        # we are on cpu head node, so we need to check P2P access on a node with 2 GPUs
        ray.init()
        pg = ray.get(placement_group([{"CPU": 1, "GPU": 2}], strategy="PACK").ready(), timeout=120)
        result = ray.get(
            ray.remote(num_gpus=2, scheduling_strategy=PlacementGroupSchedulingStrategy(pg))(
                run_p2p_access_check
            ).remote()
        )
        ray.shutdown()
        return result
    else:
        return run_p2p_access_check()


def update_model_config(module_config, override_config_kwargs):
    """Update the module config with the override_config_kwargs.
    Args:
        module_config: The module config from Huggingface Transformers.
        override_config_kwargs: The kwargs to override the module config.
    """
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            setattr(module_config, key, val)
