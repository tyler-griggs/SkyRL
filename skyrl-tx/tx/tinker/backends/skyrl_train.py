"""SkyRL-Train backend for TinkerEngine.

Uses SkyRL-Train infrastructure for supervised training with cross-entropy loss.
Currently supports a single model only.
"""

import asyncio
import os
import tarfile
import tempfile
from typing import Any

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from tx.tinker import types
from tx.tinker.backends.backend import AbstractBackend
from tx.utils.log import logger

try:  # Optional dependency: keep other backends importable without ray/skyrl-train.
    import ray
    from ray.util.placement_group import placement_group
    from skyrl_train.training_batch import TrainingInputBatch
    from skyrl_train.workers.worker import PPORayActorGroup
    from skyrl_train.workers.worker_dispatch import WorkerDispatch
    from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
    from skyrl_train.utils import get_ray_pg_ready_with_timeout
    from skyrl_train.config.utils import get_default_config
    from skyrl_train.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S

    SKYRL_TRAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in non-ray installs
    ray = None
    placement_group = None
    TrainingInputBatch = Any
    PPORayActorGroup = Any
    WorkerDispatch = Any
    PolicyWorker = Any
    get_ray_pg_ready_with_timeout = None
    get_default_config = None
    SKYRL_RAY_PG_TIMEOUT_IN_S = None
    SKYRL_TRAIN_AVAILABLE = False


class SkyRLTrainBackendConfig(BaseModel, extra="forbid"):
    """Configuration for the SkyRL-Train backend."""

    # Inference engine configuration
    num_inference_engines: int = 0  # 0 = SFT-only (no sampling)
    inference_engine_tensor_parallel_size: int = 1
    inference_engine_pipeline_parallel_size: int = 1
    inference_engine_data_parallel_size: int = 1

    # Backend selection
    inference_backend: str = "vllm"  # "vllm" or "sglang"

    # vLLM/SGLang settings
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False
    enforce_eager: bool = False


def _build_config(base_model: str, config: SkyRLTrainBackendConfig, lora_config: types.LoraConfig | None = None):
    """Build config for SkyRL-Train workers using default config."""
    cfg = get_default_config()
    cfg.trainer.policy.model.path = base_model

    # Disable scheduler - Tinker manages learning rate externally via set_lr()
    cfg.trainer.policy.optimizer_config.scheduler = "constant"
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0

    return cfg


def _create_inference_engines(
    base_model: str,
    config: SkyRLTrainBackendConfig,
    tokenizer,
    cfg,
    shared_pg,
):
    """Create inference engines for sampling."""
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

    # Get colocate_all from cfg if available, otherwise infer from config
    colocate_all = cfg.trainer.placement.colocate_all if hasattr(cfg.trainer.placement, "colocate_all") else False

    engine_kwargs = {
        "num_inference_engines": config.num_inference_engines,
        "tensor_parallel_size": config.inference_engine_tensor_parallel_size,
        "pipeline_parallel_size": config.inference_engine_pipeline_parallel_size,
        "data_parallel_size": config.inference_engine_data_parallel_size,
        "model_dtype": "bfloat16",  # TODO: Make configurable
        "pretrain": base_model,
        "seed": 42,  # TODO: Make configurable
        "vllm_v1_disable_multiproc": True,
        "enable_prefix_caching": config.enable_prefix_caching,
        "enforce_eager": config.enforce_eager,
        "shared_pg": shared_pg if colocate_all else None,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "inference_engine_enable_sleep": colocate_all,  # Enable sleep if colocated
        "async_engine": True,  # Always use async for Tinker
        "tokenizer": tokenizer,
        "backend": config.inference_backend,
    }

    engines = create_ray_wrapped_inference_engines(**engine_kwargs)
    return InferenceEngineClient(engines, tokenizer, cfg)


class SkyRLTrainBackend(AbstractBackend):
    """SkyRL-Train backend for supervised training."""

    def __init__(self, base_model: str, config: SkyRLTrainBackendConfig):
        logger.warning("=" * 80)
        logger.warning("SkyRLTrainBackend is currently EXPERIMENTAL!")
        logger.warning("=" * 80)

        if not SKYRL_TRAIN_AVAILABLE or ray is None:
            raise ImportError(
                "SkyRLTrainBackend requires `ray`. Install the appropriate extras (e.g. `--extra skyrl_train`)."
            )

        self.base_model = base_model
        self.config = config
        self._model_id: str | None = None
        self._model_metadata: types.ModelMetadata | None = None
        self._actor_group: PPORayActorGroup | None = None
        self._dispatch: WorkerDispatch | None = None
        self._cfg = None
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._inference_engine_client = None  # InferenceEngineClient for sampling

    def has_model(self, model_id: str) -> bool:
        return self._model_id == model_id

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._model_id is not None:
            raise ValueError(f"Model '{self._model_id}' already exists. Only one model supported.")

        self._cfg = _build_config(self.base_model, self.config, lora_config)
        num_nodes = self._cfg.trainer.placement.policy_num_nodes
        num_gpus = self._cfg.trainer.placement.policy_num_gpus_per_node

        pg = placement_group([{"GPU": 1, "CPU": 1}] * num_nodes * num_gpus, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

        self._actor_group = PPORayActorGroup(
            cfg=self._cfg,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=1,
            colocate_all=True,
            sequence_parallel_size=self._cfg.trainer.policy.sequence_parallel_size,
            record_memory=self._cfg.trainer.policy.record_memory,
        )
        ray.get(self._actor_group.async_init_model(self.base_model))
        self._dispatch = WorkerDispatch(self._cfg, policy_actor_group=self._actor_group)

        # Create inference engines if sampling is enabled
        if self.config.num_inference_engines > 0:
            logger.info(f"Creating {self.config.num_inference_engines} inference engines for sampling")
            self._inference_engine_client = _create_inference_engines(
                self.base_model, self.config, self._tokenizer, self._cfg, pg
            )
            # Register with WorkerDispatch for weight sync
            self._dispatch.set_inference_engine_client(self._inference_engine_client)
            self._dispatch.init_weight_sync_state(self._inference_engine_client)
            logger.info("Inference engines initialized successfully")
        else:
            logger.info("Sampling disabled (num_inference_engines=0), SFT-only mode")

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id}")

    def delete_model(self, model_id: str) -> None:
        if self._model_id != model_id:
            raise ValueError(f"Model {model_id} not found")
        raise NotImplementedError("Deleting models not yet implemented")

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch) -> TrainingInputBatch:
        """Convert PreparedModelPassBatch to TrainingInputBatch."""
        if not prepared_batch.all_input_ids:
            return TrainingInputBatch({})

        # SkyRL-Train shifts internally, so provide the full sequence length by
        # appending the last target token to each already-shifted input.
        full_sequences = [
            list(input_ids) + ([targets[-1]] if targets else [])
            for input_ids, targets in zip(prepared_batch.all_input_ids, prepared_batch.all_targets)
        ]

        max_seq_len = max(len(seq) for seq in full_sequences)
        max_response_len = max(len(weights) for weights in prepared_batch.all_token_weights)

        sequences, attention_masks, loss_masks, response_masks = [], [], [], []

        for seq, weights in zip(full_sequences, prepared_batch.all_token_weights):
            pad_len = max_seq_len - len(seq)
            sequences.append([self._tokenizer.pad_token_id] * pad_len + list(seq))
            attention_masks.append([0] * pad_len + [1] * len(seq))
            action_pad = max_response_len - len(weights)
            loss_masks.append([0.0] * action_pad + [float(w) for w in weights])
            response_masks.append([0] * action_pad + [1] * len(weights))

        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.float32)
        response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)

        batch = TrainingInputBatch(
            {
                "sequences": sequences_tensor,
                "attention_mask": attention_mask_tensor,
                "loss_mask": loss_mask_tensor,
                "response_mask": response_mask_tensor,
            }
        )
        batch.metadata = {"response_length": max_response_len}
        return batch

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        loss_fn: str = "cross_entropy",
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_input_ids:
            return {}

        batch = self._to_training_batch(prepared_batch)
        data = self._dispatch.forward_backward("policy", batch, loss_fn=loss_fn)

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                raw_output = data["loss_fn_outputs"][i]
                logprobs = list(raw_output.get("logprobs", []))
                elementwise_loss = list(raw_output.get("elementwise_loss", []))
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {
                            "data": elementwise_loss,
                            "dtype": "float32",
                            "shape": [len(elementwise_loss)],
                        },
                        "logprobs": {
                            "data": logprobs,
                            "dtype": "float32",
                            "shape": [len(logprobs)],
                        },
                    }
                )
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )
        return results

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        raise NotImplementedError("Forward-only pass not supported")

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")

        # Apply learning rate from AdamParams before optimizer step
        # Note: beta1, beta2, eps are fixed at optimizer creation and cannot be changed dynamically
        adam_params = request_data.adam_params
        self._dispatch.set_lr("policy", adam_params.learning_rate)

        grad_norm = self._dispatch.optim_step("policy")
        logger.info(f"optim_step: lr={adam_params.learning_rate}, grad_norm={grad_norm}")
        return types.OptimStepOutput()

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Generate samples using InferenceEngineClient.

        NOTE: Weight sync is NOT triggered automatically. The caller must call
        save_weights_for_sampler() explicitly before calling sample() if weights
        have been updated.
        """
        # 1. Validate inference is enabled
        if self._inference_engine_client is None:
            error = types.ErrorResponse(
                error="Sampling not enabled. Set num_inference_engines > 0 in backend_config.",
                status="error",
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}

        # 2. Validate single model
        unique_models = set(prepared_batch.all_model_ids)
        if len(unique_models) != 1 or list(unique_models)[0] != self._model_id:
            error = types.ErrorResponse(
                error=f"Model mismatch. Expected {self._model_id}, got {unique_models}", status="error"
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}

        # 3. Sample each prompt
        sample_outputs = []
        for i in range(len(prepared_batch.all_prompts)):
            prompt = prepared_batch.all_prompts[i]
            sampling_params = prepared_batch.all_sampling_params[i]

            # Convert to InferenceEngineClient format
            params_dict = {
                "temperature": sampling_params.temperature,
                "max_tokens": sampling_params.max_tokens,
            }

            if sampling_params.top_k is not None:
                params_dict["top_k"] = sampling_params.top_k
            if sampling_params.top_p is not None:
                params_dict["top_p"] = sampling_params.top_p
            if sampling_params.stop:
                params_dict["stop"] = sampling_params.stop

            try:
                # Call InferenceEngineClient.sample()
                output = asyncio.run(
                    self._inference_engine_client.sample(
                        prompt_token_ids=prompt,
                        num_samples=1,  # Tinker batches multiple samples separately
                        sampling_params=params_dict,
                    )
                )
                sample_outputs.append(output)
            except Exception as e:
                logger.error(f"Sampling failed for prompt {i}: {e}")
                sample_outputs.append(None)

        # 4. Aggregate results by request
        return self._aggregate_sample_results(prepared_batch, sample_outputs)

    def _aggregate_sample_results(
        self,
        prepared_batch: types.PreparedSampleBatch,
        sample_outputs: list,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Convert InferenceEngineClient outputs to Tinker format."""
        results = {}

        for request_id, model_id, start_idx, end_idx, needs_prompt_logprobs in prepared_batch.request_batch_slices:
            sequences = []
            has_error = False
            error_msg = None

            for i in range(start_idx, end_idx):
                output = sample_outputs[i]

                if output is None:
                    has_error = True
                    error_msg = f"Sampling failed for sample {i}"
                    break

                # Extract tokens and logprobs
                response_tokens = output["response_ids"][0]
                response_logprobs = output.get("response_logprobs", [[]])[0] if output.get("response_logprobs") else []
                stop_reason_raw = output["stop_reasons"][0]

                # Map vLLM stop reason to Tinker format
                stop_reason = "stop" if stop_reason_raw in ["stop", "stop_token"] else "length"

                # Ensure logprobs exist (critical for RL)
                if response_logprobs is None or len(response_logprobs) == 0:
                    logger.warning("No logprobs returned - filling with zeros")
                    response_logprobs = [0.0] * len(response_tokens)

                sequences.append(
                    types.GeneratedSequence(
                        tokens=response_tokens,
                        logprobs=response_logprobs,
                        stop_reason=stop_reason,
                    )
                )

            if has_error:
                results[request_id] = types.ErrorResponse(
                    error=error_msg or "Unknown sampling error",
                    status="error",
                )
            else:
                # Note: prompt_logprobs not supported initially
                if needs_prompt_logprobs:
                    logger.warning("Prompt logprobs requested but not yet supported")

                results[request_id] = types.SampleOutput(
                    sequences=sequences,
                    prompt_logprobs=None,
                )

        return results

    async def save_weights_for_sampler(self, model_id: str) -> None:
        """Sync training weights to inference engines for sampling.

        This performs in-memory weight sync from training GPUs to inference GPUs.
        This is DIFFERENT from save_sampler_checkpoint() which saves to disk.

        Called explicitly by the caller (e.g., after training, before sampling).
        """
        self._validate_model_state(model_id)

        if self._inference_engine_client is None:
            raise RuntimeError("No inference engines configured. Cannot sync weights for sampling.")

        # Call WorkerDispatch to sync weights
        await self._dispatch.save_weights_for_sampler()
        logger.info(f"Synced weights for {model_id} to inference engines")

    def _validate_model_state(self, model_id: str) -> None:
        """Validate that model exists and is initialized."""
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")
        if self._dispatch is None:
            raise RuntimeError("Model not initialized")

    def _create_tar_from_directory(self, source_dir: str, output_path: str) -> None:
        """Create an uncompressed tar archive from a directory."""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use uncompressed tar - gzip adds 5-10min CPU time on 6-7GB FSDP checkpoints
        with tarfile.open(output_path, "w") as tar:
            tar.add(source_dir, arcname=".")

    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save full training checkpoint (model + optimizer + scheduler) as tar."""
        self._validate_model_state(model_id)

        # Create temp directory for checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "checkpoint")

            # Save checkpoint directory (includes optimizer state automatically)
            self._dispatch.save_checkpoint(model="policy", ckpt_dir=ckpt_dir, tokenizer=self._tokenizer)

            # Create tar archive
            self._create_tar_from_directory(ckpt_dir, output_path)

        logger.info(f"Saved checkpoint for {model_id} to {output_path}")

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load full training checkpoint (model + optimizer + scheduler) from tar."""
        self._validate_model_state(model_id)

        # Extract tar to temp directory (filter='data' prevents path traversal attacks)
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(checkpoint_path, "r") as tar:
                tar.extractall(temp_dir, filter="data")

            # Load checkpoint (includes optimizer and scheduler states)
            self._dispatch.load_checkpoint(
                model="policy", ckpt_dir=temp_dir, load_optimizer_states=True, load_lr_scheduler_states=True
            )

        logger.info(f"Loaded checkpoint for {model_id} from {checkpoint_path}")

    def save_sampler_checkpoint(self, output_path, model_id: str) -> None:
        """Save sampler checkpoint as tar (model only, no optimizer)."""
        self._validate_model_state(model_id)

        # Create temp directory for HuggingFace export
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_dir = os.path.join(temp_dir, "model")

            # Save in HuggingFace format (model weights + tokenizer only)
            self._dispatch.save_hf_model(model="policy", hf_model_dir=hf_dir, tokenizer=self._tokenizer)

            # Create tar archive
            self._create_tar_from_directory(hf_dir, output_path)

        logger.info(f"Saved sampler checkpoint for {model_id} to {output_path}")
