"""SkyRL-Train backend for TinkerEngine.

Uses SkyRL-Train infrastructure for supervised training with cross-entropy loss.
Currently supports a single model only.
"""

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
    from skyrl_train.utils.utils import initialize_ray
    from skyrl_train.config.utils import get_default_config
    from skyrl_train.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config

    SKYRL_TRAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in non-ray installs
    ray = None
    placement_group = None
    TrainingInputBatch = Any
    PPORayActorGroup = Any
    WorkerDispatch = Any
    PolicyWorker = Any
    get_ray_pg_ready_with_timeout = None
    initialize_ray = None
    get_default_config = None
    SKYRL_RAY_PG_TIMEOUT_IN_S = None
    InferenceEngineClient = Any
    create_ray_wrapped_inference_engines_from_config = None
    SKYRL_TRAIN_AVAILABLE = False


class SkyRLTrainBackendConfig(BaseModel, extra="forbid"):
    """Configuration for the SkyRL-Train backend.

    Attributes:
        num_gpus: Number of GPUs to use for training (default: 4)
        enable_inference: Whether to create inference engines for sampling (default: False, SFT-only mode)
        backend: Inference backend to use when enable_inference=True (default: "vllm")
    """

    num_gpus: int = 4
    enable_inference: bool = False
    backend: str = "vllm"


def _build_config(
    base_model: str,
    config: SkyRLTrainBackendConfig,
    lora_config: types.LoraConfig | None = None,
):
    """Build config for SkyRL-Train workers using default config.

    Args:
        base_model: HuggingFace model path
        config: Backend configuration
        lora_config: LoRA configuration if using LoRA
    """
    cfg = get_default_config()
    cfg.trainer.policy.model.path = base_model

    # Disable scheduler - Tinker manages learning rate externally via set_lr()
    cfg.trainer.policy.optimizer_config.scheduler = "constant"
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0

    # Configure placement
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = config.num_gpus

    if config.enable_inference:
        # Colocated training + inference mode
        cfg.trainer.placement.colocate_all = True

        # Configure inference engines to use same number of GPUs
        cfg.generator.num_inference_engines = config.num_gpus
        cfg.generator.inference_engine_tensor_parallel_size = 1
        cfg.generator.inference_engine_pipeline_parallel_size = 1
        cfg.generator.inference_engine_data_parallel_size = 1
        cfg.generator.inference_engine_expert_parallel_size = 1
        cfg.generator.gpu_memory_utilization = 0.8
        cfg.generator.backend = config.backend
        cfg.generator.run_engines_locally = True
    else:
        # SFT-only mode: no inference engines, no colocation
        cfg.trainer.placement.colocate_all = False

    return cfg


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
        self.num_gpus = config.num_gpus
        self._model_id: str | None = None
        self._model_metadata: types.ModelMetadata | None = None
        self._actor_group: PPORayActorGroup | None = None
        self._dispatch: WorkerDispatch | None = None
        self._inference_engine_client: InferenceEngineClient | None = None
        self._placement_group = None
        self._cfg = None
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)

    def has_model(self, model_id: str) -> bool:
        return self._model_id == model_id

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._model_id is not None:
            raise ValueError(f"Model '{self._model_id}' already exists. Only one model supported.")

        # Build config
        self._cfg = _build_config(self.base_model, self.config, lora_config)
        num_nodes = self._cfg.trainer.placement.policy_num_nodes
        num_gpus_per_node = self._cfg.trainer.placement.policy_num_gpus_per_node

        if self.config.enable_inference:
            # Mode 1: Colocated training + inference (for RL)
            self._create_model_with_inference(model_id, lora_config, num_nodes, num_gpus_per_node)
        else:
            # Mode 2: SFT-only (no inference engines)
            self._create_model_sft_only(model_id, lora_config, num_nodes, num_gpus_per_node)

    def _create_model_sft_only(self, model_id: str, lora_config: types.LoraConfig, num_nodes: int, num_gpus_per_node: int) -> None:
        """Create model for SFT-only mode (no inference engines, simpler setup)."""
        logger.info(f"Creating model in SFT-only mode with {num_nodes} nodes, {num_gpus_per_node} GPUs/node")

        # Initialize Ray with proper runtime environment (critical for worker initialization)
        if not ray.is_initialized():
            logger.info("Initializing Ray with runtime environment")
            initialize_ray(self._cfg)

        # Create simple placement group for training workers
        total_gpus = num_nodes * num_gpus_per_node
        pg = placement_group([{"GPU": 1, "CPU": 1}] * total_gpus, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        self._placement_group = pg

        # Create training workers (full GPU per actor, no colocation)
        self._actor_group = PPORayActorGroup(
            cfg=self._cfg,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=1,  # Full GPU per actor (no sharing)
            colocate_all=False,
            sequence_parallel_size=self._cfg.trainer.policy.sequence_parallel_size,
            record_memory=self._cfg.trainer.policy.record_memory,
        )

        # Initialize model
        ray.get(self._actor_group.async_init_model(self.base_model))

        # Create dispatch layer (no inference client)
        self._dispatch = WorkerDispatch(self._cfg, policy_actor_group=self._actor_group)

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id} in SFT-only mode")

    def _create_model_with_inference(self, model_id: str, lora_config: types.LoraConfig, num_nodes: int, num_gpus_per_node: int) -> None:
        """Create model with colocated training + inference (for RL)."""
        # Initialize Ray with proper runtime environment (critical for worker initialization)
        if not ray.is_initialized():
            logger.info("Initializing Ray with runtime environment")
            initialize_ray(self._cfg)

        # Create placement group based on inference engine configuration (like main_base.py)
        num_inference_engines = self._cfg.generator.num_inference_engines
        tensor_parallel_size = self._cfg.generator.inference_engine_tensor_parallel_size
        pipeline_parallel_size = self._cfg.generator.inference_engine_pipeline_parallel_size
        data_parallel_size = self._cfg.generator.inference_engine_data_parallel_size

        total_gpu_slots = (
            num_inference_engines * tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        )

        logger.info(f"Creating placement group with {total_gpu_slots} GPU slots for colocated training+inference")
        pg = placement_group([{"GPU": 1, "CPU": 1}] * total_gpu_slots, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        self._placement_group = pg

        # Create inference engines using the placement group
        logger.info(f"Creating {num_inference_engines} inference engines")
        self._inference_engine_client = InferenceEngineClient(
            create_ray_wrapped_inference_engines_from_config(self._cfg, pg, self._tokenizer),
            self._tokenizer,
            self._cfg,
        )

        # Create training workers with fractional GPUs (0.2) to share with inference engines
        logger.info(f"Creating training workers with {num_nodes} nodes, {num_gpus_per_node} GPUs/node (fractional allocation)")
        self._actor_group = PPORayActorGroup(
            cfg=self._cfg,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=0.2,  # Fractional GPUs to share with inference engines
            colocate_all=True,
            sequence_parallel_size=self._cfg.trainer.policy.sequence_parallel_size,
            record_memory=self._cfg.trainer.policy.record_memory,
        )

        # Initialize model and offload to CPU (inference engines occupy GPU initially)
        ray.get(self._actor_group.async_init_model(self.base_model))
        self._actor_group.offload_to_cpu()

        # Create dispatch layer
        self._dispatch = WorkerDispatch(
            self._cfg,
            policy_actor_group=self._actor_group,
            inference_engine_client=self._inference_engine_client,
        )
        self._dispatch.mark_all_offloaded()  # Policy starts on CPU when colocating

        # Initialize weight sync between training and inference
        self._dispatch.init_weight_sync_state(self._inference_engine_client)

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id} with colocated training+inference")

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
        if not self.config.enable_inference:
            raise NotImplementedError(
                "Sampling not supported in SFT-only mode. "
                "Set enable_inference=true in backend_config to enable inference engines for sampling."
            )
        # Inference-enabled mode: implemented in tyler/tinker-sampling-main branch
        raise NotImplementedError("Sampling implementation will be merged from tyler/tinker-sampling-main")

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
