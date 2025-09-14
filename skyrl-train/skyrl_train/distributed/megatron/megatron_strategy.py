import os
import random
from datetime import timedelta
from typing import List, Union, Optional
from jaxtyping import Float

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import distributed as dist

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.models import Actor
from skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl_train.utils import io
import megatron.core.parallel_state as mpu
from skyrl_train.distributed.megatron.megatron_utils import (
    offload_megatron_model_to_cpu,
    load_megatron_model_to_gpu,
    offload_megatron_optimizer,
    load_megatron_optimizer,
)
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from transformers import GenerationConfig

class MegatronStrategy(DistributedStrategy):
    """
    The strategy for training with Megatron.
    """

    def __init__(
        self,
        megatron_config,
        optimizer_config=None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.megatron_config = megatron_config
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.hf_config = None  # To be set once configs are initialized.

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.device_count() > 0:
            from megatron.core import tensor_parallel

            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=self.megatron_config.context_parallel_size,
            nccl_communicator_config_path=None,
        )
        self.set_seed(self.seed)
        self.world_size = dist.get_world_size()

    def offload_to_cpu(self, model, optimizer, pin_memory=True, non_blocking=True):
        """
        Offload model weights and optimizer to CPU memory.
        """
        offload_megatron_model_to_cpu(model)
        if optimizer is not None:
            offload_megatron_optimizer(optimizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, non_blocking=True):
        """Reload model weights back to GPU."""
        load_megatron_model_to_gpu(model)
        if optimizer is not None:
            load_megatron_optimizer(optimizer)
        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        raise NotImplementedError()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        _, grad_norm, _ = optimizer.step()
        scheduler.step(1)
        optimizer.zero_grad()
        return grad_norm

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        raise NotImplementedError()



    def save_hf_configs(self, ckpt_dir: str, tokenizer=None):
        """
        Save model and tokenizer configs to ckpt_dir/huggingface

        Args:
            ckpt_dir: str - the directory to save the configs to
            tokenizer: AutoTokenizer - tokenizer to save
        """
        hf_config_tokenizer_path = os.path.join(ckpt_dir, "huggingface")
        io.makedirs(hf_config_tokenizer_path, exist_ok=True)

        with io.local_work_dir(hf_config_tokenizer_path) as work_dir:
            self.hf_config.save_pretrained(work_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(work_dir)

            if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)
                    # with io.local_work_dir(hf_config_tokenizer_path) as work_dir:
                    generation_config.save_pretrained(work_dir)
                except Exception as e:
                    # if the generation config isn't available, we don't save it
                    # TODO(tgriggs): Unify this under whatever logging system we use.
                    print(f"Could not save generation config for '{self.hf_config.name_or_path}'. Error: {e}")
                    pass


    # TODO(tgriggs): Consider renaming ckpt -> checkpoint every where. 
    # TODO(tgriggs): Type-hinting?
    # TODO(tgriggs): Change prints to logs
    # TODO(tgriggs): Msg Eric, let's rename MegatronPPOPolicy as well?
    # TODO(tgriggs): prune args as needed?
    def save_ckpt(
        self,
        model,
        ckpt_dir,
        global_step,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
        tokenizer=None,
    ):
        if isinstance(model, Actor):
            model = model.model
        if hasattr(model, "actor_module"):
            model = model.actor_module
        assert len(model) == 1, "Megatron virtual pipeline model parallel is not yet supported"
        model = model[0]
        
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        dist.barrier()
        
        rank = self.get_rank()
        world_size = self.world_size
        
        # TODO(tgriggs): Understand then update these comments.
        # All ranks Save Model to reduce memory pressure
        # Get sharded state dict, notice that state_dict will collect among dp groups, causing memory pressure
        sharded_state_dict = {}
        sharded_state_dict["model"] = model.sharded_state_dict()
        
        if optimizer:
            # TODO(tgriggs): Where is ``sharded_state_dict`` coming from?
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(sharded_state_dict)
        
        if scheduler:
            sharded_state_dict["scheduler"] = scheduler.state_dict()
            
        # Save client state and any additional info
        sharded_state_dict["client_state"] = client_state
        sharded_state_dict["tag"] = tag
        sharded_state_dict["global_step"] = global_step
        sharded_state_dict["rng"] = self.get_rng_state()
        # extra_state_dict = {
        #     "client_state": client_state,
        #     "tag": tag,
        #     # "world_size": world_size,
        #     # "rank": rank,
        #     "global_step": global_step,
        #     "rng": self.get_rng_state(),  # Add RNG state for reproducibility
        # }
        
        print(f"[Rank {rank}/{world_size}]: Generated state dict for saving: {sharded_state_dict.keys()}")
        print(f"[Rank {rank}/{world_size}]: Generated model state dict for saving: {sharded_state_dict['model'].keys()}")
        
        validate_sharding_integrity = True
        # Get checkpointing strategies
        save_strategy = get_default_save_sharded_strategy("torch_dist")
        save_strategy = FullyParallelSaveStrategyWrapper(
            save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )
        
        # Save state dicts
        async_save_request = dist_checkpointing.save(
            sharded_state_dict=sharded_state_dict,
            checkpoint_dir=ckpt_dir,
            sharded_strategy=save_strategy,
            async_sharded_save=False,  # TODO(tgriggs): Make async save configurable
            validate_access_integrity=validate_sharding_integrity,
        )
        assert async_save_request is None, "TODO(tgriggs): Async save is not yet supported for Megatron"
        dist.barrier()
        
        if rank == 0:
            self.save_hf_configs(ckpt_dir, tokenizer)
            print(f"[Rank {rank}/{world_size}]: Saved Huggingface config and tokenizer to {ckpt_dir}/huggingface")
        
        dist.barrier()
            

    def load_ckpt(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        if not ckpt_dir or not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        if isinstance(model, Actor):
            model = model.model
        if hasattr(model, "actor_module"):
            model = model.actor_module
        assert len(model) == 1, "Megatron virtual pipeline model parallel is not yet supported"
        
        sharded_state_dict = {}
        sharded_state_dict["model"] = model[0].sharded_state_dict()
        
        if optimizer and load_optimizer_states:
            # TODO(tgriggs): Where is ``sharded_state_dict`` coming from?
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(sharded_state_dict)
        
        if scheduler and load_lr_scheduler_states:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()
            
        rank = self.get_rank()
        world_size = self.world_size
        print(f"[Rank {rank}/{world_size}]: Generated state dict for loading: {sharded_state_dict.keys()}")
        print(f"[Rank {rank}/{world_size}]: Generated model state dict for loading: {sharded_state_dict['model'].keys()}")
        
        
        # Get checkpointing strategies
        load_strategy = get_default_load_sharded_strategy(ckpt_dir)
        load_strategy = FullyParallelLoadStrategyWrapper(
            load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )

        # Load model sharded state dicts
        state_dict = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=ckpt_dir, sharded_strategy=load_strategy)
        print(f"[Rank {rank}/{world_size}]: Loaded state dict with keys: {state_dict.keys()}")
        
        # TODO(tgriggs): add file path and state dict keys to these error logs so that people can debug
        assert "model" in state_dict, "Model state dict not found in loaded state dict"
        
        model[0].load_state_dict(state_dict["model"])
        print(f"[Rank {rank}/{world_size}]: Loaded model state dict with keys: {state_dict['model'].keys()}")
        
        if optimizer and load_optimizer_states:
            assert "optimizer" in state_dict, "Optimizer state dict not found in loaded state dict"
            optimizer.load_state_dict(state_dict["optimizer"])
            print(f"[Rank {rank}/{world_size}]: Loaded optimizer state dict with keys: {state_dict['optimizer'].keys()}")
            
        if scheduler and load_lr_scheduler_states:
            assert "lr_scheduler" in state_dict, "LR scheduler state dict not found in loaded state dict"
            scheduler.load_state_dict(state_dict["lr_scheduler"])
            print(f"[Rank {rank}/{world_size}]: Loaded LR scheduler state dict with keys: {state_dict['lr_scheduler'].keys()}")
            
        extra_states = {}
        for key in ["client_state", "tag", "global_step", "rng"]:
            if key in state_dict:
                extra_states[key] = state_dict[key]
                print(f"[Rank {rank}/{world_size}]: Loaded {key} state dict")
                
        return ckpt_dir, extra_states
                
        
    def save_hf_model(self, model: Union[Actor, nn.Module], output_dir: str, tokenizer=None, **kwargs) -> None:
        pass
