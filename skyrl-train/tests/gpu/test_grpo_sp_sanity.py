"""
uv run --isolated --extra vllm --extra dev -- pytest -s -vvv tests/gpu/test_grpo_sp_sanity.py
"""

import os
from hydra import initialize, compose
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.trainer import RayPPOTrainer
import ray
from skyrl_train.utils import Timer
from skyrl_train.utils.ppo_utils import normalize_advantages_dict


import asyncio


class TestExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return RayPPOTestTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def run(self):
        trainer = self._setup_trainer()
        return trainer.train()


class RayPPOTestTrainer(RayPPOTrainer):
    def train(self):
        self.all_metrics = {}
        self.all_timings = {}
        self.global_step = 1

        with Timer("setup_policy_and_generator"):
            self.setup_policy_and_generator()

        # Run just one iteration for testing
        for _, rand_prompts in enumerate(self.train_dataloader):
            with Timer("step", self.all_timings):
                # 0) make batch mesh-aligned
                rand_prompts = self._remove_tail_data(rand_prompts)

                # 1) generation phase
                with Timer("prepare_generator_input", self.all_timings):
                    generator_input, uids = self._prepare_generator_input(
                        self.cfg.generator.n_samples_per_prompt, rand_prompts
                    )
                with Timer("generate", self.all_timings):
                    generator_output = asyncio.run(self.generate(generator_input))

                # 2) postprocess rewards and basic metrics
                with Timer("postprocess_generator_output", self.all_timings):
                    generator_output = self.postprocess_generator_output(generator_output, uids)

                # debug example
                vis = self.tokenizer.decode(generator_output["response_ids"][0])
                print("example: ", vis)

                # 3) convert to training batch
                with Timer("convert_to_training_input", self.all_timings):
                    training_input = self.convert_to_training_input(generator_output, uids)

                # 4) forward passes to get values/logprobs/rewards
                with Timer("fwd_logprobs_values_reward", self.all_timings):
                    training_input = self.fwd_logprobs_values_reward(training_input)

                # 5) optional reward KL penalty
                if self.cfg.trainer.algorithm.use_kl_in_reward:
                    with Timer("apply_reward_kl_penalty", self.all_timings):
                        training_input = self.apply_reward_kl_penalty(training_input)

                # 6) advantages and returns (+ optional normalization)
                with Timer("compute_advantages_and_returns", self.all_timings):
                    training_input = self.compute_advantages_and_returns(training_input)
                    if self.cfg.trainer.algorithm.advantage_batch_normalize:
                        training_input = normalize_advantages_dict(training_input)

                # 7) train policy/critic
                with Timer("train_critic_and_policy", self.all_timings):
                    _ = self.train_critic_and_policy(training_input)

            # Return metrics after one iteration
            return self.all_metrics


def run_exp_and_get_metrics(exp: BasePPOExp, cfg: DictConfig):
    metrics = exp.run()
    # ray shutdown will clear all state for the ray session
    ray.shutdown()
    return metrics


def run_with_hydra(func, config_name: str):
    current_directory = Path(__file__).parent.absolute()
    abs_config_dir = Path(config_dir).absolute()
    relative_config_dir = os.path.relpath(abs_config_dir, current_directory)
    print("relative_config_dir: ", relative_config_dir)
    with initialize(version_base=None, config_path=relative_config_dir):
        cfg = compose(config_name=config_name)
        func(cfg)


def ppo_run(cfg: DictConfig) -> None:
    # Configure test settings
    cfg.trainer.train_batch_size = 8
    cfg.trainer.epochs = 1
    cfg.trainer.policy_mini_batch_size = 8
    cfg.trainer.eval_interval = -1
    cfg.trainer.eval_before_train = False
    cfg.trainer.logger = "console"
    # use zero temperature for consistency.
    # We will anyways only check for log probability values so this is fine.
    cfg.generator.sampling_params.temperature = 0.0
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.critic_num_gpus_per_node = 4
    cfg.trainer.placement.ref_num_gpus_per_node = 4
    cfg.generator.num_inference_engines = 2
    cfg.generator.inference_engine_tensor_parallel_size = 2
    cfg.generator.gpu_memory_utilization = 0.7

    # Run baseline (no sequence parallel)
    cfg.trainer.policy.sequence_parallel_size = 1
    exp_baseline = TestExp(cfg)
    metrics_baseline = run_exp_and_get_metrics(exp_baseline, cfg)
    print("Baseline metrics: ", metrics_baseline)

    # Run with sequence parallel
    cfg.trainer.policy.sequence_parallel_size = 2
    exp_sp = TestExp(cfg)
    metrics_sp = run_exp_and_get_metrics(exp_sp, cfg)
    print("Metrics with sequence parallel: ", metrics_sp)

    # Compare policy entropy values
    # NOTE: typical values are ~ 0.225 and ~ 0.228
    # Some difference can be due to ignoring attention mask with seq parallelism
    np.testing.assert_allclose(
        metrics_sp["policy/policy_entropy"], metrics_baseline["policy/policy_entropy"], atol=5e-3
    )


def test_ppo_run():
    run_with_hydra(ppo_run, "ppo_base_config")


if __name__ == "__main__":
    test_ppo_run()
