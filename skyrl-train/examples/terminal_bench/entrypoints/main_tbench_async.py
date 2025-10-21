"""
Main entrypoint for async training on terminal bench tasks.
"""

import importlib
import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from examples.terminal_bench.generator.terminal_bench_generator import TerminalBenchGenerator

# TODO(tgriggs): Rename async directory, move async trainer into skyrl-train package (out of examples)
# Import from the examples.async module using importlib since 'async' is a reserved keyword
_async_main = importlib.import_module('examples.async.main_async')
AsyncPPOExp = _async_main.AsyncPPOExp


class AsyncTerminalBenchExp(AsyncPPOExp):
    """
    Combines AsyncPPOTrainer + TerminalBenchGenerator.
    """
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the TerminalBenchGenerator.
        """
        return TerminalBenchGenerator(
            generator_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = AsyncTerminalBenchExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
