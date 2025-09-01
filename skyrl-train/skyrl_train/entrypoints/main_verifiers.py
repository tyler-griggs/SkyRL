"""
uv run --isolated --extra vllm -m examples.async.main_async
"""

import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray


class VerifiersPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """
        from skyrl_train.generators.verifiers_generator import VerifiersGenerator

        return VerifiersGenerator(
            generator_cfg=cfg.generator,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = VerifiersPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
