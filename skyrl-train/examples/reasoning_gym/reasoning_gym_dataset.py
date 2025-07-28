from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.dataset import PromptDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset
from typing import Optional
from transformers import PreTrainedTokenizer
import torch
from skyrl_train.utils.data_utils import StatefulDataLoader

# Import reasoning gym related modules
from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.dataset import ProceduralDataset


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"

        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        row_dict = self.data[index].copy()
        question = row_dict["question"]

        # Build the prompt in chat format
        prompt = []
        if self.developer_prompt is not None:
            prompt.append({"role": self.developer_role, "content": self.developer_prompt})
        prompt.append({"role": "user", "content": question})

        # Return in SkyRL format similar to gsm8k_dataset.py
        return {
            "data_source": "reasoning_gym",
            "prompt": prompt,
            "env_class": "reasoning_gym",
            "reward_spec": {
                "method": "reasoning_gym_score",
                "original_item": row_dict,  # Store original item for scoring
            },
            "extra_info": {
                "index": index,
                "question": question,
                "original_item": row_dict,
            }
        }


class ReasoningGymTrainer(RayPPOTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, dataset: PromptDataset, is_train=True):
        """
        Build the dataloader for the training or evaluation dataset
        """
        # prepare dataloader
        batch_size = self.cfg.trainer.train_batch_size if is_train else self.cfg.trainer.eval_batch_size

        # Seed the dataloader for reproducibility.
        seeded_generator = torch.Generator()
        seeded_generator.manual_seed(self.cfg.trainer.seed)

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if is_train else False,
            collate_fn=dataset.collate_fn,
            num_workers=8,
            drop_last=True if is_train else False,
            generator=seeded_generator,
        )
        if is_train:
            self.total_training_steps = len(dataloader) * self.cfg.trainer.epochs
            print(f"Total steps: {self.total_training_steps}")
        else:
            print(f"Validation set size: {len(dataloader)}")

        return dataloader
      