"""ReasoningGym dataset integration for SkyRL training."""

from typing import Optional, Dict, Any, Union
from datasets import Dataset

import reasoning_gym
from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer
REASONING_GYM_AVAILABLE = True



class ReasoningGymDataset:
    """
    Dataset class that converts ReasoningGym's procedurally generated datasets
    into the format that SkyRL expects for training.
    
    This class handles the format conversion from ReasoningGym's procedural generation
    approach to SkyRL's static dataset expectation.
    """
    
    def __init__(
        self,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
    ):
        """
        Initialize the ReasoningGym dataset.
        
        Args:
            procedural_dataset: ReasoningGym ProceduralDataset object
            experiment: ReasoningGym Experiment object
            developer_prompt: System prompt to prepend to all examples
            developer_role: Role for the developer prompt (default: "system")
        """
            
        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"
        
        self.data_source = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        
        # Convert to SkyRL format
        self.skyrl_dataset = self._convert_to_skyrl_format()
    
    def _convert_to_skyrl_format(self) -> Dataset:
        """
        Convert ReasoningGym dataset to SkyRL format.
        
        Returns:
            Dataset: Dataset in SkyRL format
        """
        skyrl_examples = []
        
        for idx, entry in enumerate(self.data_source):
            question = entry.get("question", "")
            if not question:
                continue
                
            prompt = []
            if self.developer_prompt is not None:
                prompt.append({"role": self.developer_role, "content": self.developer_prompt})
            prompt.append({"role": "user", "content": question})
            
            ground_truth = entry.get("answer", "")
            if not ground_truth:
                solution = entry.get("solution", "")
                if solution:
                    try:
                        ground_truth = extract_answer(solution, tag_name="answer")
                    except Exception:
                        ground_truth = solution.strip()
            
            # Create SkyRL format entry
            skyrl_entry = {
                "data_source": "reasoning_gym",
                "prompt": prompt,
                "env_class": "reasoning_gym",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "index": idx,
                    "dataset_entry": entry,  # Store original entry for scoring
                    "question": question,
                    "solution": entry.get("solution", ""),
                },
            }
            
            skyrl_examples.append(skyrl_entry)
        
        return Dataset.from_list(skyrl_examples)
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.skyrl_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset example by index."""
        return self.skyrl_dataset[idx]
    
    def score_generation(self, idx: int, model_output: str) -> float:
        """Score a single generation using ReasoningGym."""
        skyrl_entry = self.skyrl_dataset[idx]
        original_entry = skyrl_entry["extra_info"]["dataset_entry"]
        
        try:
            found_answer = extract_answer(model_output, tag_name="answer")
        except Exception:
            found_answer = model_output

        if hasattr(self.data_source, "score_answer"):
            reward = self.data_source.score_answer(found_answer, entry=original_entry)
            return float(reward)
        return 0

    

def make_reasoning_gym_dataset(
    data_source,
    developer_prompt: str,
) -> ReasoningGymDataset:
    """
    Create ReasoningGymDataset object using either a ProceduralDataset or Experiment as the underlying data source.
    
    Args:
        data_source: Either a ProceduralDataset or Experiment from ReasoningGym
        developer_prompt: System prompt to prepend to all examples
        
    Returns:
        ReasoningGymDataset: Dataset in SkyRL format
    """
    
    if isinstance(data_source, Experiment):
        return ReasoningGymDataset(
            experiment=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
        )
    else:
        return ReasoningGymDataset(
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
        )