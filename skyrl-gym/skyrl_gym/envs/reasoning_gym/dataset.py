"""ReasoningGym dataset integration for SkyRL training."""

from typing import Optional, Dict, Any, Union
from datasets import Dataset, Features, Value, LargeList

import reasoning_gym
import json
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
        skyrl_examples = []

        for idx, entry in enumerate(self.data_source):
            question = entry.get("question", "")
            if not question:
                continue

            # Build prompt structure in conversation format
            prompt = []
            if self.developer_prompt is not None:
                prompt.append({"role": self.developer_role, "content": self.developer_prompt})
            prompt.append({"role": "user", "content": question})

            # Extract ground truth answer
            ground_truth = entry.get("answer", "")
            if not ground_truth:
                solution = entry.get("solution", "")
                if solution:
                    try:
                        ground_truth = extract_answer(solution, tag_name="answer")
                    except Exception:
                        ground_truth = solution.strip()

            entry_str = json.dumps(entry)
            skyrl_entry = {
                "data_source": "reasoning_gym",
                "prompt": prompt, 
                "env_class": "gsm8k",
                "reward_spec": {
                    "method": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "index": idx,
                    "dataset_entry": entry_str,
                    "question": question,
                    "solution": entry.get("solution", ""),
                },
            }

            skyrl_examples.append(skyrl_entry)
    

        features = Features({
            "data_source": Value("string"),
            "prompt": LargeList(
                Features({
                    "role": Value("string"),
                    "content": Value("string"),
                })
            ),
            "env_class": Value("string"),
            "reward_spec": Features({
                "method": Value("string"),
                "ground_truth": Value("string"),
            }),
            "extra_info": Features({
                "index": Value("int32"),
                "dataset_entry": Value("string"),
                "question": Value("string"),
                "solution": Value("string"),
            })
        })
        
        dataset = Dataset.from_list(skyrl_examples, features=features)
        return dataset

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
        original_entry = json.loads(original_entry)
        try:
            found_answer = extract_answer(model_output, tag_name="answer")
        except Exception:
            found_answer = model_output

        if hasattr(self.data_source, "score_answer"):
            try:
                reward = self.data_source.score_answer(found_answer, entry=original_entry)
                return float(reward)
            except Exception:
                pass
        
        return 0.0
    
    def write_to_parquet(self, path: str):
        """
        Write the SkyRL-formatted dataset to a Parquet file.
        
        Args:
            path (str): The file path to write the dataset to.
        """
        self.skyrl_dataset.to_parquet(path)


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
