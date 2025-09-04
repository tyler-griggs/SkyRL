from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
from omegaconf import DictConfig
import json
from reasoning_gym.utils import extract_answer
from reasoning_gym import create_dataset


from functools import lru_cache
@lru_cache(maxsize=None)
def _get_dataset_for_scoring(dataset_name: str):
    return create_dataset(dataset_name, size=1, seed=1)

def _score_answer(dataset_name: str, answer: str, entry: Any) -> float:
    ds = _get_dataset_for_scoring(dataset_name)
    return float(ds.score_answer(answer, entry=entry))


class ReasoningGymEnv(BaseTextEnv):
    """
    Environment for ReasoningGym tasks.
    Handles reward calculation using ReasoningGym's scoring methods.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "extra_info" in extras, "extra_info field is required"
        self.extra_info = extras["extra_info"]
        
        # Prefer dataset_name-based shared scorer; fall back to serialized data source if not provided
        assert "data_source" in extras, "data_source field is required"
        self.dataset_name = extras["data_source"].split("/")[-1]
        print(f"Dataset name: {self.dataset_name}")
        # if self.dataset_name is None:
        #     # Deserialize the data source from the stored serialized version
        #     data_source_serialized = self.extra_info.get("data_source_serialized")
        #     _pickle_start_time = time.perf_counter()
        #     data_source_bytes = base64.b64decode(data_source_serialized)
        #     self.reasoning_gym_data_source = pickle.loads(data_source_bytes)
        #     _pickle_elapsed_s = time.perf_counter() - _pickle_start_time
        #     print(f"Deserialize and pickle load took {_pickle_elapsed_s:.6f} seconds")
        # else:
        #     self.reasoning_gym_data_source = None

        try:
            self.original_entry = json.loads(self.extra_info["dataset_entry"])
        except (json.JSONDecodeError, TypeError):
            self.original_entry = self.extra_info["dataset_entry"]


    def _get_reward(self, action: str) -> float:
        """
        Calculate reward using ReasoningGym's built-in scoring logic.
        """
        
        try:
            found_answer = extract_answer(action, tag_name="answer")
        except Exception as e:
            print(f"Warning: Error extracting answer between <answer></answer> tags from model output, scoring the entire model output: {e}")
            found_answer = action

        # if self.dataset_name is not None:
        reward = _score_answer(self.dataset_name, found_answer, self.original_entry)
        # else:
            # reward = self.reasoning_gym_data_source.score_answer(found_answer, entry=self.original_entry)
        return float(reward)

        
    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process one step in the reasoning environment.
        For reasoning tasks, we typically complete in one step.
        """
        done = True  # Most reasoning tasks complete in one step
        reward = self._get_reward(action)

        # No additional observations needed for reasoning tasks
        return BaseTextEnvStepOutput(
            observations=[], 
            reward=reward, 
            done=done, 
            metadata={}
        )
