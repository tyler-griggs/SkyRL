from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
from omegaconf import DictConfig
import json
from reasoning_gym.utils import extract_answer
from reasoning_gym import create_dataset


class ReasoningGymEnv(BaseTextEnv):
    """
    Environment for ReasoningGym tasks.
    Handles reward calculation using ReasoningGym's scoring methods.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "extra_info" in extras, "extra_info field is required"
        self.extra_info = extras["extra_info"]
        self.data_source = extras["data_source"]
        
        dataset_name = self.extra_info.get("dataset_name")
        size = self.extra_info.get("size")
        seed = self.extra_info.get("seed")
        
        self.reasoning_gym_data_source = create_dataset(dataset_name, size=size, seed=seed)

        try:
            self.original_entry = json.loads(self.extra_info["dataset_entry"])
        except (json.JSONDecodeError, TypeError):
            self.original_entry = self.extra_info["dataset_entry"]


    def _get_reward(self, action: str) -> float:
        """
        Calculate reward using ReasoningGym's built-in scoring logic.
        """
        found_answer = extract_answer(action, tag_name="answer")

        if self.reasoning_gym_data_source and hasattr(self.reasoning_gym_data_source, 'score_answer'):
            try:
                reward = self.reasoning_gym_data_source.score_answer(found_answer, entry=self.original_entry)
                return float(reward)
            except Exception as e:
                print("Warning: Error scoring answer, returning 0.0", e)
                return 0.0
        
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
