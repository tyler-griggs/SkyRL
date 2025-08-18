#!/usr/bin/env python3
"""
Test script using real ReasoningGym datasets.

This script demonstrates how to use the ReasoningGymDataset class
with actual ReasoningGym procedurally generated data.
"""
from dataset import ReasoningGymDataset, make_reasoning_gym_dataset
import reasoning_gym
print("Successfully imported ReasoningGymDataset")

def test_procedural_dataset():
    dataset = reasoning_gym.create_dataset('leg_counting', size=10, seed=42)
    reasoning_gym_dataset = ReasoningGymDataset(
        procedural_dataset=dataset,
        developer_prompt="Test prompt"
    )
    # print("data source: ", reasoning_gym_dataset.__getitem__(0)["data_source"])
    # print("prompt: ", reasoning_gym_dataset.__getitem__(0)["prompt"])
    # print("env_class: ", reasoning_gym_dataset.__getitem__(0)["env_class"])
    # print("reward_spec: ", reasoning_gym_dataset.__getitem__(0)["reward_spec"])
    # print("extra_info: ", reasoning_gym_dataset.__getitem__[0]["extra_info"])

    partial_output_1 = "The total number of legs is 100. Final answer: <answer>100</answer>"

    print("score_generation: ", reasoning_gym_dataset.score_generation(0, partial_output_1))




if __name__ == "__main__":
    test_procedural_dataset()