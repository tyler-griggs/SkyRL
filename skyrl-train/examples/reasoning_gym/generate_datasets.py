import os
import argparse
import random
from skyrl_gym.envs.reasoning_gym.dataset import ReasoningGymDataset
from reasoning_gym import create_dataset

def main(args):
    dataset_name = args.dataset_name
    size = args.size
    developer_prompt = args.developer_prompt
    output_dir = os.path.expanduser(args.file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    seed = random.randint(0, 1000000)
    ds = create_dataset(dataset_name, size=size, seed=seed)
    
    rg = ReasoningGymDataset(
        procedural_dataset=ds,
        developer_prompt=developer_prompt,
        dataset_name=dataset_name,
        size=size,
        seed=seed
    )
    
    rg.write_to_parquet(os.path.join(output_dir, "train.parquet"))
    print(f"Dataset with {len(ds)} examples saved to {output_dir}/train.parquet")

    seed = random.randint(0, 1000000)
    ds = create_dataset(dataset_name, size=size, seed=seed)
    
    rg = ReasoningGymDataset(
        procedural_dataset=ds,
        developer_prompt=developer_prompt,
        dataset_name=dataset_name,
        size=size,
        seed=seed
    )
    
    rg.write_to_parquet(os.path.join(output_dir, "validation.parquet"))
    print(f"Dataset with {len(ds)} examples saved to {output_dir}/validation.parquet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="leg_counting")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--developer_prompt", type=str, default="Think step by step. Return ONLY the final numeric answer wrapped in <answer> tags, like <answer>42</answer>.")
    parser.add_argument("--file_path", type=str, default="~/data/reasoning_gym")
    
    args = parser.parse_args()
    main(args)