import argparse
import os
from typing import Any, Dict

from verifiers import load_environment


def build_row(example: Dict[str, Any], idx: int, split: str, data_source: str, env_id: str) -> Dict[str, Any]:
    question = example.get("question", "")
    answer = example.get("answer", "")
    prompt = example.get("prompt")  # already formatted by the env as chat messages

    return {
        "data_source": data_source,
        "prompt": prompt,
        "env_class": env_id,
        "reward_spec": {"method": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "answer": answer,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Parquet dataset from a verifiers environment.")
    parser.add_argument("--env_id", default="wordle", help="Environment identifier to load (e.g., 'wordle').")
    # parser.add_argument("--num_train_examples", type=int, default=2000, help="Number of training examples to generate.")
    # parser.add_argument("--num_eval_examples", type=int, default=20, help="Number of evaluation examples to generate.")
    parser.add_argument("--output_dir", default="~/data/verifiers/wordle", help="Output directory for Parquet files.")

    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load verifiers environment
    vf_env = load_environment(
        env_id=args.env_id,
        # num_train_examples=args.num_train_examples,
        # num_eval_examples=args.num_eval_examples,
    )

    # Get HF datasets from environment (already includes 'prompt' and retains 'answer'/'question')
    train_ds = vf_env.get_dataset()
    eval_ds = vf_env.get_eval_dataset()

    data_source = f"verifiers/{args.env_id}"

    # Map to the standardized schema mirroring gsm8k formatting
    train_ds = train_ds.map(
        lambda ex, idx: build_row(ex, idx, split="train", data_source=data_source, env_id=args.env_id),
        with_indices=True,
    )
    eval_ds = eval_ds.map(
        lambda ex, idx: build_row(ex, idx, split="test", data_source=data_source, env_id=args.env_id),
        with_indices=True,
    )

    # Save to Parquet
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")
    train_ds.to_parquet(train_path)
    eval_ds.to_parquet(val_path)