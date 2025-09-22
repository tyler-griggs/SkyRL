# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import re
import os
import random

import datasets


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/aime")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "AI-MO/aimo-validation-aime"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    
    instruction_following = 'Please reason step by step, and put your final answer within \\boxed{}.'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": example.pop("problem") + " " + instruction_following,
                    }
                ],
                "env_class": "aime",
                "reward_model": {
                    "method": "rule",
                    "ground_truth": str(example.pop("answer")),
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_split = list(range(len(train_dataset)))
    random.Random(0).shuffle(train_split)
    train_dataset = train_dataset.select(train_split)
    first_half_count = 45
    second_half_count = 45
    num_examples = len(train_dataset)
    first_half = train_dataset.select(range(first_half_count))
    second_half = train_dataset.select(range(num_examples - second_half_count, num_examples))
    repeated_times = 23
    train_repeated = datasets.concatenate_datasets([first_half] * repeated_times)
    # val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Training dataset size: {len(train_repeated)}")
    print(f"Validation dataset size: {len(second_half)}")
    train_repeated.to_parquet(os.path.join(output_dir, "train.parquet"))
    second_half.to_parquet(os.path.join(output_dir, "validation.parquet"))
    # val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))
