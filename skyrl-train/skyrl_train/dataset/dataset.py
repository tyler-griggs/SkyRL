import datasets
from loguru import logger
import os
from typing import List
from transformers import PreTrainedTokenizerBase
from pathlib import Path


class PromptDataset:
    def __init__(
        self,
        datasets: str | List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers

        self.datasets = datasets
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        loaded_datasets = []
        for source in self.datasets:
            ext = os.path.splitext(source)[-1].lower()
            if ext == ".parquet":
                ds = datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
            elif ext in [".json", ".jsonl"]:
                ds = datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
            else:
                # Treat as HF dataset spec: "name" or "name:split"
                dataset_name, has_split, split = source.partition(":")
                try:
                    ds_dict = datasets.load_dataset(path=dataset_name, keep_in_memory=True)
                except ValueError:
                    raise ValueError(f"Dataset `{dataset_name}` not found on Hugging Face.")
                split = split if has_split else "train"
                if split not in ds_dict:
                    raise ValueError(
                        f"Split `{split}` not found in dataset `{dataset_name}`. Configured split was `{split}` and default is `train`"
                    )
                ds = ds_dict[split]
            loaded_datasets.append(ds)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(loaded_datasets)

        logger.info(f"Total dataset size: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
            <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"Filtered dataset size: {len(self.dataframe)}")

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)

        extra = {key: value for key, value in row_dict.items() if key != self.prompt_key and key != self.env_class_key}

        return messages, env_class, extra

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, env_class, env_extras in item_list:
            all_inputs.append({"prompt": prompt, "env_class": env_class, "env_extras": env_extras})
        return all_inputs

    def __len__(self):
        return len(self.dataframe)


class EnvironmentDataset:
    """
    A dataset that loads environment data from direct file/directory paths.
    Each dataset item is a path to a task directory.
    """

    def __init__(
        self,
        data_files: List[str],
        **kwargs,
    ):
        """
        Initialize the EnvironmentDataset.

        Args:
            data_files: List of direct file/directory paths pointing to environment data
        """
        self.data_files = data_files

        # Load all data files
        self.task_paths = self._load_data_files()

        logger.info(f"EnvironmentDataset initialized with {len(self.task_paths)} task paths")

    def _load_data_files(self) -> List[Path]:
        """Load all data files from direct paths and return list of task paths."""
        task_paths = []

        for data_source in self.data_files:
            source_path = Path(data_source)

            if not source_path.exists():
                logger.warning(f"Path does not exist: {data_source}")
                continue

            logger.info(f"Loading data from: {data_source}")

            # If the path is a directory, find all valid task subdirectories
            if source_path.is_dir():
                # Look for task subdirectories and validate them
                all_dirs = [d for d in source_path.iterdir() if d.is_dir()]
                valid_task_dirs = [d for d in all_dirs if self._is_valid_task_directory(d)]

                if valid_task_dirs:
                    task_paths.extend(valid_task_dirs)
                    logger.info(
                        f"Found {len(valid_task_dirs)} valid task directories out of {len(all_dirs)} total directories"
                    )
                elif self._is_valid_task_directory(source_path):
                    # If no subdirectories but the main directory is valid, treat it as a task
                    task_paths.append(source_path)
                    logger.info("Using main directory as valid task")
                else:
                    logger.warning(f"No valid task directories found in {source_path}")
            else:
                # If it's a file, treat it as a single task (files can't be valid task directories)
                logger.warning(f"File {source_path} cannot be a valid task directory (missing instruction.md)")

        return task_paths

    def _is_valid_task_directory(self, task_path: Path) -> bool:
        """Check if a directory is a valid task directory (has instruction.md file)."""
        if not task_path.is_dir():
            return False

        instruction_file = task_path / "instruction.md"
        return instruction_file.exists() and instruction_file.is_file()

    def __getitem__(self, index: int) -> dict:
        """Get a task path by index as a dictionary with 'prompt', 'env_class', and 'env_extras' keys."""
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_paths)}")
        return {
            "prompt": str(self.task_paths[index]),
            "env_class": None,
            "env_extras": {"data_source": str(self.task_paths[index])},
        }

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.task_paths)

    def __iter__(self):
        """Iterate over all task paths as dictionaries."""
        for task_path in self.task_paths:
            yield {"prompt": str(task_path), "env_class": None, "env_extras": {"data_source": str(task_path)}}

    def get_task_paths(self) -> List[Path]:
        """Return all task paths as a list."""
        return self.task_paths.copy()

    def collate_fn(self, item_list):
        """Collate function for batching task dictionaries."""
        return item_list
