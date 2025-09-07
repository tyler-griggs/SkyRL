import datasets
from loguru import logger
import os
import boto3
import hashlib
from pathlib import Path
from typing import List, Union
import tempfile
import requests
import zipfile
import tarfile
from urllib.parse import urlparse
import shutil


class PromptDataset:
    def __init__(
        self,
        data_files,
        tokenizer: callable,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.data_files = data_files
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            ext = os.path.splitext(data_file)[-1].lower()
            if ext == ".parquet":
                dataset = datasets.load_dataset("parquet", data_files=data_file, keep_in_memory=True)["train"]
            elif ext == ".json":
                dataset = datasets.load_dataset("json", data_files=data_file, keep_in_memory=True)["train"]
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            dataframes.append(dataset)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        logger.info(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
            <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"filter dataset len: {len(self.dataframe)}")

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
    A dataset that downloads environment data from S3 URLs, GitHub repositories, or Hugging Face datasets and caches them locally.
    Each dataset item is a path to a task directory.
    """
    
    def __init__(
        self,
        data_files: List[str],
        cache_dir: Union[str, Path] = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
        region_name: str = "us-east-1",
        github_token: str = None,
        hf_token: str = None,
        **kwargs,
    ):
        """
        Initialize the EnvironmentDataset.
        
        Args:
            data_files: List of S3 URLs, GitHub URLs, or Hugging Face dataset names pointing to environment data
            cache_dir: Directory to cache downloaded files. If None, uses temp directory
            aws_access_key_id: AWS access key ID for S3 access (overrides env vars)
            aws_secret_access_key: AWS secret access key for S3 access (overrides env vars)
            aws_session_token: AWS session token (optional, overrides env vars)
            region_name: AWS region name
            github_token: GitHub token for private repositories (overrides env vars)
            hf_token: Hugging Face token for private datasets (overrides env vars)
        """
        self.data_files = data_files
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "skyRL_env_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get AWS credentials from environment variables if not provided
        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        # Get GitHub token from environment variable if not provided
        github_token = github_token or os.getenv("GITHUB_TOKEN")
        
        # Get Hugging Face token from environment variable if not provided
        hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        # Initialize S3 client (only if we have S3 URLs)
        self.s3_client = None
        if any(url.startswith("s3://") for url in data_files):
            s3_kwargs = {"region_name": region_name}
            if aws_access_key_id:
                s3_kwargs["aws_access_key_id"] = aws_access_key_id
            if aws_secret_access_key:
                s3_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                s3_kwargs["aws_session_token"] = aws_session_token
                
            self.s3_client = boto3.client("s3", **s3_kwargs)
        
        # Store tokens for private repositories/datasets
        self.github_token = github_token
        self.hf_token = hf_token
        
        # Download and cache all data files
        self.task_paths = self._download_and_cache_files()
        
        logger.info(f"EnvironmentDataset initialized with {len(self.task_paths)} task paths")
    
    def _get_cache_path(self, url: str) -> Path:
        """Generate a cache path for a URL based on its hash."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}"
    
    def _is_github_url(self, url: str) -> bool:
        """Check if URL is a GitHub URL."""
        return "github.com" in url or "githubusercontent.com" in url
    
    def _is_s3_url(self, url: str) -> bool:
        """Check if URL is an S3 URL."""
        return url.startswith("s3://")
    
    def _is_huggingface_dataset(self, dataset_name: str) -> bool:
        """Check if string is a Hugging Face dataset name."""
        # Hugging Face dataset names typically don't contain URLs or special characters
        # and are usually in format "username/dataset-name" or just "dataset-name"
        return (
            not dataset_name.startswith(("http://", "https://", "s3://")) and
            not "/" in dataset_name or 
            (dataset_name.count("/") == 1 and not dataset_name.startswith("/") and not dataset_name.endswith("/"))
        )
    
    def _download_s3_file(self, s3_url: str, local_path: Path) -> None:
        """Download a file from S3 to local path."""
        try:
            # Parse S3 URL
            if not s3_url.startswith("s3://"):
                raise ValueError(f"Invalid S3 URL: {s3_url}")
            
            # Remove s3:// prefix and split bucket/key
            s3_path = s3_url[5:]  # Remove "s3://"
            if "/" not in s3_path:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            
            bucket_name, s3_key = s3_path.split("/", 1)
            
            # Download file
            logger.info(f"Downloading {s3_url} to {local_path}")
            self.s3_client.download_file(bucket_name, s3_key, str(local_path))
            logger.info(f"Successfully downloaded {s3_url}")
            
        except Exception as e:
            logger.error(f"Failed to download {s3_url}: {e}")
            raise
    
    def _download_github_file(self, github_url: str, local_path: Path) -> None:
        """Download a file or repository from GitHub to local path."""
        try:
            # Prepare headers for GitHub API
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            # Handle different GitHub URL formats
            if github_url.endswith("/archive/refs/heads/main.zip") or github_url.endswith("/archive/refs/heads/master.zip"):
                # Direct zip download
                download_url = github_url
            elif "/releases/download/" in github_url:
                # Release asset download
                download_url = github_url
            elif "github.com" in github_url and "/archive/" not in github_url:
                # Convert to zip download URL
                if github_url.endswith("/"):
                    github_url = github_url[:-1]
                download_url = f"{github_url}/archive/refs/heads/main.zip"
            else:
                download_url = github_url
            
            logger.info(f"Downloading {download_url} to {local_path}")
            
            # Download the file
            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Save to temporary file first
            temp_path = local_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract if it's an archive
            if download_url.endswith((".zip", ".tar.gz", ".tar.bz2")):
                self._extract_archive(temp_path, local_path)
                temp_path.unlink()  # Remove temp file
            else:
                # Move temp file to final location
                temp_path.rename(local_path)
            
            logger.info(f"Successfully downloaded {github_url}")
            
        except Exception as e:
            logger.error(f"Failed to download {github_url}: {e}")
            raise
    
    def _extract_archive(self, archive_path: Path, extract_to: Path) -> None:
        """Extract archive file to directory."""
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffixes[-2:] == [".tar", ".bz2"]:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        
        # Move contents from subdirectory to main directory if needed
        extracted_dirs = [d for d in extract_to.iterdir() if d.is_dir()]
        if len(extracted_dirs) == 1:
            # If there's only one subdirectory, move its contents up
            subdir = extracted_dirs[0]
            for item in subdir.iterdir():
                item.rename(extract_to / item.name)
            subdir.rmdir()
    
    def _download_huggingface_dataset(self, dataset_name: str, local_path: Path) -> None:
        """Download a Hugging Face dataset directly from the Hub and extract task directories."""
        try:
            logger.info(f"Downloading Hugging Face dataset: {dataset_name}")
            
            # Set up authentication if token is provided
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
            
            # Create the local directory
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Use huggingface_hub to download the dataset repository
            from huggingface_hub import snapshot_download
            
            # Download the entire dataset repository
            repo_path = snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=str(local_path),
                token=self.hf_token
            )
            
            logger.info(f"Successfully downloaded Hugging Face dataset: {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to download Hugging Face dataset {dataset_name}: {e}")
            raise
    
    
    def _download_and_cache_files(self) -> List[Path]:
        """Download all files from S3, GitHub, or Hugging Face and return list of local paths."""
        task_paths = []
        
        for data_source in self.data_files:
            cache_path = self._get_cache_path(data_source)
            
            # Check if file already exists in cache
            if cache_path.exists():
                logger.info(f"Using cached data: {cache_path}")
            else:
                # Download data based on source type
                if self._is_s3_url(data_source):
                    if self.s3_client is None:
                        raise ValueError("S3 client not initialized. Check AWS credentials.")
                    self._download_s3_file(data_source, cache_path)
                elif self._is_github_url(data_source):
                    self._download_github_file(data_source, cache_path)
                elif self._is_huggingface_dataset(data_source):
                    self._download_huggingface_dataset(data_source, cache_path)
                else:
                    raise ValueError(f"Unsupported data source: {data_source}. Only S3 URLs, GitHub URLs, and Hugging Face dataset names are supported.")
            
            # If the downloaded data is a directory, find all task subdirectories
            if cache_path.is_dir():
                # Look for task subdirectories
                task_dirs = [d for d in cache_path.iterdir() if d.is_dir()]
                if task_dirs:
                    task_paths.extend(task_dirs)
                else:
                    # If no subdirectories, treat the directory itself as a task
                    task_paths.append(cache_path)
            else:
                # If it's a file, treat it as a single task
                task_paths.append(cache_path)
        
        return task_paths
    
    def __getitem__(self, index: int) -> dict:
        """Get a task path by index as a dictionary with 'prompt', 'env_class', and 'env_extras' keys."""
        if index >= len(self.task_paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.task_paths)}")
        return {"prompt": str(self.task_paths[index]), "env_class": None, "env_extras": None}
    
    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        return len(self.task_paths)
    
    def __iter__(self):
        """Iterate over all task paths as dictionaries."""
        for task_path in self.task_paths:
            yield {"prompt": str(task_path), "env_class": None, "env_extras": None}
    
    def get_task_paths(self) -> List[Path]:
        """Return all task paths as a list."""
        return self.task_paths.copy()
    
    def clear_cache(self) -> None:
        """Clear the cache directory."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared cache directory: {self.cache_dir}")
    
    def collate_fn(self, item_list):
        """Collate function for batching task dictionaries."""
        return item_list
