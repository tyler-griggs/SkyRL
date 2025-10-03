import sys
from typing import Final

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


DEFAULT_BUCKET_NAME: Final[str] = "skyrl-test"
DEFAULT_OBJECT_KEY: Final[str] = "tyler-test2.txt"
DEFAULT_CONTENT: Final[str] = "This is a test upload from s3_upload.py.\n"


def upload_test_file(
    bucket_name: str = DEFAULT_BUCKET_NAME,
    object_key: str = DEFAULT_OBJECT_KEY,
    content: str = DEFAULT_CONTENT,
) -> None:
    """Upload a simple text file to the specified S3 bucket.

    Requires AWS credentials to be configured in the environment or config files.
    """
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=content.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )
    print(f"Uploaded to s3://{bucket_name}/{object_key}")


def main() -> int:
    try:
        upload_test_file()
        return 0
    except NoCredentialsError:
        print("AWS credentials not found. Configure them (e.g., via environment variables or AWS config) and rerun.")
        return 1
    except (BotoCoreError, ClientError) as error:
        print(f"Failed to upload to S3: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
