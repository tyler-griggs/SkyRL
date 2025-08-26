#!/usr/bin/env python3
import argparse
import os
import sys


# Ensure the package root is on sys.path when running from the scripts directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(scripts_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from skyrl_train.utils.io import open_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Super simple S3 write/read test using skyrl_train.utils.io.open_file",
    )
    parser.add_argument(
        "--s3_uri",
        nargs="?",
        help="S3 URI to write to (e.g., s3://my-bucket/test/test_io.txt). If omitted, uses $S3_TEST_URI.",
    )
    parser.add_argument(
        "--content",
        default="hello s3\n",
        help="Text content to write (default: 'hello s3\\n')",
    )
    args = parser.parse_args()

    s3_uri = args.s3_uri or os.environ.get("S3_TEST_URI")
    if not s3_uri:
        print("Error: Provide an S3 URI as a positional argument or set the S3_TEST_URI environment variable.")
        sys.exit(1)

    data = args.content.encode("utf-8")

    print(f"Writing {len(data)} bytes to {s3_uri} ...")
    with open_file(s3_uri, "wb") as f:
        f.write(data)
    print("Write complete. Verifying readback ...")

    with open_file(s3_uri, "rb") as f:
        read_back = f.read()

    if read_back == data:
        print("Success: read content matches what was written.")
        sys.exit(0)
    else:
        print("Failure: read content does not match what was written.")
        sys.exit(2)


if __name__ == "__main__":
    main() 