# s3_test.py
import os
import sys
import uuid
import urllib.parse
import boto3
from botocore.exceptions import ClientError


def parse_s3_uri(uri: str):
    p = urllib.parse.urlparse(uri)
    assert p.scheme == "s3", f"Not an S3 URI: {uri}"
    return p.netloc, p.path.lstrip("/")


def main():
    if len(sys.argv) < 2 and "S3_TEST_URI" not in os.environ:
        print("Usage: python s3_test.py s3://your-bucket/path/test.txt  (or set S3_TEST_URI)")
        sys.exit(2)

    s3_uri = sys.argv[1] if len(sys.argv) > 1 else os.environ["S3_TEST_URI"]
    bucket, key = parse_s3_uri(s3_uri)

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),  # harmless if None
        region_name=region,
    )

    # 1) Who am I?
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    print(f"Caller: {ident['Arn']} (Acct {ident['Account']}) in {region}")

    s3 = session.client("s3", region_name=region)

    data = b"hello s3\n" + uuid.uuid4().hex.encode()

    # 2) Write
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data)
        print(f"Wrote {len(data)} bytes to s3://{bucket}/{key}")
    except ClientError as e:
        print("PUT failed:", e.response.get("Error", {}))
        print(
            "Hints: ensure IAM allows s3:PutObject and bucket policy doesn't Deny; "
            "if bucket uses SSE-KMS, grant kms:Encrypt/GenerateDataKey."
        )
        sys.exit(1)

    # 3) Read
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        rb = obj["Body"].read()
        print("Readback OK" if rb == data else "Readback mismatch")
    except ClientError as e:
        print("GET failed:", e.response.get("Error", {}))
        print("Hints: need s3:GetObject; if cross-account, bucket policy must allow your principal.")
        sys.exit(1)


if __name__ == "__main__":
    main()
