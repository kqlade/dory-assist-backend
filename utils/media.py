import os
import tempfile
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError

from config import TELNYX_MMS_S3_BUCKET


def download_file(url: str) -> str:
    """Download the file from a URL into a temporary location and return the local path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Error downloading file from {url}: {e}")

    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path) or "downloaded_media"
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, file_name)

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return local_path, response.headers.get("Content-Type", "application/octet-stream")


def upload_to_s3(local_path: str, content_type: str) -> str:
    """Upload the file at local_path to S3 and return the public URL."""
    if not TELNYX_MMS_S3_BUCKET:
        raise RuntimeError("TELNYX_MMS_S3_BUCKET environment variable is not set")

    file_name = os.path.basename(local_path)
    s3_client = boto3.client("s3")
    try:
        extra_args = {"ContentType": content_type, "ACL": "public-read"}
        s3_client.upload_file(local_path, TELNYX_MMS_S3_BUCKET, file_name, ExtraArgs=extra_args)
    except ClientError as e:
        raise RuntimeError(f"Error uploading file to S3: {e}")

    return f"https://{TELNYX_MMS_S3_BUCKET}.s3.amazonaws.com/{file_name}"


def media_downloader_uploader(url: str) -> str:
    """Download a media file and re-upload it to S3, returning its new URL."""
    local_path, content_type = download_file(url)
    try:
        return upload_to_s3(local_path, content_type)
    finally:
        # Always remove the local file, even if upload fails
        try:
            os.remove(local_path)
        except Exception:
            pass
