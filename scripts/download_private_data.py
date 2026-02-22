from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download private test labels using Google Drive, URL, or S3."
    )
    parser.add_argument(
        "--method",
        default=os.getenv("PRIVATE_DATA_METHOD", "google_drive"),
        choices=["google_drive", "url", "s3"],
        help="Data source method.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "private" / "test_labels.csv"),
        help="Destination path.",
    )
    parser.add_argument(
        "--google-drive-file-id",
        default=os.getenv("GOOGLE_DRIVE_FILE_ID", ""),
        help="Google Drive file ID for private data.",
    )
    parser.add_argument(
        "--google-credentials-json",
        default=os.getenv("GOOGLE_CREDENTIALS_JSON", ""),
        help="Service account JSON content or file path.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("PRIVATE_DATA_URL", ""),
        help="Direct URL to private data when method=url.",
    )
    parser.add_argument(
        "--s3-uri",
        default=os.getenv("PRIVATE_DATA_S3_URI", ""),
        help="S3 URI (s3://bucket/key) when method=s3.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists.")
    return parser.parse_args()


def ensure_output(output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists. Use --force to overwrite.")


def load_google_credentials(raw: str) -> dict:
    if not raw:
        raise ValueError("GOOGLE_CREDENTIALS_JSON is required for google_drive method.")
    try:
        candidate_path = Path(raw)
        if candidate_path.exists():
            return json.loads(candidate_path.read_text(encoding="utf-8"))
    except OSError:
        pass
    return json.loads(raw)


def download_from_google_drive(file_id: str, credentials_raw: str) -> bytes:
    if not file_id:
        raise ValueError("GOOGLE_DRIVE_FILE_ID is required for google_drive method.")
    creds_info = load_google_credentials(credentials_raw)

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError(
            "Google API dependencies missing. Install requirements.txt before running this script."
        ) from exc

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_info, scopes=scopes
    )
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    request = service.files().get_media(fileId=file_id)
    output = io.BytesIO()
    downloader = MediaIoBaseDownload(output, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return output.getvalue()


def download_from_url(url: str) -> bytes:
    if not url:
        raise ValueError("PRIVATE_DATA_URL is required for url method.")
    req = urllib.request.Request(url, headers={"User-Agent": "gnn-competition-bot"})
    with urllib.request.urlopen(req) as response:
        return response.read()


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3 URI must start with s3://")
    parsed = urllib.parse.urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError("Invalid s3 URI; expected s3://bucket/key")
    return bucket, key


def download_from_s3(s3_uri: str) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for s3 method. Install requirements.txt before running this script."
        ) from exc

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    ensure_output(output, args.force)

    if args.method == "google_drive":
        data = download_from_google_drive(args.google_drive_file_id, args.google_credentials_json)
    elif args.method == "url":
        data = download_from_url(args.url)
    else:
        data = download_from_s3(args.s3_uri)

    output.write_bytes(data)
    print(f"Private data downloaded to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
