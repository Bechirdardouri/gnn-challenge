from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition.submission_utils import normalize_test_nodes, read_label_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize private test labels from secret content, legacy encrypted file, or external source."
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "private" / "test_labels.csv"),
        help="Output CSV path for materialized labels.",
    )
    parser.add_argument(
        "--test-nodes-path",
        default=str(ROOT / "data" / "test.csv"),
        help="Public test nodes CSV used to validate ID coverage.",
    )
    parser.add_argument(
        "--encrypted-labels-path",
        default=str(ROOT / "data" / "test_labels.csv.enc"),
        help="Legacy encrypted labels file path.",
    )

    parser.add_argument(
        "--private-test-labels-csv",
        default=os.getenv("PRIVATE_TEST_LABELS_CSV", ""),
        help="Raw private labels CSV content. Defaults to PRIVATE_TEST_LABELS_CSV env var.",
    )
    parser.add_argument(
        "--test-labels-key",
        default=os.getenv("TEST_LABELS_KEY", ""),
        help="Legacy passphrase for encrypted labels file. Defaults to TEST_LABELS_KEY env var.",
    )

    parser.add_argument(
        "--private-data-method",
        default=os.getenv("PRIVATE_DATA_METHOD", "google_drive"),
        choices=["google_drive", "url", "s3"],
        help="External source method when direct/legacy methods are unavailable.",
    )
    parser.add_argument(
        "--google-drive-file-id",
        default=os.getenv("GOOGLE_DRIVE_FILE_ID", ""),
        help="Google Drive file id (for google_drive method).",
    )
    parser.add_argument(
        "--google-credentials-json",
        default=os.getenv("GOOGLE_CREDENTIALS_JSON", ""),
        help="Service account JSON content or file path.",
    )
    parser.add_argument(
        "--private-data-url",
        default=os.getenv("PRIVATE_DATA_URL", ""),
        help="Direct URL to private labels (for url method).",
    )
    parser.add_argument(
        "--private-data-s3-uri",
        default=os.getenv("PRIVATE_DATA_S3_URI", ""),
        help="S3 URI (for s3 method).",
    )
    return parser.parse_args()


def validate_labels(labels_path: Path, test_nodes_path: Path) -> None:
    labels = read_label_csv(labels_path)
    if labels.empty:
        raise ValueError("Private labels file is empty.")
    if labels["id"].duplicated().any():
        raise ValueError("Private labels contain duplicate IDs.")
    if labels["true"].isna().any():
        raise ValueError("Private labels contain NaN targets.")

    if test_nodes_path.exists():
        expected = normalize_test_nodes(pd.read_csv(test_nodes_path))
        label_ids = set(labels["id"])
        expected_ids = set(expected["id"])
        if label_ids != expected_ids:
            missing = len(expected_ids - label_ids)
            extra = len(label_ids - expected_ids)
            raise ValueError(
                f"Private labels do not match public test IDs (missing={missing}, extra={extra})."
            )


def try_from_inline_secret(raw_csv: str, output_path: Path) -> bool:
    if not raw_csv.strip():
        return False
    output_path.write_text(raw_csv, encoding="utf-8")
    print("Materialized labels from PRIVATE_TEST_LABELS_CSV.")
    return True


def try_from_legacy_encrypted_file(enc_path: Path, key: str, output_path: Path) -> bool:
    if not key.strip():
        return False
    if not enc_path.exists():
        return False

    command = [
        "openssl",
        "enc",
        "-d",
        "-aes-256-cbc",
        "-pbkdf2",
        "-in",
        str(enc_path),
        "-out",
        str(output_path),
        "-pass",
        "env:LEGACY_TEST_LABELS_KEY",
    ]
    env = os.environ.copy()
    env["LEGACY_TEST_LABELS_KEY"] = key
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if result.returncode == 0:
        print(f"Materialized labels by decrypting legacy file: {enc_path}")
        return True

    stderr = (result.stderr or "").strip().splitlines()
    hint = stderr[-1] if stderr else "unknown openssl error"
    print(f"Legacy decrypt failed: {hint}")
    output_path.unlink(missing_ok=True)
    return False


def try_from_external_source(
    output_path: Path,
    method: str,
    google_drive_file_id: str,
    google_credentials_json: str,
    private_data_url: str,
    private_data_s3_uri: str,
) -> bool:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "download_private_data.py"),
        "--method",
        method,
        "--output",
        str(output_path),
        "--google-drive-file-id",
        google_drive_file_id,
        "--google-credentials-json",
        google_credentials_json,
        "--url",
        private_data_url,
        "--s3-uri",
        private_data_s3_uri,
        "--force",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print(f"Materialized labels via external source method={method}.")
        return True

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    msg = stderr or stdout or "unknown download error"
    print(f"External private-data download failed (method={method}): {msg}")
    output_path.unlink(missing_ok=True)
    return False


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    test_nodes_path = Path(args.test_nodes_path)
    encrypted_labels_path = Path(args.encrypted_labels_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    source = ""

    if try_from_inline_secret(args.private_test_labels_csv, output_path):
        source = "inline_secret"
    else:
        used_legacy = try_from_legacy_encrypted_file(
            encrypted_labels_path, args.test_labels_key, output_path
        )
        if used_legacy:
            source = "legacy_encrypted_file"
        else:
            used_external = try_from_external_source(
                output_path=output_path,
                method=args.private_data_method,
                google_drive_file_id=args.google_drive_file_id,
                google_credentials_json=args.google_credentials_json,
                private_data_url=args.private_data_url,
                private_data_s3_uri=args.private_data_s3_uri,
            )
            if used_external:
                source = f"external:{args.private_data_method}"

    if not source:
        msg = (
            "Unable to materialize private labels. Configure one of: "
            "PRIVATE_TEST_LABELS_CSV, TEST_LABELS_KEY (+ data/test_labels.csv.enc), "
            "or PRIVATE_DATA_METHOD with its required source secrets."
        )
        print(msg)
        print(f"::error::{msg}")
        return 1

    try:
        validate_labels(output_path, test_nodes_path)
    except Exception as exc:  # noqa: BLE001
        output_path.unlink(missing_ok=True)
        msg = f"Private labels validation failed: {exc}"
        print(msg)
        print(f"::error::{msg}")
        return 1

    print(f"Private labels ready at: {output_path} (source={source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
