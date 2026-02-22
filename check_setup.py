#!/usr/bin/env python3
"""Quick environment check for the challenge repository."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_PACKAGES = ["numpy", "pandas", "sklearn", "scipy", "cryptography"]
OPTIONAL_PACKAGES = ["torch", "torch_geometric", "googleapiclient", "boto3"]

REQUIRED_FILES = [
    "data/train.csv",
    "data/val.csv",
    "data/test.csv",
    "data/edges.csv",
    "encryption/public_key.pem",
]


def check_python_version() -> bool:
    version = sys.version_info
    ok = version.major > 3 or (version.major == 3 and version.minor >= 10)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
    return ok


def check_import(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError:
        print(f"[MISSING] {package_name}")
        return False


def check_files() -> bool:
    ok = True
    for rel_path in REQUIRED_FILES:
        exists = Path(rel_path).exists()
        status = "OK" if exists else "MISSING"
        print(f"[{status}] {rel_path}")
        ok = ok and exists
    return ok


def main() -> int:
    print("Challenge Environment Check\n")

    py_ok = check_python_version()
    print("\nRequired packages:")
    req_ok = all(check_import(pkg) for pkg in REQUIRED_PACKAGES)

    print("\nOptional packages (needed only for specific workflows/models):")
    _ = [check_import(pkg) for pkg in OPTIONAL_PACKAGES]

    print("\nRequired files:")
    files_ok = check_files()

    print("\nSuggested next steps:")
    print("1. Train SOTA baseline: python starter_code/sota_graph_ensemble.py")
    print("2. Encrypt submission: python encryption/encrypt.py <csv> encryption/public_key.pem submissions/<team>.enc")
    print("3. Submit encrypted file via Pull Request")

    if py_ok and req_ok and files_ok:
        print("\nEnvironment check passed.")
        return 0

    print("\nEnvironment check failed. Resolve missing items above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
