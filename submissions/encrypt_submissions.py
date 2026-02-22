from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encryption.crypto import encrypt_bytes, load_public_key, write_encrypted_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypt one submission CSV file into .enc format."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to submission CSV. If omitted, auto-detect one CSV in the current directory.",
    )
    parser.add_argument(
        "--public-key",
        default=None,
        help="Path to public key PEM. Defaults to ../encryption/public_key.pem.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .enc file path. Defaults to <input>.enc",
    )
    return parser.parse_args()


def detect_single_csv() -> Path:
    candidates = [
        p
        for p in Path.cwd().glob("*.csv")
        if not p.name.endswith(".enc.csv") and not p.name.endswith(".template.csv")
    ]
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "none"
        raise ValueError(
            f"Expected exactly one CSV in {Path.cwd()}, found {len(candidates)}: {names}. "
            "Pass --input explicitly."
        )
    return candidates[0]


def resolve_public_key(user_path: str | None) -> Path:
    if user_path:
        return Path(user_path)
    local_candidate = ROOT / "encryption" / "public_key.pem"
    if local_candidate.exists():
        return local_candidate
    raise FileNotFoundError(
        "Public key not found. Place it at encryption/public_key.pem or pass --public-key."
    )


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input) if args.input else detect_single_csv()
    output_enc = Path(args.output) if args.output else Path(f"{input_csv}.enc")
    public_key_path = resolve_public_key(args.public_key)

    try:
        public_key = load_public_key(public_key_path)
    except Exception as exc:  # noqa: BLE001 - clear setup guidance
        raise ValueError(
            "Failed to read public key. Ask organizers to publish encryption/public_key.pem."
        ) from exc
    plaintext = input_csv.read_bytes()
    payload = encrypt_bytes(plaintext, public_key)
    write_encrypted_file(output_enc, payload)
    print(f"Encrypted submission written to: {output_enc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
