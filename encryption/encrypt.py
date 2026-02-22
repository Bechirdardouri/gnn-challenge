from __future__ import annotations

import argparse
from pathlib import Path

from crypto import encrypt_bytes, load_public_key, write_encrypted_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypt a CSV submission using the competition public key."
    )
    parser.add_argument("input_csv", help="Path to plaintext submission CSV")
    parser.add_argument("public_key_pem", help="Path to public_key.pem")
    parser.add_argument("output_enc", help="Path to output encrypted .enc file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_enc)

    plaintext = input_path.read_bytes()
    try:
        public_key = load_public_key(args.public_key_pem)
    except Exception as exc:  # noqa: BLE001 - show clear setup message
        raise ValueError(
            "Failed to read public key. Generate a keypair with "
            "`python encryption/generate_keys.py` and commit encryption/public_key.pem."
        ) from exc
    payload = encrypt_bytes(plaintext, public_key)
    write_encrypted_file(output_path, payload)
    print(f"Encrypted submission written to: {output_path}")


if __name__ == "__main__":
    main()
