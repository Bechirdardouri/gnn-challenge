from __future__ import annotations

import argparse
from pathlib import Path

from crypto import decrypt_payload, load_private_key, read_encrypted_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decrypt an encrypted .enc submission with the private key."
    )
    parser.add_argument("input_enc", help="Path to encrypted .enc file")
    parser.add_argument("private_key_pem", help="Path to private_key.pem")
    parser.add_argument("output_csv", help="Path to output plaintext CSV")
    parser.add_argument(
        "--private-key-password",
        default=None,
        help="Password for encrypted private key (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = read_encrypted_file(args.input_enc)
    private_key = load_private_key(args.private_key_pem, args.private_key_password)
    plaintext = decrypt_payload(payload, private_key)
    Path(args.output_csv).write_bytes(plaintext)
    print(f"Decrypted CSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()
