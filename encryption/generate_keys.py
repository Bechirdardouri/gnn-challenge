from __future__ import annotations

import argparse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RSA keypair for encrypted competition submissions."
    )
    parser.add_argument(
        "--public-key-out",
        default="encryption/public_key.pem",
        help="Path to write public key PEM.",
    )
    parser.add_argument(
        "--private-key-out",
        default="encryption/private_key.pem",
        help="Path to write private key PEM.",
    )
    parser.add_argument(
        "--private-key-password",
        default=None,
        help="Optional password to encrypt private key PEM.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing key files.",
    )
    return parser.parse_args()


def ensure_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    public_path = Path(args.public_key_out)
    private_path = Path(args.private_key_out)
    ensure_writable(public_path, args.force)
    ensure_writable(private_path, args.force)

    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

    if args.private_key_password:
        encryption_algo = serialization.BestAvailableEncryption(
            args.private_key_password.encode("utf-8")
        )
    else:
        encryption_algo = serialization.NoEncryption()

    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption_algo,
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)

    print(f"Public key written to: {public_path}")
    print(f"Private key written to: {private_path}")
    print("Set PRIVATE_KEY_PEM in GitHub Secrets using the private key content.")


if __name__ == "__main__":
    main()
