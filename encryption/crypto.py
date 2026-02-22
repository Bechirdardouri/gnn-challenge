from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


SCHEME_VERSION = 1
AAD = b"gnn-competition-submission-v1"


def _b64_encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("utf-8")


def _b64_decode(raw: str) -> bytes:
    return base64.b64decode(raw.encode("utf-8"))


def load_public_key(path: str | Path):
    content = Path(path).read_bytes()
    return serialization.load_pem_public_key(content)


def load_private_key(path: str | Path, password: str | None = None):
    content = Path(path).read_bytes()
    return serialization.load_pem_private_key(
        content,
        password=password.encode("utf-8") if password else None,
    )


def encrypt_bytes(plaintext: bytes, public_key) -> dict[str, Any]:
    aes_key = os.urandom(32)
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, AAD)

    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return {
        "version": SCHEME_VERSION,
        "aad": _b64_encode(AAD),
        "encrypted_key": _b64_encode(encrypted_key),
        "nonce": _b64_encode(nonce),
        "ciphertext": _b64_encode(ciphertext),
    }


def decrypt_payload(payload: dict[str, Any], private_key) -> bytes:
    version = int(payload.get("version", 0))
    if version != SCHEME_VERSION:
        raise ValueError(f"Unsupported encryption version: {version}")

    encrypted_key = _b64_decode(payload["encrypted_key"])
    nonce = _b64_decode(payload["nonce"])
    ciphertext = _b64_decode(payload["ciphertext"])
    aad = _b64_decode(payload["aad"])

    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


def write_encrypted_file(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_encrypted_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
