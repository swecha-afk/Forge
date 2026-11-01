"""Audio utilities for encoding, hashing, and validating voice notes."""

import base64
import hashlib
import os
from typing import Tuple


def compute_audio_hash(audio_bytes: bytes) -> str:
    """Computes SHA-256 hash of raw audio bytes for write-through cache lookup."""
    return hashlib.sha256(audio_bytes).hexdigest()


def decode_base64_audio(audio_base64: str) -> bytes:
    """Decodes a base64 string (including data URI prefixes) into raw audio bytes."""
    clean_b64 = audio_base64
    if "," in clean_b64:
        clean_b64 = clean_b64.split(",", 1)[1]
    return base64.b64decode(clean_b64.strip())


def encode_audio_file(file_path: str) -> Tuple[str, str]:
    """Reads a local audio file and returns (base64_string, format_extension)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lstrip(".").lower() or "wav"
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    b64_str = base64.b64encode(audio_bytes).decode("utf-8")
    return b64_str, ext

