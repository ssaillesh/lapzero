"""ULIDs for lap and episode IDs: 26-character, time-sortable identifiers (contract Section 5.2)."""

from __future__ import annotations

import secrets
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def encode_ulid(timestamp_ms: int, randomness: bytes) -> str:
    """Encode a ULID: a 48-bit millisecond timestamp then 80 random bits, in Crockford base32.

    Args:
        timestamp_ms: Milliseconds since the Unix epoch, in [0, 2**48).
        randomness: Exactly 10 bytes.

    Returns:
        26-character ULID; ULIDs with later timestamps sort after earlier ones.

    Raises:
        ValueError: If `timestamp_ms` or `randomness` is out of range.
    """
    if not 0 <= timestamp_ms < 2**48:
        raise ValueError(f"timestamp_ms must be in [0, 2**48), got {timestamp_ms}")
    if len(randomness) != 10:
        raise ValueError(f"randomness must be 10 bytes, got {len(randomness)}")
    value = (timestamp_ms << 80) | int.from_bytes(randomness, "big")
    return "".join(_CROCKFORD[(value >> shift) & 0b11111] for shift in range(125, -1, -5))


def new_ulid() -> str:
    """Return a ULID for the current time. IDs must be unique, not reproducible, so not seeded."""
    return encode_ulid(time.time_ns() // 1_000_000, secrets.token_bytes(10))
