"""Unit tests for lapzero/record/ulid.py: ULID encoding and generation."""

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lapzero.io.telemetry import ULID_PATTERN
from lapzero.record.ulid import encode_ulid, new_ulid

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
TIMESTAMPS = st.integers(0, 2**48 - 1)
RANDOMNESS = st.binary(min_size=10, max_size=10)


def _decode(ulid: str) -> tuple[int, bytes]:
    value = 0
    for char in ulid:
        value = value * 32 + CROCKFORD.index(char)
    return value >> 80, (value & (2**80 - 1)).to_bytes(10, "big")


def test_smallest_and_largest_ulids_match_the_spec_bounds() -> None:
    assert encode_ulid(0, bytes(10)) == "0" * 26
    assert encode_ulid(2**48 - 1, b"\xff" * 10) == "7" + "Z" * 25


@given(TIMESTAMPS, RANDOMNESS)
def test_ulid_is_26_crockford_chars_that_decode_back_to_its_inputs(
    timestamp_ms: int, randomness: bytes
) -> None:
    ulid = encode_ulid(timestamp_ms, randomness)
    assert re.fullmatch(ULID_PATTERN, ulid)
    assert _decode(ulid) == (timestamp_ms, randomness)


@given(TIMESTAMPS, TIMESTAMPS, RANDOMNESS, RANDOMNESS)
def test_later_timestamps_sort_after_earlier_ones(
    t1: int, t2: int, random1: bytes, random2: bytes
) -> None:
    if t1 < t2:
        assert encode_ulid(t1, random1) < encode_ulid(t2, random2)


@pytest.mark.parametrize(("timestamp_ms", "randomness"), [(-1, bytes(10)), (2**48, bytes(10))])
def test_out_of_range_timestamp_raises_value_error(timestamp_ms: int, randomness: bytes) -> None:
    with pytest.raises(ValueError, match="timestamp_ms"):
        encode_ulid(timestamp_ms, randomness)


def test_wrong_randomness_length_raises_value_error() -> None:
    with pytest.raises(ValueError, match="randomness"):
        encode_ulid(0, bytes(9))


def test_new_ulids_are_valid_and_distinct() -> None:
    ids = {new_ulid() for _ in range(100)}
    assert len(ids) == 100
    assert all(re.fullmatch(ULID_PATTERN, ulid) for ulid in ids)
