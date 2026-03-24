"""
test_nand_flash_model.py — Unit tests for NANDFlashModel.

Covers: initialisation, write/read, single and bulk bit-flip injection,
double-flip cancellation, fill_random, reset, and error-query helpers.
"""

from __future__ import annotations

import numpy as np

from nand_flash_model import NANDFlashModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flip_addresses(
    memory: NANDFlashModel, page: int, bit_offsets: list[int]
) -> np.ndarray:
    """Return flat bit addresses for given offsets within a page."""
    return np.array([page * memory.page_bits + b for b in bit_offsets], dtype=np.int64)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_zero_errors(memory: NANDFlashModel) -> None:
    assert memory.total_error_count() == 0
    assert len(memory.pages_with_errors()) == 0


def test_init_shape(memory: NANDFlashModel) -> None:
    assert memory._reference.shape == (memory.n_pages, memory.page_bits)
    assert memory._stored.shape == (memory.n_pages, memory.page_bits)


def test_init_stats_zero(memory: NANDFlashModel) -> None:
    assert memory.stats["total_injected_errors"] == 0
    assert memory.stats["total_corrected_errors"] == 0
    assert memory.stats["total_uncorrectable_pages"] == 0


# ---------------------------------------------------------------------------
# Write / Read
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(memory: NANDFlashModel, rng: np.random.Generator) -> None:
    data = rng.integers(0, 2, size=memory.page_bits, dtype=np.bool_)
    memory.write_page(0, data)
    assert np.array_equal(memory.read_page(0), data)
    assert np.array_equal(memory.read_reference(0), data)


def test_write_clears_existing_errors(
    memory: NANDFlashModel, rng: np.random.Generator
) -> None:
    memory.fill_random()
    memory.inject_bit_flip(0)
    assert memory.page_error_count(0) >= 1
    # Writing new data should reset error count
    data = rng.integers(0, 2, size=memory.page_bits, dtype=np.bool_)
    memory.write_page(0, data)
    assert memory.page_error_count(0) == 0


def test_read_returns_copy(memory: NANDFlashModel, rng: np.random.Generator) -> None:
    data = rng.integers(0, 2, size=memory.page_bits, dtype=np.bool_)
    memory.write_page(0, data)
    page = memory.read_page(0)
    page[0] = not page[0]  # mutate the returned copy
    assert np.array_equal(memory.read_page(0), data)  # original unchanged


# ---------------------------------------------------------------------------
# Single bit-flip injection
# ---------------------------------------------------------------------------


def test_inject_bit_flip_increments_error(memory: NANDFlashModel) -> None:
    memory.fill_random()
    assert memory.page_error_count(0) == 0
    memory.inject_bit_flip(0)
    assert memory.page_error_count(0) == 1
    assert memory.stats["total_injected_errors"] == 1


def test_inject_bit_flip_double_cancels(memory: NANDFlashModel) -> None:
    """Flipping the same bit twice should restore the original value."""
    memory.fill_random()
    memory.inject_bit_flip(5)
    memory.inject_bit_flip(5)
    assert memory.page_error_count(0) == 0
    assert memory.total_error_count() == 0


def test_inject_bit_flip_correct_page(memory: NANDFlashModel) -> None:
    memory.fill_random()
    # Flip a bit in page 2
    addr = 2 * memory.page_bits
    memory.inject_bit_flip(addr)
    assert memory.page_error_count(2) == 1
    assert memory.page_error_count(0) == 0
    assert memory.page_error_count(1) == 0


# ---------------------------------------------------------------------------
# Bulk bit-flip injection
# ---------------------------------------------------------------------------


def test_inject_bit_flips_empty_array(memory: NANDFlashModel) -> None:
    memory.fill_random()
    memory.inject_bit_flips(np.empty(0, dtype=np.int64))
    assert memory.total_error_count() == 0
    assert memory.stats["total_injected_errors"] == 0


def test_inject_bit_flips_multiple_bits(memory: NANDFlashModel) -> None:
    memory.fill_random()
    addrs = _flip_addresses(memory, page=1, bit_offsets=[0, 1, 2])
    memory.inject_bit_flips(addrs)
    assert memory.page_error_count(1) == 3
    assert memory.page_error_count(0) == 0
    assert memory.stats["total_injected_errors"] == 3


def test_inject_bit_flips_double_cancel(memory: NANDFlashModel) -> None:
    """Injecting the same address twice should cancel out."""
    memory.fill_random()
    addrs = _flip_addresses(memory, page=0, bit_offsets=[10, 10])
    memory.inject_bit_flips(addrs)
    # The XOR of the same address twice is identity — net zero errors
    assert memory.page_error_count(0) == 0


def test_inject_bit_flips_cross_page(memory: NANDFlashModel) -> None:
    memory.fill_random()
    addr_p0 = _flip_addresses(memory, page=0, bit_offsets=[0])
    addr_p2 = _flip_addresses(memory, page=2, bit_offsets=[0, 1])
    memory.inject_bit_flips(np.concatenate([addr_p0, addr_p2]))
    assert memory.page_error_count(0) == 1
    assert memory.page_error_count(2) == 2
    assert memory.total_error_count() == 3


# ---------------------------------------------------------------------------
# fill_random / reset
# ---------------------------------------------------------------------------


def test_fill_random_no_errors(memory: NANDFlashModel) -> None:
    memory.fill_random()
    assert memory.total_error_count() == 0
    assert np.array_equal(memory._reference, memory._stored)


def test_fill_random_nonzero(memory: NANDFlashModel) -> None:
    memory.fill_random()
    # With 4 pages of 65,536 bits it is astronomically unlikely all bits are 0
    assert memory._stored.any()


def test_reset_clears_all(memory: NANDFlashModel) -> None:
    memory.fill_random()
    memory.inject_bit_flip(0)
    memory.reset()
    assert memory.total_error_count() == 0
    assert not memory._stored.any()
    assert not memory._reference.any()
    assert memory.stats["total_injected_errors"] == 0


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def test_pages_with_errors(memory: NANDFlashModel) -> None:
    memory.fill_random()
    assert len(memory.pages_with_errors()) == 0
    memory.inject_bit_flip(1 * memory.page_bits)  # page 1
    memory.inject_bit_flip(3 * memory.page_bits)  # page 3
    dirty = set(memory.pages_with_errors().tolist())
    assert dirty == {1, 3}


def test_pages_exceeding_threshold(memory: NANDFlashModel) -> None:
    memory.fill_random()
    # Inject 5 errors in page 0 (exceeds t=4) and 2 in page 1 (does not)
    addrs_p0 = _flip_addresses(memory, page=0, bit_offsets=list(range(5)))
    addrs_p1 = _flip_addresses(memory, page=1, bit_offsets=[0, 1])
    memory.inject_bit_flips(np.concatenate([addrs_p0, addrs_p1]))
    exceeding = memory.pages_exceeding_threshold(4).tolist()
    assert 0 in exceeding
    assert 1 not in exceeding
