"""
nand_flash_model.py — Page-level NAND Flash memory model.

Represents a scaled-down slice of the full memory array as a 2-D numpy bit
array of shape (n_pages, page_bits).  Tracks both the "stored" (potentially
corrupted) state and the "reference" (original written) state so that error
counts can be verified precisely.

The model is deliberately flat: it does not simulate blocks, planes, or chips.
The full NAND hierarchy (chip → plane → block → page → sector) is omitted
because it has no bearing on the EDAC correctability or BER results:
  - SEUs are modelled as independent uniform bit flips, so spatial grouping
    does not change the Poisson statistics.
  - BCH correctability depends only on error count within a sector, not on
    where that sector sits in the physical hierarchy.
  - Scrubbing iterates over pages; the block grouping is irrelevant.

A structured hierarchy would only be needed to model MBU clustering, retention
errors, or block-level wear — none of which are in scope for this simulation.
"""

from __future__ import annotations

import config
import numpy as np
from numpy.random import Generator


class NANDFlashModel:
    """
    Simulated NAND Flash memory slice.

    Parameters
    ----------
    n_pages : int
        Number of pages to simulate.  Defaults to SIM_MEMORY_BYTES / PAGE_DATA_BYTES.
    rng : numpy Generator, optional
    """

    def __init__(
        self,
        n_pages: int | None = None,
        rng: Generator | None = None,
    ) -> None:
        if n_pages is None:
            n_pages = config.SIM_MEMORY_BYTES // config.PAGE_DATA_BYTES

        self.n_pages = n_pages
        self.page_bits = config.PAGE_DATA_BYTES * 8
        self.total_bits = n_pages * self.page_bits
        self.rng = rng if rng is not None else np.random.default_rng(config.RANDOM_SEED)

        # Memory arrays: uint8 for efficiency; stored as individual bits via
        # a flat boolean array (True = 1, False = 0).
        # shape: (n_pages, page_bits)
        self._reference: np.ndarray = np.zeros(
            (n_pages, self.page_bits), dtype=np.bool_
        )
        self._stored: np.ndarray = np.zeros((n_pages, self.page_bits), dtype=np.bool_)

        # Per-page error count (number of bits that differ from reference)
        self._error_count: np.ndarray = np.zeros(n_pages, dtype=np.int32)

        # Statistics
        self.stats = {
            "total_injected_errors": 0,
            "total_corrected_errors": 0,
            "total_uncorrectable_pages": 0,
        }

    # ------------------------------------------------------------------
    # Write / Read
    # ------------------------------------------------------------------

    def write_page(self, page_idx: int, data_bits: np.ndarray) -> None:
        """Write a page of data (also updates the reference)."""
        self._reference[page_idx] = data_bits.astype(np.bool_)
        self._stored[page_idx] = data_bits.astype(np.bool_)
        self._error_count[page_idx] = 0

    def read_page(self, page_idx: int) -> np.ndarray:
        """Return the stored (possibly corrupted) bits for a page."""
        return self._stored[page_idx].copy()

    def read_reference(self, page_idx: int) -> np.ndarray:
        """Return the original written bits (ground truth) for a page."""
        return self._reference[page_idx].copy()

    def write_corrected_page(self, page_idx: int, corrected_bits: np.ndarray) -> None:
        """Write back a corrected page (used by the scrubber)."""
        self._stored[page_idx] = corrected_bits.astype(np.bool_)
        self._error_count[page_idx] = int(
            np.sum(self._stored[page_idx] != self._reference[page_idx])
        )

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    def inject_bit_flip(self, flat_bit_address: int) -> None:
        """
        Flip a single bit at the given flat address [0, total_bits).

        The flat address maps to (page_idx, bit_within_page).
        """
        page_idx = flat_bit_address // self.page_bits
        bit_idx = flat_bit_address % self.page_bits
        self._stored[page_idx, bit_idx] ^= True
        self._error_count[page_idx] = int(
            np.sum(self._stored[page_idx] != self._reference[page_idx])
        )
        self.stats["total_injected_errors"] += 1

    def inject_bit_flips(self, flat_bit_addresses: np.ndarray) -> None:
        """Flip multiple bits given an array of flat addresses."""
        if len(flat_bit_addresses) == 0:
            return
        page_indices = flat_bit_addresses // self.page_bits
        bit_indices = flat_bit_addresses % self.page_bits
        self._stored[page_indices, bit_indices] ^= True
        # Update error counts for affected pages
        affected_pages = np.unique(page_indices)
        for p in affected_pages:
            self._error_count[p] = int(np.sum(self._stored[p] != self._reference[p]))
        self.stats["total_injected_errors"] += len(flat_bit_addresses)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def page_error_count(self, page_idx: int) -> int:
        """Number of bits in page that differ from the reference."""
        return int(self._error_count[page_idx])

    def total_error_count(self) -> int:
        """Total number of bit errors across all pages."""
        return int(np.sum(self._error_count))

    def pages_with_errors(self) -> np.ndarray:
        """Indices of pages that have at least one error."""
        return np.where(self._error_count > 0)[0]

    def pages_exceeding_threshold(self, threshold: int) -> np.ndarray:
        """Indices of pages with more errors than the EDAC can correct."""
        return np.where(self._error_count > threshold)[0]

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def fill_random(self) -> None:
        """Fill memory with pseudo-random data (realistic content)."""
        data = self.rng.integers(
            0, 2, size=(self.n_pages, self.page_bits), dtype=np.bool_
        )
        self._reference = data.copy()
        self._stored = data.copy()
        self._error_count[:] = 0

    def reset(self) -> None:
        """Reset memory to all-zeros and clear statistics."""
        self._reference[:] = False
        self._stored[:] = False
        self._error_count[:] = 0
        self.stats = {k: 0 for k in self.stats}

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NANDFlashModel(n_pages={self.n_pages}, "
            f"page_bits={self.page_bits}, "
            f"total_MB={self.total_bits / 8 / 1024**2:.2f})"
        )
