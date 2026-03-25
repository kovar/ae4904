"""
bad_block_manager.py — Bad Block Management (BBM) table for NAND Flash.

Tracks bad blocks at full-memory scale (64 GB across both chips) as a
lightweight overlay on top of the flat simulation model.

Two sources of bad blocks are modelled:
  - Factory bad blocks: pre-seeded at initialisation, drawn from the
    Binomial(total_blocks, factory_bad_fraction) distribution.  These
    represent blocks marked defective at manufacture and replaced by
    spare blocks before the device ships.
  - Runtime bad blocks: blocks retired during the mission when the
    scrubber reports an uncorrectable sector.  Each new bad block
    consumes one spare.

The simulation slice is only 1 MB (= 1 erase block), so BBM operates at
full 64 GB scale.  When the scrubber reports N uncorrectable pages in the
simulation slice, that count is extrapolated to the full array and the
corresponding blocks are randomly retired.

REQ-06 BBM is PASS if the spare pool is never exhausted over the mission.

References:
  - JEDEC JESD47: Stress-Test-Driven Qualification of ICs (bad block limits)
  - Micron MT29F256G08AUCABH3 datasheet: 32,768 blocks per chip, 128 pages/block
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

import config


class BadBlockManager:
    """
    Bad Block Management table for the full NAND Flash array.

    Parameters
    ----------
    total_blocks_per_chip : int
        Number of erase blocks per chip.
    n_chips : int
        Number of chips in the array.
    pages_per_block : int
        Pages per erase block.
    spare_blocks_per_chip : int
        Spare blocks reserved per chip for bad-block replacement.
    factory_bad_fraction : float
        Expected fraction of blocks bad at manufacture.
    rng : numpy Generator, optional
    """

    def __init__(
        self,
        total_blocks_per_chip: int = config.BLOCKS_PER_CHIP,
        n_chips: int = config.NAND_NUM_CHIPS,
        pages_per_block: int = config.PAGES_PER_BLOCK,
        spare_blocks_per_chip: int = config.BBM_SPARE_BLOCKS_PER_CHIP,
        factory_bad_fraction: float = config.BBM_FACTORY_BAD_FRACTION,
        rng: Generator | None = None,
    ) -> None:
        self.total_blocks_per_chip = total_blocks_per_chip
        self.n_chips = n_chips
        self.total_blocks = total_blocks_per_chip * n_chips
        self.pages_per_block = pages_per_block
        self.rng = rng if rng is not None else np.random.default_rng(config.RANDOM_SEED)

        # Spare pool: one pool for the entire array
        initial_spare = spare_blocks_per_chip * n_chips

        # Pre-seed factory bad blocks (drawn independently per the Binomial model)
        n_factory = int(self.rng.binomial(self.total_blocks, factory_bad_fraction))
        factory_indices = self.rng.choice(
            self.total_blocks, size=n_factory, replace=False
        )
        self._factory_bad: set[int] = set(factory_indices.tolist())
        self._runtime_bad: set[int] = set()

        # Each factory bad block consumed one spare at manufacture
        self._spares_remaining: int = initial_spare - n_factory

    # ------------------------------------------------------------------
    # Block retirement
    # ------------------------------------------------------------------

    def retire_block(self, block_idx: int) -> bool:
        """
        Mark *block_idx* as a runtime bad block and consume one spare.

        Idempotent: if the block is already bad (factory or runtime),
        returns True without changing the spare count.

        Returns
        -------
        bool
            True if a spare was available (or block already bad),
            False if the spare pool is exhausted.
        """
        if block_idx in self._factory_bad or block_idx in self._runtime_bad:
            return True
        self._runtime_bad.add(block_idx)
        self._spares_remaining -= 1
        return self._spares_remaining >= 0

    def is_bad(self, block_idx: int) -> bool:
        """Return True if block_idx is marked bad (factory or runtime)."""
        return block_idx in self._factory_bad or block_idx in self._runtime_bad

    # ------------------------------------------------------------------
    # Bulk event registration (from simulation slice)
    # ------------------------------------------------------------------

    def register_uncorrectable(
        self,
        n_pages_sim: int,
        scale_factor: float,
    ) -> dict:
        """
        Register uncorrectable-page events from the simulation slice.

        Extrapolates *n_pages_sim* to the full memory array using
        *scale_factor* (= TOTAL_PAGES / sim_n_pages), randomly distributes
        them across the full block address space, and retires each
        affected block.

        Parameters
        ----------
        n_pages_sim : int
            Number of uncorrectable pages found in the simulation slice.
        scale_factor : float
            Extrapolation factor from sim slice to full array.

        Returns
        -------
        dict with keys 'blocks_retired' (int) and 'spares_ok' (bool).
        """
        if n_pages_sim == 0:
            return {"blocks_retired": 0, "spares_ok": True}

        n_full_pages = max(1, round(n_pages_sim * scale_factor))
        total_pages = self.total_blocks * self.pages_per_block
        page_indices = self.rng.integers(0, total_pages, size=n_full_pages)
        block_indices = np.unique(page_indices // self.pages_per_block)

        spares_ok = True
        for b in block_indices:
            if not self.retire_block(int(b)):
                spares_ok = False

        return {
            "blocks_retired": len(block_indices),
            "spares_ok": spares_ok,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @property
    def summary(self) -> dict:
        """Return a dict of BBM statistics suitable for logging / JSON."""
        total_bad = len(self._factory_bad) + len(self._runtime_bad)
        return {
            "total_blocks": self.total_blocks,
            "factory_bad": len(self._factory_bad),
            "runtime_bad": len(self._runtime_bad),
            "total_bad": total_bad,
            "spares_remaining": self._spares_remaining,
            "spares_exhausted": self._spares_remaining < 0,
            "effective_capacity_fraction": 1.0 - total_bad / self.total_blocks,
        }

    def __repr__(self) -> str:
        s = self.summary
        return (
            f"BadBlockManager(total={s['total_blocks']}, "
            f"factory_bad={s['factory_bad']}, "
            f"runtime_bad={s['runtime_bad']}, "
            f"spares_remaining={s['spares_remaining']})"
        )
