"""
Tests for BadBlockManager — bad block tracking at full 64 GB scale.

The `bbm` fixture uses a small array (200 blocks, 20 spares) for speed.
Full-scale parameters are tested only where scale matters.
"""

from __future__ import annotations

import numpy as np

from bad_block_manager import BadBlockManager


# ---------------------------------------------------------------------------
# Factory bad blocks
# ---------------------------------------------------------------------------


def test_factory_bad_blocks_seeded(bbm: BadBlockManager) -> None:
    """Factory bad count should be in a plausible range (Binomial, p=0.05)."""
    s = bbm.summary
    # Expected ~10 (5% of 200); allow generous slack
    assert 0 < s["factory_bad"] <= 40


def test_factory_bad_blocks_deterministic() -> None:
    """Same seed produces the same factory bad blocks."""
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    bbm1 = BadBlockManager(
        total_blocks_per_chip=50,
        n_chips=1,
        spare_blocks_per_chip=5,
        factory_bad_fraction=0.10,
        rng=rng1,
    )
    bbm2 = BadBlockManager(
        total_blocks_per_chip=50,
        n_chips=1,
        spare_blocks_per_chip=5,
        factory_bad_fraction=0.10,
        rng=rng2,
    )
    assert bbm1._factory_bad == bbm2._factory_bad


# ---------------------------------------------------------------------------
# Block retirement
# ---------------------------------------------------------------------------


def test_retire_block_consumes_spare(bbm: BadBlockManager) -> None:
    """Retiring a good block decrements the spare count by 1."""
    # Find a block not already bad
    good_block = next(i for i in range(bbm.total_blocks) if not bbm.is_bad(i))
    spares_before = bbm.summary["spares_remaining"]
    result = bbm.retire_block(good_block)
    assert result is True
    assert bbm.summary["spares_remaining"] == spares_before - 1
    assert bbm.is_bad(good_block)


def test_retire_same_block_twice_is_noop(bbm: BadBlockManager) -> None:
    """Retiring an already-bad block is idempotent (no spare consumed)."""
    good_block = next(i for i in range(bbm.total_blocks) if not bbm.is_bad(i))
    bbm.retire_block(good_block)
    spares_after_first = bbm.summary["spares_remaining"]

    # Second retirement should not change spare count
    result = bbm.retire_block(good_block)
    assert result is True
    assert bbm.summary["spares_remaining"] == spares_after_first


def test_spares_exhaustion() -> None:
    """retire_block returns False when the spare pool is empty."""
    rng = np.random.default_rng(0)
    bbm = BadBlockManager(
        total_blocks_per_chip=20,
        n_chips=1,
        pages_per_block=4,
        spare_blocks_per_chip=2,
        factory_bad_fraction=0.0,  # no factory bad so all 2 spares are fresh
        rng=rng,
    )
    # Retire blocks until spares are gone
    ok_results = []
    for i in range(20):
        ok_results.append(bbm.retire_block(i))

    # At least one retirement should have returned False
    assert False in ok_results
    assert bbm.summary["spares_exhausted"]


# ---------------------------------------------------------------------------
# register_uncorrectable
# ---------------------------------------------------------------------------


def test_register_uncorrectable_scales_correctly(bbm: BadBlockManager) -> None:
    """Scaled-up page count retires at least 1 block."""
    result = bbm.register_uncorrectable(n_pages_sim=1, scale_factor=10.0)
    assert result["blocks_retired"] >= 1


def test_register_uncorrectable_zero_pages(bbm: BadBlockManager) -> None:
    """Zero uncorrectable pages → no blocks retired."""
    result = bbm.register_uncorrectable(n_pages_sim=0, scale_factor=1000.0)
    assert result["blocks_retired"] == 0
    assert result["spares_ok"] is True


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------


def test_summary_keys(bbm: BadBlockManager) -> None:
    """summary contains all expected keys."""
    expected = {
        "total_blocks",
        "factory_bad",
        "runtime_bad",
        "total_bad",
        "spares_remaining",
        "spares_exhausted",
        "effective_capacity_fraction",
    }
    assert expected <= set(bbm.summary.keys())


def test_effective_capacity_decreases(bbm: BadBlockManager) -> None:
    """Retiring blocks reduces effective_capacity_fraction."""
    cap_before = bbm.summary["effective_capacity_fraction"]
    good_block = next(i for i in range(bbm.total_blocks) if not bbm.is_bad(i))
    bbm.retire_block(good_block)
    assert bbm.summary["effective_capacity_fraction"] < cap_before
