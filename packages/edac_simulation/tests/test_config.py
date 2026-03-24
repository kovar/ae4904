"""
test_config.py — Analytical consistency checks for config.py.

These tests verify that all derived parameters match their hand-calculated
values and that physical constraints (OOB capacity, mission arithmetic) are
satisfied.  A failure here means a parameter was changed without updating
dependent values, which would silently corrupt simulation results.
"""

from __future__ import annotations

import math

import config


def test_total_bits():
    assert config.TOTAL_BITS == 64 * 8 * 1024**3


def test_sectors_per_page():
    assert config.SECTORS_PER_PAGE == config.PAGE_DATA_BYTES // config.SECTOR_DATA_BYTES
    assert config.SECTORS_PER_PAGE == 16


def test_total_pages():
    expected = config.PAGES_PER_BLOCK * config.BLOCKS_PER_CHIP * config.NAND_NUM_CHIPS
    assert config.TOTAL_PAGES == expected


def test_seu_rate_derived_from_day_rate():
    assert math.isclose(
        config.SEU_RATE_BIT_S, config.SEU_RATE_BIT_DAY / 86400, rel_tol=1e-9
    )


def test_mission_duration_seconds():
    expected_s = config.MISSION_DURATION_YEARS * 365.25 * 86400
    assert math.isclose(config.MISSION_DURATION_S, expected_s, rel_tol=1e-9)


def test_bch_ecc_bytes():
    # BCH(t) over GF(2^m): parity bits = m * t; round up to bytes
    expected = math.ceil(13 * config.BCH_T / 8)
    assert config.BCH_ECC_BYTES_PER_SECTOR == expected


def test_ecc_fits_in_oob():
    """Total ECC bytes per page must not exceed the available OOB area."""
    ecc_per_page = config.BCH_ECC_BYTES_PER_SECTOR * config.SECTORS_PER_PAGE
    assert ecc_per_page <= config.PAGE_OOB_BYTES, (
        f"ECC overhead {ecc_per_page} B exceeds OOB {config.PAGE_OOB_BYTES} B"
    )


def test_num_chips():
    assert (
        config.NAND_NUM_CHIPS
        == config.NAND_TOTAL_CAPACITY_GB // config.NAND_CHIP_CAPACITY_GB
    )


def test_scrub_period_seconds():
    assert math.isclose(
        config.SCRUB_PERIOD_S, config.SCRUB_PERIOD_HOURS * 3600, rel_tol=1e-9
    )


def test_sim_total_bits():
    assert config.SIM_TOTAL_BITS == config.SIM_MEMORY_BYTES * 8
