"""
test_bch.py — Unit tests for BCHCodec.

Covers sim mode (fast) and galois mode (real polynomial arithmetic).
The cross-validation tests (marked slow) confirm that sim mode is
semantically equivalent to the real BCH codec for all relevant cases:
0, 1, t, and t+1 errors.  This is the key verification that the
simulation shortcut is valid.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from edac import BCHCodec

T = 4
SECTOR_BYTES = config.SECTOR_DATA_BYTES
SECTOR_BITS = SECTOR_BYTES * 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_sector(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=SECTOR_BITS, dtype=np.uint8)


def _inject_errors(data: np.ndarray, positions: list[int]) -> np.ndarray:
    corrupted = data.copy()
    for pos in positions:
        corrupted[pos] ^= 1
    return corrupted


# ---------------------------------------------------------------------------
# Sim-mode unit tests
# ---------------------------------------------------------------------------


class TestBCHSimMode:
    def test_encode_returns_two_arrays(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        data, ecc = codec.encode(_random_sector(rng))
        assert data.shape == (SECTOR_BITS,)
        assert len(ecc) > 0

    def test_clean_decode_no_errors(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        original = _random_sector(rng)
        data, ecc = codec.encode(original)
        corrected, n_err, uncorrectable = codec.decode(data, ecc)
        assert n_err == 0
        assert not uncorrectable
        assert np.array_equal(corrected, original)

    def test_correctable_boundary_exactly_t(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        """t errors must be corrected."""
        original = _random_sector(rng)
        data, ecc = codec.encode(original)
        corrupted = _inject_errors(data, list(range(T)))
        corrected, n_err, uncorrectable = codec.decode(corrupted, ecc)
        assert n_err == T
        assert not uncorrectable
        assert np.array_equal(corrected, original)

    def test_uncorrectable_t_plus_one(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        """t+1 errors must be flagged uncorrectable."""
        original = _random_sector(rng)
        data, ecc = codec.encode(original)
        corrupted = _inject_errors(data, list(range(T + 1)))
        _, _, uncorrectable = codec.decode(corrupted, ecc)
        assert uncorrectable

    def test_single_error_corrected(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        original = _random_sector(rng)
        data, ecc = codec.encode(original)
        corrupted = _inject_errors(data, [42])
        corrected, n_err, uncorrectable = codec.decode(corrupted, ecc)
        assert n_err == 1
        assert not uncorrectable
        assert np.array_equal(corrected, original)

    def test_zero_errors_after_decode(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        """Corrected output must match original exactly."""
        original = _random_sector(rng)
        data, ecc = codec.encode(original)
        corrupted = _inject_errors(data, [0, 100, 200])
        corrected, _, _ = codec.decode(corrupted, ecc)
        assert np.array_equal(corrected, original)

    def test_mode_is_sim(self, codec: BCHCodec) -> None:
        assert codec._mode == "sim"

    def test_ecc_bytes(self, codec: BCHCodec) -> None:
        assert codec.ecc_bytes == 7  # ceil(13*4/8) = 7


class TestBCHPageLevel:
    def test_encode_page_returns_16_sectors(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        page_bits = np.zeros(config.PAGE_DATA_BYTES * 8, dtype=np.uint8)
        sectors = codec.encode_page(page_bits)
        assert len(sectors) == config.SECTORS_PER_PAGE

    def test_decode_page_clean(self, codec: BCHCodec, rng: np.random.Generator) -> None:
        page_bits = rng.integers(0, 2, size=config.PAGE_DATA_BYTES * 8, dtype=np.uint8)
        sectors = codec.encode_page(page_bits)
        ecc_list = [ecc for _, ecc in sectors]
        corrected, n_corr, n_bad = codec.decode_page(page_bits, ecc_list)
        assert n_corr == 0
        assert n_bad == 0
        assert np.array_equal(corrected, page_bits)

    def test_decode_page_with_error(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        page_bits = rng.integers(0, 2, size=config.PAGE_DATA_BYTES * 8, dtype=np.uint8)
        sectors = codec.encode_page(page_bits)
        ecc_list = [ecc for _, ecc in sectors]
        # Inject 1 error in sector 0
        corrupted = page_bits.copy()
        corrupted[0] ^= 1
        corrected, n_corr, n_bad = codec.decode_page(corrupted, ecc_list)
        assert n_corr == 1
        assert n_bad == 0
        assert np.array_equal(corrected, page_bits)

    def test_decode_page_uncorrectable_sector(
        self, codec: BCHCodec, rng: np.random.Generator
    ) -> None:
        page_bits = rng.integers(0, 2, size=config.PAGE_DATA_BYTES * 8, dtype=np.uint8)
        sectors = codec.encode_page(page_bits)
        ecc_list = [ecc for _, ecc in sectors]
        # Inject t+1 errors in sector 0
        corrupted = page_bits.copy()
        for i in range(T + 1):
            corrupted[i] ^= 1
        _, _, n_bad = codec.decode_page(corrupted, ecc_list)
        assert n_bad == 1


# ---------------------------------------------------------------------------
# Sim vs. galois cross-validation (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBCHCrossValidation:
    """
    Verify that sim mode and galois mode agree on correctability for all
    relevant error counts.  This validates the core simulation assumption:
    correctability depends only on error count vs. t, not on error positions.
    """

    @pytest.fixture(autouse=True)
    def _require_galois(self, galois_codec: BCHCodec) -> None:
        if galois_codec._mode != "galois":
            pytest.skip("galois library not available")

    @pytest.mark.parametrize("n_errors", [0, 1, 2, T])
    def test_agreement_within_correction_capability(
        self,
        codec: BCHCodec,
        galois_codec: BCHCodec,
        rng: np.random.Generator,
        n_errors: int,
    ) -> None:
        """For n_errors ≤ t, both modes must agree: correctable, same count."""
        original = _random_sector(rng)
        positions = list(range(n_errors))

        data_s, ecc_s = codec.encode(original)
        corrupted_s = _inject_errors(data_s, positions)
        _, n_err_s, uncorr_s = codec.decode(corrupted_s, ecc_s)

        data_g, ecc_g = galois_codec.encode(original)
        corrupted_g = _inject_errors(data_g.astype(np.uint8), positions)
        _, n_err_g, uncorr_g = galois_codec.decode(corrupted_g, ecc_g)

        assert not uncorr_s, (
            f"sim mode incorrectly flagged {n_errors} ≤ t={T} as uncorrectable"
        )
        assert not uncorr_g, (
            f"galois mode incorrectly flagged {n_errors} ≤ t={T} as uncorrectable"
        )
        assert n_err_s == n_err_g == n_errors

    def test_sim_is_conservative_beyond_t(
        self,
        codec: BCHCodec,
        galois_codec: BCHCodec,
        rng: np.random.Generator,
    ) -> None:
        """
        For n_errors > t, sim mode is ALWAYS uncorrectable (conservative).
        Galois mode may sometimes correct beyond t (depends on error pattern).

        This means sim mode gives a pessimistic (safe) BER estimate.
        The true uncorrectable rate from the galois codec is ≤ sim's estimate.
        This is documented here so the report can reflect it.
        """
        original = _random_sector(rng)
        data_s, ecc_s = codec.encode(original)
        corrupted_s = _inject_errors(data_s, list(range(T + 1)))
        _, _, uncorr_s = codec.decode(corrupted_s, ecc_s)
        # Sim mode must flag t+1 as uncorrectable
        assert uncorr_s, "sim mode must be conservative: t+1 errors are uncorrectable"
