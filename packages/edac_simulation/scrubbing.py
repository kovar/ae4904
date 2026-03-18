"""
scrubbing.py — Periodic memory scrubbing algorithm.

Memory scrubbing reads every page, corrects errors via EDAC, and writes the
corrected data back.  This prevents error accumulation: without scrubbing,
multiple SEUs could accumulate in one sector until they exceed the BCH(t)
correction capability, causing an uncorrectable error.

The scrubbing period is the key trade-off between:
  - Bandwidth overhead (shorter period → more scrub bandwidth consumed)
  - Data integrity   (longer period → more errors accumulate between scrubs)

REQ-02 caps net payload throughput at 20 Mbps; scrubbing must use only its
allocated fraction (config.SCRUB_BANDWIDTH_FRACTION).

This module provides:
  - Scrubber class: performs one full scrub pass over the memory model
  - scrub_period_analysis(): sweeps scrub period and reports statistics
"""

from __future__ import annotations

import numpy as np

import config
from nand_flash_model import NANDFlashModel
from edac import BCHCodec


class Scrubber:
    """
    Periodic memory scrubber.

    Parameters
    ----------
    memory : NANDFlashModel
    codec  : BCHCodec
    """

    def __init__(self, memory: NANDFlashModel, codec: BCHCodec) -> None:
        self.memory = memory
        self.codec = codec

        # Pre-encode all pages to get reference ECC (in a real system this is
        # stored in the OOB area of each NAND page at write time).
        self._ecc_store: list[list[np.ndarray]] = self._initial_encode_all()

        # Scrubbing statistics accumulated over the lifetime of this scrubber
        self.total_scrub_passes = 0
        self.total_errors_corrected = 0
        self.total_uncorrectable_pages = 0

    # ------------------------------------------------------------------
    # Initial ECC computation
    # ------------------------------------------------------------------

    def _initial_encode_all(self) -> list[list[np.ndarray]]:
        """Encode all pages at initialisation (simulates write-time ECC generation)."""
        ecc_store = []
        for page_idx in range(self.memory.n_pages):
            page_bits = self.memory.read_reference(page_idx)
            sector_eccs = []
            for s in range(config.SECTORS_PER_PAGE):
                start = s * self.codec.sector_bits
                end = start + self.codec.sector_bits
                _, ecc = self.codec.encode(page_bits[start:end])
                sector_eccs.append(ecc)
            ecc_store.append(sector_eccs)
        return ecc_store

    def update_ecc(self, page_idx: int, corrected_bits: np.ndarray) -> None:
        """Recompute and store ECC after a corrected write-back."""
        sector_eccs = []
        for s in range(config.SECTORS_PER_PAGE):
            start = s * self.codec.sector_bits
            end = start + self.codec.sector_bits
            _, ecc = self.codec.encode(corrected_bits[start:end])
            sector_eccs.append(ecc)
        self._ecc_store[page_idx] = sector_eccs

    # ------------------------------------------------------------------
    # Single scrub pass
    # ------------------------------------------------------------------

    def scrub_pass(self) -> dict:
        """
        Perform one full scrub pass over all pages.

        Reads each page, decodes with stored ECC, writes corrected data back,
        and updates the ECC for the corrected content.

        Returns
        -------
        dict with keys:
            errors_corrected   : int
            uncorrectable_pages: int
            pages_scrubbed     : int
        """
        errors_corrected = 0
        uncorrectable_pages = 0

        # Simulation optimisation: only invoke the BCH decoder on pages that
        # the memory model reports as having at least one error.  Clean pages
        # are trivially correct.  This is valid in simulation because we have
        # ground-truth error tracking; it does not affect correctness results.
        dirty_pages = set(self.memory.pages_with_errors())

        for page_idx in range(self.memory.n_pages):
            if page_idx not in dirty_pages:
                continue  # no errors — BCH would pass unchanged, skip

            page_bits = self.memory.read_page(page_idx)
            ecc_list = self._ecc_store[page_idx]

            corrected, n_corr, n_bad = self.codec.decode_page(page_bits, ecc_list)

            if n_bad > 0:
                uncorrectable_pages += 1
                self.memory.stats["total_uncorrectable_pages"] += 1
            else:
                if n_corr > 0:
                    self.memory.write_corrected_page(page_idx, corrected)
                    self.update_ecc(page_idx, corrected)
                    self.memory.stats["total_corrected_errors"] += n_corr
                errors_corrected += n_corr

        self.total_scrub_passes += 1
        self.total_errors_corrected += errors_corrected
        self.total_uncorrectable_pages += uncorrectable_pages

        return {
            "errors_corrected": errors_corrected,
            "uncorrectable_pages": uncorrectable_pages,
            "pages_scrubbed": self.memory.n_pages,
        }

    # ------------------------------------------------------------------
    # Bandwidth estimate
    # ------------------------------------------------------------------

    @staticmethod
    def scrub_bandwidth_bps(scrub_period_s: float) -> float:
        """
        Data rate required to scrub the full memory array once per scrub_period_s.

        Returns bits-per-second consumed by scrubbing.
        """
        total_bits = config.TOTAL_BITS
        return total_bits / scrub_period_s

    @staticmethod
    def max_scrub_period_s() -> float:
        """
        Maximum scrub period that keeps bandwidth below the allocated fraction.
        """
        allocated_bps = (
            config.INTERFACE_THROUGHPUT_BPS * config.SCRUB_BANDWIDTH_FRACTION
        )
        return config.TOTAL_BITS / allocated_bps

    @staticmethod
    def scrub_overhead_fraction(scrub_period_s: float) -> float:
        """Fraction of interface bandwidth consumed by scrubbing."""
        bps = Scrubber.scrub_bandwidth_bps(scrub_period_s)
        return bps / config.INTERFACE_THROUGHPUT_BPS


def scrub_period_analysis(
    periods_hours: list[float] | None = None,
    seu_rate_bit_s: float = config.SEU_RATE_BIT_S,
) -> None:
    """
    Print a table of scrubbing period vs. bandwidth and SEU accumulation risk.

    Parameters
    ----------
    periods_hours : list of floats, optional
        Scrub periods to evaluate [hours].  Defaults to a log-spaced sweep.
    """
    from radiation_model import RadiationModel

    rm = RadiationModel(seu_rate_bit_s=seu_rate_bit_s)

    if periods_hours is None:
        periods_hours = [0.5, 1, 2, 6, 12, 24, 48, 72, 168]

    max_period_s = Scrubber.max_scrub_period_s()

    print("=" * 75)
    print(
        f"Scrubbing Period Trade-off  (BCH t={config.BCH_T}, "
        f"max BW fraction={config.SCRUB_BANDWIDTH_FRACTION:.0%})"
    )
    print(f"  Max allowed scrub period: {max_period_s / 3600:.1f} h")
    print("-" * 75)
    print(
        f"{'Period':>10}  {'BW fraction':>12}  {'SEUs/page/period':>18}  "
        f"{'Exceeds BCH?':>12}  {'OK?':>5}"
    )
    print("-" * 75)

    for ph in periods_hours:
        ps = ph * 3600.0
        bw_frac = Scrubber.scrub_overhead_fraction(ps)
        seus = rm.expected_seus_per_page_per_scrub(ps)
        exceeds = seus > config.BCH_T
        bw_ok = bw_frac <= config.SCRUB_BANDWIDTH_FRACTION
        ok = bw_ok and not exceeds
        print(
            f"{ph:>8.1f}h  {bw_frac:>12.4%}  {seus:>18.4e}  "
            f"{'YES' if exceeds else 'no':>12}  {'OK' if ok else 'FAIL':>5}"
        )
    print("=" * 75)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrub period analysis")
    parser.add_argument(
        "--seu-rate-day",
        type=float,
        default=config.SEU_RATE_BIT_DAY,
        help="SEU rate per bit per day (overrides config)",
    )
    args = parser.parse_args()
    scrub_period_analysis(seu_rate_bit_s=args.seu_rate_day / 86400)
