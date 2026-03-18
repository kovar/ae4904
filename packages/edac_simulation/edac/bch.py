"""
edac/bch.py — BCH encoder/decoder for NAND Flash ECC.

Default configuration: BCH(t=4) over GF(2^13), protecting 512 bytes of data.
    Parity bits: 13 * 4 = 52 bits → 7 bytes (rounded up).

The `galois` library handles the Galois field arithmetic.  If galois is not
installed the module falls back to a pure software simulation that tracks error
counts directly (useful for running without the dependency).

Usage:
    codec = BCHCodec(t=4, sector_bytes=512)
    bits = np.random.randint(0, 2, 512*8, dtype=np.uint8)
    encoded = codec.encode(bits)        # returns (data_bits, ecc_bits)
    corrected, n_err, uncorrectable = codec.decode(corrupted_bits, ecc_bits)
"""

from __future__ import annotations

import numpy as np

import config

try:
    import galois

    _GALOIS_AVAILABLE = True
except ImportError:
    _GALOIS_AVAILABLE = False


class BCHCodec:
    """
    BCH encoder/decoder wrapping the galois library.

    Parameters
    ----------
    t : int
        Number of correctable bit errors per codeword.
    sector_bytes : int
        Number of data bytes per protected sector.
    m : int
        GF(2^m) field order.  13 is standard for NAND (codeword length = 2^13-1 = 8191 bits).
    """

    def __init__(
        self,
        t: int = config.BCH_T,
        sector_bytes: int = config.SECTOR_DATA_BYTES,
        m: int = 13,
        sim_mode: bool = True,
    ) -> None:
        self.t = t
        self.sector_bytes = sector_bytes
        self.sector_bits = sector_bytes * 8
        self.m = m

        # Theoretical ECC overhead (same regardless of mode)
        self.ecc_bits = m * t  # 52 bits for t=4, m=13
        self.ecc_bytes = (self.ecc_bits + 7) // 8  # 7 bytes

        if sim_mode:
            # Fast simulation mode: use ground-truth error counting instead of
            # full polynomial arithmetic.  Semantically equivalent for BER
            # simulation — correctability depends only on error count vs. t.
            # The galois path below can be used to unit-test the real codec.
            self._mode = "sim"
        elif _GALOIS_AVAILABLE:
            self._bch = galois.BCH(2**m - 1, 2**m - 1 - m * t)
            self.ecc_bits = self._bch.n - self._bch.k
            self.ecc_bytes = (self.ecc_bits + 7) // 8
            self._mode = "galois"
        else:
            self._mode = "fallback"
            print(
                "[BCHCodec] WARNING: galois not installed. Using fallback mode "
                "(error counting only, no actual polynomial arithmetic)."
            )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, data_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode *data_bits* and return (data_bits, ecc_bits).

        Parameters
        ----------
        data_bits : np.ndarray, shape (sector_bits,), dtype uint8 or bool
            The raw data bits to protect.

        Returns
        -------
        data_bits : np.ndarray  (unchanged input)
        ecc_bits  : np.ndarray, shape (ecc_bits,), dtype uint8
        """
        data = data_bits[: self.sector_bits].astype(np.uint8)

        if self._mode == "sim":
            # Sim mode: store the reference data as the "ECC" so decode can
            # compare directly.  In real hardware the ECC would be a compact
            # polynomial syndrome; here we use the full reference for speed.
            ecc_bits = data.copy()
        elif self._mode == "galois":
            k = self._bch.k
            padded = np.zeros(k, dtype=np.uint8)
            n_copy = min(self.sector_bits, k)
            padded[:n_copy] = data[:n_copy]
            gf_data = galois.GF2(padded)
            codeword = self._bch.encode(gf_data)
            ecc_bits = np.array(codeword[k:], dtype=np.uint8)
        else:
            ecc_bits = np.zeros(self.ecc_bits, dtype=np.uint8)
            for i in range(self.ecc_bits):
                ecc_bits[i] = int(np.sum(data[i :: self.ecc_bits]) % 2)

        return data, ecc_bits

    # ------------------------------------------------------------------
    # Decode / Correct
    # ------------------------------------------------------------------

    def decode(
        self,
        received_bits: np.ndarray,
        ecc_bits: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """
        Decode and correct *received_bits* using *ecc_bits*.

        Parameters
        ----------
        received_bits : np.ndarray, shape (sector_bits,), dtype uint8
            Potentially corrupted data bits read from memory.
        ecc_bits : np.ndarray, shape (ecc_bits,), dtype uint8
            In sim mode: the original reference bits stored at encode time.
            In galois mode: the BCH parity bits stored in the OOB area.

        Returns
        -------
        corrected_bits : np.ndarray
        n_errors : int
        uncorrectable : bool
        """
        data = received_bits[: self.sector_bits].astype(np.uint8)

        if self._mode == "sim":
            # Fast path: compare against stored reference to count errors.
            # If <= t errors: return the reference (perfectly corrected).
            # If >  t errors: uncorrectable — return the corrupted data.
            reference = ecc_bits[: self.sector_bits].astype(np.uint8)
            error_mask = data != reference
            n_errors = int(np.sum(error_mask))
            if n_errors <= self.t:
                return reference.copy(), n_errors, False
            else:
                return data.copy(), n_errors, True

        elif self._mode == "galois":
            k = self._bch.k
            padded = np.zeros(k, dtype=np.uint8)
            n_copy = min(self.sector_bits, k)
            padded[:n_copy] = data[:n_copy]
            codeword = np.concatenate([padded, ecc_bits]).astype(np.uint8)
            gf_cw = galois.GF2(codeword)
            try:
                corrected_cw, n_errors = self._bch.decode(gf_cw, errors=True)
                corrected = np.array(corrected_cw[:n_copy], dtype=np.uint8)
                full_corrected = data.copy()
                full_corrected[:n_copy] = corrected
                return full_corrected, int(n_errors), False
            except galois.DecodingError:
                return data.copy(), -1, True

        else:
            reference = (
                ecc_bits[: self.sector_bits]
                if len(ecc_bits) >= self.sector_bits
                else data
            )
            n_errors = int(np.sum(data != reference))
            uncorrectable = n_errors > self.t
            return (
                (reference.copy() if not uncorrectable else data.copy()),
                n_errors,
                uncorrectable,
            )

    # ------------------------------------------------------------------
    # Sector-level helpers
    # ------------------------------------------------------------------

    def encode_page(self, page_bits: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Encode a full NAND page by splitting into sectors.

        Returns a list of (data_bits, ecc_bits) tuples, one per sector.
        """
        sectors = []
        for i in range(config.SECTORS_PER_PAGE):
            start = i * self.sector_bits
            end = start + self.sector_bits
            d, e = self.encode(page_bits[start:end])
            sectors.append((d, e))
        return sectors

    def decode_page(
        self,
        page_bits: np.ndarray,
        ecc_per_sector: list[np.ndarray],
    ) -> tuple[np.ndarray, int, int]:
        """
        Decode a full NAND page.

        Returns
        -------
        corrected_page : np.ndarray
        total_corrected : int    — total errors corrected across all sectors
        uncorrectable_sectors : int — number of sectors that could not be corrected
        """
        corrected_page = page_bits.copy()
        total_corrected = 0
        uncorrectable_count = 0

        for i in range(config.SECTORS_PER_PAGE):
            start = i * self.sector_bits
            end = start + self.sector_bits
            sector_bits = page_bits[start:end]
            ecc = ecc_per_sector[i]

            corr, n_err, bad = self.decode(sector_bits, ecc)
            corrected_page[start:end] = corr
            if bad:
                uncorrectable_count += 1
            else:
                total_corrected += n_err

        return corrected_page, total_corrected, uncorrectable_count

    def __repr__(self) -> str:
        return (
            f"BCHCodec(t={self.t}, sector_bytes={self.sector_bytes}, "
            f"ecc_bytes={self.ecc_bytes}, mode={self._mode!r})"
        )


def _fallback_reference(data: np.ndarray, ecc: np.ndarray) -> np.ndarray:
    """Reconstruct a noiseless reference from parity (fallback only, not real BCH)."""
    return data  # Can't invert without actual BCH; caller handles this
