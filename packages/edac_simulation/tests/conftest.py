"""
Shared fixtures for the edac_simulation test suite.

The simulation modules use bare `import config` / `import nand_flash_model` etc.
(they live flat in packages/edac_simulation/, not in a proper package).  pytest's
`pythonpath` setting in pyproject.toml adds that directory to sys.path so all
imports resolve correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from bad_block_manager import BadBlockManager
from edac import BCHCodec
from nand_flash_model import NANDFlashModel
from scrubbing import Scrubber

SEED = 42

# Use a very small page size so fixtures create tiny memories that run fast.
# One sector per page keeps page/sector logic simple in most tests.
TINY_SECTOR_BYTES = 512  # matches real config — one BCH codeword
TINY_N_PAGES = 4


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def codec() -> BCHCodec:
    """BCHCodec in fast sim mode (default for all unit tests)."""
    return BCHCodec(t=4, sector_bytes=TINY_SECTOR_BYTES, sim_mode=True)


@pytest.fixture
def galois_codec() -> BCHCodec:
    """BCHCodec using real GF polynomial arithmetic (slow — mark tests slow)."""
    return BCHCodec(t=4, sector_bytes=TINY_SECTOR_BYTES, sim_mode=False)


@pytest.fixture
def memory(rng: np.random.Generator) -> NANDFlashModel:
    """4-page memory model with a fixed seed."""
    return NANDFlashModel(n_pages=TINY_N_PAGES, rng=rng)


@pytest.fixture
def filled_memory(memory: NANDFlashModel) -> NANDFlashModel:
    """4-page memory filled with random data (reference == stored, no errors)."""
    memory.fill_random()
    return memory


@pytest.fixture
def scrubber(filled_memory: NANDFlashModel, codec: BCHCodec) -> Scrubber:
    """Scrubber attached to filled_memory with encoded ECC."""
    return Scrubber(filled_memory, codec)


@pytest.fixture
def bbm(rng: np.random.Generator) -> BadBlockManager:
    """Small-scale BBM (200 blocks, 10 spares) for fast unit tests."""
    return BadBlockManager(
        total_blocks_per_chip=100,
        n_chips=2,
        pages_per_block=4,
        spare_blocks_per_chip=10,
        factory_bad_fraction=0.05,
        rng=rng,
    )
