"""
config.py — Mission and hardware parameters for the EDAC simulation.
"""

# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------
MISSION_DURATION_YEARS = 2.0
MISSION_DURATION_DAYS = MISSION_DURATION_YEARS * 365.25
MISSION_DURATION_S = MISSION_DURATION_DAYS * 86400.0

ORBIT_ALTITUDE_KM = 600  # Sun-Synchronous Orbit
ORBIT_INCLINATION_DEG = 97.8  # Typical SSO inclination
ORBIT_PERIOD_S = 5765.0  # ~96 min for 600 km SSO

# ---------------------------------------------------------------------------
# Radiation environment — from SPENVIS analysis (600 km SSO)
# ---------------------------------------------------------------------------

# SEU rate per bit per day, derived directly from SPENVIS output.
# Two candidate values — select the active one below.
SEU_RATE_BIT_DAY_V1 = 1.40e-7  # initial SPENVIS estimate
SEU_RATE_BIT_DAY_V2 = 2.88e-6  # revised SPENVIS value
SEU_RATE_BIT_DAY_V3 = 3.09e-9  # mission-average SEU rate

SEU_RATE_BIT_DAY = SEU_RATE_BIT_DAY_V2  # <-- active value

# Derived: SEU rate per bit per second
SEU_RATE_BIT_S = SEU_RATE_BIT_DAY / 86400.0

# ---------------------------------------------------------------------------
# Memory component — Micron MT29F256G08 SLC NAND Flash (32 GB per chip)
# ---------------------------------------------------------------------------
NAND_TOTAL_CAPACITY_GB = 64  # Required by REQ-01
NAND_CHIP_CAPACITY_GB = 32  # MT29F256G08: 32 GB per chip
NAND_NUM_CHIPS = NAND_TOTAL_CAPACITY_GB // NAND_CHIP_CAPACITY_GB  # = 2

# Page geometry (SLC NAND, large-page)
# Source: MT29F256G08AUCABH3 datasheet, Rev. H, p. 1 — Features section
PAGE_DATA_BYTES = 8192  # User data bytes per page  (datasheet: "8192 + 448 bytes")
PAGE_OOB_BYTES = 448  # Out-Of-Band (spare) bytes per page
PAGES_PER_BLOCK = 128  # Pages per erase block     (datasheet: "Block size: 128 pages")
BLOCKS_PER_CHIP = 32768  # Erase blocks per chip     (datasheet: "256Gb: 32,768 blocks")

TOTAL_PAGES_PER_CHIP = PAGES_PER_BLOCK * BLOCKS_PER_CHIP
TOTAL_PAGES = TOTAL_PAGES_PER_CHIP * NAND_NUM_CHIPS

# Total bits in the memory array
TOTAL_BITS = NAND_TOTAL_CAPACITY_GB * 8 * 1024**3

# ---------------------------------------------------------------------------
# EDAC parameters  — design variables; sweep in simulation
# ---------------------------------------------------------------------------
BCH_T = 4  # Number of correctable bit errors per sector
SECTOR_DATA_BYTES = 512  # Data bytes protected by one BCH codeword
SECTORS_PER_PAGE = PAGE_DATA_BYTES // SECTOR_DATA_BYTES  # = 16

# BCH overhead: for t=4 over GF(2^13): parity = 13*4 = 52 bits = 7 bytes
# (rounded up to byte boundary)
BCH_ECC_BYTES_PER_SECTOR = 7  # Exact value set by galois library

# ---------------------------------------------------------------------------
# Scrubbing parameters  — key trade-off variable; sweep in analysis
# ---------------------------------------------------------------------------
SCRUB_PERIOD_HOURS = 72.0  # Default; swept in analysis
SCRUB_PERIOD_S = SCRUB_PERIOD_HOURS * 3600.0

# Interface throughput available for scrubbing (fraction of REQ-02 20 Mbps)
INTERFACE_THROUGHPUT_BPS = 20e6  # REQ-02
SCRUB_BANDWIDTH_FRACTION = 0.20  # 20 % of interface reserved for scrubbing

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
# REQ-06: uncorrectable BER threshold [/bit/s]
# "The uncorrectable Bit Error Rate shall be less than 10^-12 /bit/s over the
# mission lifetime."  The rate is normalised by both the total bit count and
# mission duration, so it is comparable across missions of different lengths.
BER_REQUIREMENT_BIT_S = 1e-12

# ---------------------------------------------------------------------------
# Bad Block Management (BBM)
# ---------------------------------------------------------------------------
# Factory bad block fraction (JEDEC JESD47 / MT29F256G08 datasheet: up to 2%)
BBM_FACTORY_BAD_FRACTION = 0.02

# Spare blocks reserved per chip for bad-block replacement (~3% of 32,768)
BBM_SPARE_FRACTION = 0.03
BBM_SPARE_BLOCKS_PER_CHIP = int(BLOCKS_PER_CHIP * BBM_SPARE_FRACTION)  # = 983

N_MONTE_CARLO_TRIALS = 10  # Number of independent mission simulations
RANDOM_SEED = 42  # For reproducibility

# Scaled-down memory for simulation (to keep run-time manageable).
# 1 MB representative slice; BER results are extrapolated to full memory array.
SIM_MEMORY_BYTES = 1 * 1024 * 1024  # 1 MB
SIM_TOTAL_BITS = SIM_MEMORY_BYTES * 8
