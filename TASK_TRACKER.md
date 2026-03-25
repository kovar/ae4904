# EDAC Simulation — Task Tracker

Tracks remaining work for the Python simulation code in `packages/edac_simulation/`.
Hardware deliverables (KiCad, LTSpice, report) are tracked separately.

---

## Done

- [x] `config.py` — all parameters confirmed from datasheet (MT29F256G08AUCABH3) and SPENVIS
- [x] `nand_flash_model.py` — page-level NAND memory model (reference + stored arrays)
- [x] `edac/bch.py` — BCH(t=4) over GF(2^13): sim mode (fast) and galois mode (real polynomial)
- [x] `radiation_model.py` — Poisson SEU event generator
- [x] `scrubbing.py` — full scrub pass + bandwidth trade-off table
- [x] `fault_injection.py` — orbital-rate injection and breaking-point BER sweep
- [x] `simulation.py` — Monte Carlo runner + scrub-period sweep; saves one trial's scrub_log
- [x] `analysis.py` — BER vs scrub period plot, breaking-point plot, error accumulation plot, full verification matrix
- [x] Results: `results/plots/ber_vs_scrub_period.*`, `breaking_point.*`, `error_accumulation.*`
- [x] `results/verification_matrix.csv` — all 8 requirement rows (REQ-01 through REQ-07)
- [x] Dead-code bug removed from `inject_bit_flips` (`nand_flash_model.py`)
- [x] Duplicate `results/` at repo root removed
- [x] Unit test suite: 90 tests across 8 files (`test_bch`, `test_config`, `test_nand_flash_model`, `test_radiation_model`, `test_scrubbing`, `test_fault_injection`, `test_analytical`, `test_bad_block_manager`)
- [x] galois DeprecationWarning suppressed; galois pinned to `>=0.4.10,<0.5`
- [x] BER units corrected to `/bit/s` throughout (was dimensionless)
- [x] **Bad Block Management simulation (REQ-06 BBM)** — `BadBlockManager` class added; factory bad blocks (Binomial, 2%) + runtime bad blocks from scrub events; spare pool (3% = 983 blocks/chip); wired into Monte Carlo; verification matrix REQ-06 BBM row shows PASS
- [x] **REQ-02 resolved** — scrub period increased to 72 h (10.6% of 20 Mbps, within 20% allocation); breaking-point analysis reworked to sweep scrub period at mission SEU rate instead of dimensionless BER sweep

---

## Remaining

### Potential / undecided

- [ ] **SEL simulation**
  REQ-07 requires simulation of both SEU and SEL faults. SEL is a hardware event
  (latch-up causes power cycle of the affected bank, losing unwritten data). A minimal
  model would: randomly trigger a bank power-cycle event at the SEL rate, reset that
  bank's stored data, and log the data loss event. This would satisfy REQ-07 more
  completely than SEU alone.
