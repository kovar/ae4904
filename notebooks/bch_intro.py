import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import galois
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    # BCH(15, 7) — the smallest standard BCH code with t=2.
    #   n = 15   total bits per codeword
    #   k = 7    data (message) bits
    #   t = 2    bit errors correctable per codeword
    # Small enough to display every single bit in the visualisations.
    bch = galois.BCH(n=15, k=7)
    GF2 = galois.GF2
    return GF2, bch, galois, mcolors, mpatches, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # BCH Error-Correcting Codes
    ## An interactive introduction for spacecraft EDAC

    ---

    ### Why do we need error correction in space?

    Spacecraft memories — SRAM, Flash, DRAM — store data as electrical charges in tiny
    transistors. In space, high-energy particles (protons, heavy ions, cosmic rays) pass
    through these transistors and can deposit enough charge to flip a bit from 0 to 1 or
    vice versa. This is called a
    [**Single-Event Upset (SEU)**](https://en.wikipedia.org/wiki/Single-event_upset).

    SEU rates depend on the orbit and solar activity, but a typical LEO spacecraft might
    experience one bit flip per megabyte of unprotected memory every few days. GEO and
    deep-space missions can see rates orders of magnitude higher. Without protection,
    data silently corrupts — a critical failure mode for any mission-critical system.

    The solution is **Error Detection and Correction (EDAC)**: add structured redundancy
    to stored data so that bit flips can be identified and reversed.

    ---

    ### What are BCH codes?

    [**BCH codes**](https://en.wikipedia.org/wiki/BCH_code) (named after Bose,
    Chaudhuri, and Hocquenghem, independently discovered 1959–60) are a large family of
    binary block codes with a mathematically guaranteed error-correction capability.
    They are:

    - **Systematic**: the original data bits appear unchanged inside the codeword
    - **Cyclic**: a circular shift of any codeword is also a valid codeword (simplifies hardware)
    - **Flexible**: you can choose the correction power $t$ to match your reliability requirement

    BCH codes are widely used in spacecraft memories, NAND flash controllers, DVB
    broadcast, and QR codes.

    Work through the sections below — every slider and control is **live**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1  Code Parameters
    """)
    return


@app.cell
def _(bch, galois, mo):
    _rows = []
    for _t, _d in [(1, 3), (2, 5), (3, 7)]:
        try:
            _b = galois.BCH(n=15, d=_d)
            _rows.append({
                "t  (errors corrected)": _t,
                "k  (data bits)": _b.k,
                "r = n − k  (parity bits)": 15 - _b.k,
                "Rate  R = k/n": f"{_b.k / 15:.3f}",
                "This notebook": "◀" if _b.k == bch.k else "",
            })
        except Exception:
            pass

    mo.vstack([
        mo.md(rf"""
        Every BCH code is characterised by three integers:

        | Symbol | Name | Meaning |
        |--------|------|---------|
        | $n$ | Block length | Total bits per codeword |
        | $k$ | Message length | Useful data bits inside each codeword |
        | $t$ | Correction power | Number of bit errors **guaranteed** correctable |
        | $r = n - k$ | Redundancy | Extra "parity" bits added by the encoder |

        The **code rate** $R = k/n$ measures efficiency: what fraction of each codeword
        carries real data. A higher $t$ requires more parity bits and lowers the rate.

        This notebook uses **BCH({bch.n}, {bch.k})**: correction power $t = {bch.t}$,
        redundancy $r = {bch.n - bch.k}$ bits
        ({(bch.n - bch.k) / bch.n * 100:.0f}% overhead, {bch.k / bch.n * 100:.0f}% efficiency).
        This is deliberately small — every bit is visible in the diagrams below.
        Real-world codes use much longer blocks (e.g. BCH(511, 493) for NAND Flash)
        but the same principles apply.

        The table shows the full BCH(15, ·) family.
        """),
        mo.ui.table(_rows, label="BCH(15, ·) code family"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2  Codeword Layout
    """)
    return


@app.cell
def _(bch, mo, plt):
    _r = bch.n - bch.k
    _fig, _ax = plt.subplots(figsize=(10, 1.6))
    _ax.set_xlim(0, bch.n)
    _ax.set_ylim(0, 1)
    _ax.axis("off")

    _ax.barh(0.5, bch.k, left=0, height=0.55, color="#2176ae", align="center")
    _ax.text(bch.k / 2, 0.5, f"Data  ({bch.k} bits)",
             ha="center", va="center", color="white", fontweight="bold", fontsize=13)

    _ax.barh(0.5, _r, left=bch.k, height=0.55, color="#e07b39", align="center")
    _ax.text(bch.k + _r / 2, 0.5, f"Parity ({_r} bits)",
             ha="center", va="center", color="white", fontweight="bold", fontsize=13)

    _ax.set_title(f"BCH({bch.n}, {bch.k}) codeword  ←  {bch.n} bits total  →",
                  pad=8, fontsize=12)
    plt.tight_layout()

    mo.vstack([
        mo.md(rf"""
        BCH is a **systematic** code, meaning the encoder does not scramble the
        original data. Instead it appends $r = {_r}$ **parity bits** after the
        $k = {bch.k}$ data bits, producing a {bch.n}-bit **codeword**.

        When the codeword is later read back, the decoder checks the parity, locates
        any flipped bits (up to $t = {bch.t}$), corrects them, then strips the parity
        and returns the original {bch.k} data bits. From the application's perspective
        it just reads and writes {bch.k}-bit words; the EDAC layer is transparent.

        The {_r}-bit overhead is only {_r}/{bch.n} ≈ {_r/bch.n*100:.0f}% of the stored
        block — a small price for guaranteed 2-bit correction.
        """),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3  The Generator Polynomial
    """)
    return


@app.cell
def _(bch, mo, np, plt):
    _coeffs = np.array(bch.generator_poly.coeffs, dtype=int)
    _deg    = len(_coeffs) - 1
    _x      = np.arange(_deg + 1)

    _fig, _ax = plt.subplots(figsize=(10, 2.6))
    _cols = ["#2176ae" if c else "#ddd" for c in _coeffs]
    _ax.bar(_x, _coeffs, color=_cols, width=0.7)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([rf"$x^{{{_deg - i}}}$" for i in _x], fontsize=8)
    _ax.set_yticks([0, 1])
    _ax.set_ylabel("Coefficient")
    _ax.set_title(
        rf"Generator polynomial $g(x)$,  degree {_deg}  "
        rf"({int(np.sum(_coeffs))} nonzero terms, {_deg - int(np.sum(_coeffs)) + 1} zero terms)"
    )
    plt.tight_layout()

    mo.vstack([
        mo.md(rf"""
        BCH codes are built on **polynomial arithmetic over GF(2)** — the
        [Galois field](https://en.wikipedia.org/wiki/GF(2)) with only two elements,
        0 and 1, where addition is XOR and multiplication is AND.
        Every bit string of length $n$ can be treated as the coefficients of a
        polynomial: e.g. `101` → $x^2 + 1$.

        > **Note for students:** GF(2) polynomial arithmetic is *not* typically covered
        > in standard university mathematics or engineering courses. It comes from
        > abstract algebra — specifically the theory of
        > [finite fields](https://en.wikipedia.org/wiki/Finite_field) — which is usually
        > only taught in dedicated coding theory or cryptography electives. If this feels
        > unfamiliar, that is completely normal. You do not need to master it to use BCH
        > codes; the `galois` library handles all of it. The intuition is what matters:
        > treat each bit string as a polynomial, and define arithmetic rules so that
        > coefficients stay in {0, 1}.

        The key object is the **generator polynomial** $g(x)$. Every valid codeword
        $c(x)$ must be divisible by $g(x)$ with no remainder (over GF(2)). The encoder
        enforces this by computing:

        $$c(x) = m(x)\cdot x^r + \bigl[m(x)\cdot x^r \bmod g(x)\bigr]$$

        The first term shifts the message to the high bits; the subtracted remainder
        (which equals addition in GF(2)) fills the low $r$ bits with the parity,
        making the whole codeword exactly divisible by $g(x)$.

        $g(x)$ has degree $r = {_deg}$, which is why the parity region is exactly
        {_deg} bits wide. Blue bars mark the nonzero (1) coefficients.

        You don't need to understand the polynomial maths to use BCH codes — the
        `galois` library handles it all — but it explains *why* the decoder can
        correct errors: a received word with errors will no longer divide evenly by
        $g(x)$, and the remainder (the **syndrome**) tells the decoder where errors are.
        """),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4  Encoding a Message
    """)
    return


@app.cell
def _(mo):
    seed_slider = mo.ui.slider(0, 99, value=7, label="Message seed", show_value=True)
    seed_slider
    return (seed_slider,)


@app.cell
def _(GF2, bch, mcolors, mo, np, plt, seed_slider):
    _rng    = np.random.default_rng(seed_slider.value)
    msg_bits = _rng.integers(0, 2, size=bch.k, dtype=int)
    cw_bits  = np.array(bch.encode(GF2(msg_bits)), dtype=int)

    # ── Visualise: two rows — message and codeword ────────────────────────────
    _fig, _axes = plt.subplots(2, 1, figsize=(11, 2.6))

    for _row, (_bits, _n_show, _title) in enumerate([
        (msg_bits,  bch.k,  f"Message  ({bch.k} bits)"),
        (cw_bits,   bch.n,  f"Codeword  ({bch.n} bits = data + parity)"),
    ]):
        _rgb = np.zeros((_n_show, 3))
        for _i in range(_n_show):
            if _row == 1 and _i >= bch.k:
                _rgb[_i] = mcolors.to_rgb("#e07b39")   # parity
            else:
                _rgb[_i] = mcolors.to_rgb("#2176ae")   # data

        _axes[_row].imshow(
            _rgb[np.newaxis, :, :], aspect="auto",
            extent=[-0.5, _n_show - 0.5, -0.5, 0.5],
        )
        for _i in range(_n_show):
            _axes[_row].text(
                _i, 0, str(_bits[_i]),
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )
        _axes[_row].set_xlim(-0.5, _n_show - 0.5)
        _axes[_row].set_yticks([])
        _axes[_row].set_title(_title, fontsize=10)
        if _row == 1:
            _axes[_row].axvline(bch.k - 0.5, color="white", linewidth=1.5, linestyle="--")

    plt.tight_layout()

    _parity_str = "".join(str(b) for b in cw_bits[bch.k:])
    mo.vstack([
        mo.md(f"""
        The encoder takes the {bch.k}-bit message (top row, blue) and produces the
        {bch.n}-bit codeword (bottom row). The first {bch.k} bits are identical to
        the message (systematic property). The dashed line separates data from the
        {bch.n - bch.k} appended parity bits (orange): `{_parity_str}`.

        Change the seed slider to try a different random message.
        """),
        _fig,
    ])
    return cw_bits, msg_bits


@app.cell
def _(mo):
    mo.md(r"""
    ## 5  Single-Event Upsets and Decoding

    An SEU flips one or more bits somewhere in the {n}-bit codeword while it sits
    in memory. The decoder reads the corrupted word, computes the syndrome, and
    corrects up to $t$ flips before returning the data.

    Use the slider to control how many bits are flipped, and watch whether the
    decoder succeeds.
    """)
    return


@app.cell
def _(bch, mo):
    err_slider = mo.ui.slider(
        0, bch.t + 2, value=1,
        label=f"Bit errors to inject  (t = {bch.t})",
        show_value=True,
    )
    err_slider
    return (err_slider,)


@app.cell
def _(
    GF2,
    bch,
    cw_bits,
    err_slider,
    mcolors,
    mo,
    mpatches,
    msg_bits,
    np,
    plt,
    seed_slider,
):
    _rng2   = np.random.default_rng(seed_slider.value + 200)
    _ne     = err_slider.value

    rx_bits = cw_bits.copy()
    err_pos = np.array([], dtype=int)
    if _ne > 0:
        err_pos = _rng2.choice(len(rx_bits), size=_ne, replace=False)
        rx_bits[err_pos] ^= 1

    dec_bits = np.array(bch.decode(GF2(rx_bits)), dtype=int)
    residual = int(np.sum(dec_bits != msg_bits))

    # ── Status callout ────────────────────────────────────────────────────────
    if _ne == 0:
        _kind = "success"
        _msg  = "No errors injected. The codeword is intact and decodes perfectly."
    elif residual == 0:
        _kind = "success"
        _msg  = (f"**{_ne} error(s) injected — all corrected.** "
                 f"The decoder located and flipped back every bit. "
                 f"Decoded message matches original exactly.")
    else:
        _kind = "danger"
        _msg  = (f"**{_ne} errors exceed the correction limit (t = {bch.t}).** "
                 f"The decoder made its best guess but {residual} bit(s) remain wrong — "
                 f"a **silent miscorrection**. The application would receive wrong data "
                 f"with no indication of failure.")

    # ── Three-row bit visualisation: transmitted / received / decoded ─────────
    _err_set = set(err_pos.tolist())
    _n       = bch.n

    def _row_image(bits, n, highlight_errors=False, is_decoded=False):
        _rgb = np.zeros((n, 3))
        for i in range(n):
            if highlight_errors and i in _err_set:
                _rgb[i] = mcolors.to_rgb("crimson")
            elif not is_decoded and i >= bch.k:
                _rgb[i] = mcolors.to_rgb("#e07b39")    # parity
            else:
                _rgb[i] = mcolors.to_rgb("#2176ae")    # data
        return _rgb

    _rows_data = [
        (cw_bits,  _n,     False, False, f"Transmitted codeword  ({_n} bits)"),
        (rx_bits,  _n,     True,  False, f"Received (after {_ne} SEU flip(s))"),
        (dec_bits, bch.k,  False, True,  f"Decoded message  ({bch.k} bits)"),
    ]

    _fig, _axes = plt.subplots(3, 1, figsize=(12, 4.0))

    for _ax, (bits, _n_show, _do_err, _is_dec, _title) in zip(_axes, _rows_data):
        _rgb = _row_image(bits, _n_show, _do_err, _is_dec)
        _ax.imshow(
            _rgb[np.newaxis, :, :], aspect="auto",
            extent=[-0.5, _n_show - 0.5, -0.5, 0.5],
        )
        for _i in range(_n_show):
            _ax.text(_i, 0, str(bits[_i]),
                     ha="center", va="center", fontsize=7,
                     color="white", fontweight="bold")
        _ax.set_xlim(-0.5, _n_show - 0.5)
        _ax.set_yticks([])
        _ax.set_title(_title, fontsize=10)
        if not _is_dec:
            _ax.axvline(bch.k - 0.5, color="white", linewidth=1.2, linestyle="--")

    _patches = [
        mpatches.Patch(color="#2176ae", label="Data bit"),
        mpatches.Patch(color="#e07b39", label="Parity bit"),
        mpatches.Patch(color="crimson",  label="Flipped bit (SEU)"),
    ]
    _axes[-1].legend(handles=_patches, loc="upper right", fontsize=8,
                     bbox_to_anchor=(1.0, -0.25), ncol=3)
    plt.tight_layout()

    mo.vstack([
        mo.callout(mo.md(_msg), kind=_kind),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6  The Syndrome — Error Fingerprint

    How does the decoder know *where* errors occurred? It computes the **syndrome**
    — the remainder when the received polynomial is divided by the generator polynomial:

    $$s(x) = r(x) \bmod g(x)$$

    Think of it like a checksum that is zero if and only if the received word is a
    valid codeword:

    - **$s(x) = 0$** → no errors detected (word is valid)
    - **$s(x) \neq 0$** → errors are present; the exact bit pattern of $s(x)$ encodes
      information about *where* the errors are

    The decoder then uses the syndrome as input to the
    [Berlekamp–Massey algorithm](https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm),
    which finds an "error-locator polynomial", followed by a
    [Chien search](https://en.wikipedia.org/wiki/Chien_search) to evaluate it at
    every bit position and identify the flips.

    The heatmap below shows the syndrome coefficients (each is 0 or 1) for increasing
    numbers of injected errors. The seed slider in section 4 selects the message.
    """)
    return


@app.cell
def _(GF2, bch, galois, mo, np, plt, seed_slider):
    _rng3 = np.random.default_rng(seed_slider.value + 300)
    _msg3 = GF2(_rng3.integers(0, 2, size=bch.k, dtype=int))
    _cw3  = bch.encode(_msg3)
    _r    = bch.n - bch.k

    _syndromes = []
    _labels    = []
    for _e3 in range(6):
        _rx3 = np.array(_cw3, dtype=int)
        if _e3 > 0:
            _p3 = _rng3.choice(bch.n, size=_e3, replace=False)
            _rx3[_p3] ^= 1
        _poly = galois.Poly(GF2(_rx3))
        _syn  = _poly % bch.generator_poly
        _arr  = np.array(_syn.coeffs, dtype=float)
        if len(_arr) < _r:
            _arr = np.pad(_arr, (_r - len(_arr), 0))
        _syndromes.append(_arr)
        _labels.append(f"{_e3} error(s)" + (" ← zero" if _e3 == 0 else ""))

    _mat = np.array(_syndromes)

    _fig, _ax = plt.subplots(figsize=(10, 3.4))
    _im = _ax.imshow(_mat, aspect="auto", cmap="Blues",
                     interpolation="nearest", vmin=0, vmax=1)
    _ax.set_yticks(range(6))
    _ax.set_yticklabels(_labels, fontsize=9)
    _ax.set_xlabel("Syndrome coefficient index  (0 = highest-degree term)")
    _ax.set_title(r"Syndrome $s(x) = r(x)\,\mathrm{mod}\,g(x)$  coefficients vs number of injected errors")
    plt.colorbar(_im, ax=_ax, shrink=0.8, label="bit value")
    plt.tight_layout()

    mo.vstack([
        mo.md(f"""
        Row 0 (no errors) is all-zero — the received word divides evenly by $g(x)$.
        Each additional error scrambles the syndrome differently.
        The syndrome is {_r} bits wide, matching the degree of $g(x)$ and the number of parity bits.

        Notice that each error count produces a **distinct** syndrome pattern — this
        is what allows the decoder to distinguish between different error configurations
        and pinpoint the exact bit positions that need correcting.
        """),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7  Minimum Hamming Distance

    The [**Hamming distance**](https://en.wikipedia.org/wiki/Hamming_distance) between
    two bit strings is simply the number of positions where they differ. For example:

    ```
    10110010
    11010110
     ↑↑  ↑    ← 3 positions differ → Hamming distance = 3
    ```

    The **minimum Hamming distance** $d_{\min}$ of a code is the smallest Hamming
    distance between any two *distinct valid codewords*. This single number controls
    the code's error correction ability:

    $$d_{\min} \geq 2t + 1$$

    **Intuition:** imagine each codeword as a point in a high-dimensional space, with
    distance measured in bit flips. $d_{\min}$ is the closest any two codewords ever
    get. If a codeword suffers $\leq t$ flips, it moves at most $t$ steps from its
    original position — but the nearest *other* codeword is at least $2t+1$ steps
    away. So the corrupted word is still closer to its original codeword than to any
    other, and the decoder can always identify the right one.

    Once more than $t$ bits flip, the corrupted word may be closer to a *different*
    valid codeword, and the decoder corrects it to the wrong one — a **silent
    miscorrection** with no error flag raised.
    """)
    return


@app.cell
def _(GF2, bch, mcolors, mo, np, plt):
    # Encode two different messages and show their Hamming distance
    _rng4 = np.random.default_rng(17)
    _ma   = GF2(_rng4.integers(0, 2, size=bch.k, dtype=int))
    _mb   = GF2(_rng4.integers(0, 2, size=bch.k, dtype=int))
    _ca   = np.array(bch.encode(_ma), dtype=int)
    _cb   = np.array(bch.encode(_mb), dtype=int)
    _diff = (_ca != _cb).astype(int)
    _hd   = int(_diff.sum())

    _fig, _axes = plt.subplots(3, 1, figsize=(11, 4.2), sharex=True)
    for _row, (_bits, _label, _col) in enumerate([
        (_ca,   "Valid codeword A",  "#2176ae"),
        (_cb,   "Valid codeword B",  "#e07b39"),
        (_diff, f"Bit positions that differ  →  Hamming distance = {_hd}", "crimson"),
    ]):
        _rgb = np.zeros((bch.n, 3))
        for _i in range(bch.n):
            if _bits[_i]:
                _rgb[_i] = mcolors.to_rgb(_col)
            else:
                _rgb[_i] = mcolors.to_rgb("#dddddd")
        _axes[_row].imshow(
            _rgb[np.newaxis, :, :], aspect="auto",
            extent=[-0.5, bch.n - 0.5, -0.5, 0.5],
        )
        for _i in range(bch.n):
            _axes[_row].text(
                _i, 0, str(_bits[_i]),
                ha="center", va="center", fontsize=7,
                color="white" if _bits[_i] else "#888",
                fontweight="bold",
            )
        _axes[_row].set_xlim(-0.5, bch.n - 0.5)
        _axes[_row].set_yticks([])
        _axes[_row].set_ylabel(_label, fontsize=9, rotation=0,
                               ha="right", labelpad=4)

    _axes[-1].set_xlabel("Bit position")
    _fig.suptitle(
        rf"Two valid BCH({bch.n},{bch.k}) codewords — "
        rf"Hamming distance = {_hd} ≥ $d_{{\min}}$ = {bch.d} = 2×{bch.t}+1",
        fontsize=11,
    )
    plt.tight_layout()

    mo.vstack([
        mo.md(rf"""
        The two rows above are **actual BCH({bch.n},{bch.k}) codewords** encoding
        different messages. The red row highlights the {_hd} bit positions where
        they disagree — their Hamming distance.

        Since $d_{{\min}} = {bch.d}$, no two valid codewords are ever closer than
        {bch.d} bit flips apart. A received word with $\leq {bch.t}$ flips stays
        in the "correction ball" of its original codeword and is always recoverable.
        """),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8  Correction Success Rate (Monte-Carlo Simulation)

    The guarantee $t = 2$ means BCH(15, 7) *always* corrects up to 2 errors per
    codeword. But what happens with 3, 4, or more errors? The decoder does not
    know it has been overwhelmed — it still returns *something*, which may or may
    not be correct.

    The simulation below runs many random codewords, injects a fixed number of
    errors, decodes, and checks whether the result matches the original.
    Adjust the trial count to reduce statistical noise.
    """)
    return


@app.cell
def _(mo):
    trials_slider = mo.ui.slider(
        100, 2000, value=500, step=100,
        label="Trials per error count", show_value=True,
    )
    trials_slider
    return (trials_slider,)


@app.cell
def _(GF2, bch, mo, mpatches, np, plt, trials_slider):
    _rng5  = np.random.default_rng(42)
    _N     = trials_slider.value
    _max_e = bch.t + 4
    _rates = []

    for _e5 in range(_max_e + 1):
        _ok5 = 0
        for _ in range(_N):
            _m5  = GF2(_rng5.integers(0, 2, size=bch.k, dtype=int))
            _cw5 = np.array(bch.encode(_m5), dtype=int)
            if _e5 > 0:
                _cw5[_rng5.choice(bch.n, size=_e5, replace=False)] ^= 1
            _d5  = np.array(bch.decode(GF2(_cw5)), dtype=int)
            if np.all(_d5 == np.array(_m5, dtype=int)):
                _ok5 += 1
        _rates.append(_ok5 / _N)

    _xe   = np.arange(_max_e + 1)
    _bcol = ["#2176ae" if _e <= bch.t else "crimson" for _e in _xe]

    _fig5, _ax5 = plt.subplots(figsize=(9, 4.5))
    _bars = _ax5.bar(_xe, _rates, color=_bcol, width=0.6, zorder=3)
    _ax5.axvline(bch.t + 0.5, color="black", linestyle="--", linewidth=1.8,
                 label=f"Correction limit  t = {bch.t}")
    _ax5.set_xlabel("Number of errors injected per codeword", fontsize=11)
    _ax5.set_ylabel("Fraction correctly decoded", fontsize=11)
    _ax5.set_ylim(0, 1.15)
    _ax5.set_xticks(_xe)
    _ax5.set_title(f"BCH({bch.n},{bch.k}) correction success rate  ({_N} trials each)")
    _ax5.grid(axis="y", alpha=0.4, zorder=0)

    for _bar, _rate in zip(_bars, _rates):
        _ax5.text(_bar.get_x() + _bar.get_width() / 2,
                  _rate + 0.025, f"{_rate:.2f}",
                  ha="center", va="bottom", fontsize=9)

    _p2 = [
        mpatches.Patch(color="#2176ae", label=f"≤ t = {bch.t}  (guaranteed 100%)"),
        mpatches.Patch(color="crimson",  label=f"> t = {bch.t}  (beyond guarantee)"),
    ]
    _ax5.legend(handles=_p2, fontsize=9)
    plt.tight_layout()

    mo.vstack([
        mo.md(f"""
        **Blue bars** ($\\leq t = {bch.t}$ errors) should read exactly **1.00** —
        BCH guarantees correction, confirmed by simulation.

        **Red bars** (beyond $t$) show that the decoder sometimes gets lucky (more
        errors happen to land in a correctable pattern), but the success rate falls
        rapidly. Critically, failures are **silent** — the decoder returns wrong data
        with no warning. This is why the correction limit is a hard design boundary,
        not a soft guideline.
        """),
        _fig5,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9  Python Implementation: the `galois` Library

    The [`galois`](https://mhostetter.github.io/galois/latest/) library provides
    finite-field arithmetic and block codes for Python, built on NumPy. It handles
    all the polynomial algebra over GF(2) so you don't have to.
    """)
    return


@app.cell
def _(GF2, bch, galois, mo, np):
    _bch_d  = galois.BCH(n=15, d=5)   # construct via minimum distance
    _gpoly  = bch.generator_poly
    _gdeg   = _gpoly.degree

    # Live round-trip to verify
    _rng9  = np.random.default_rng(1)
    _msg9  = GF2(_rng9.integers(0, 2, size=bch.k, dtype=int))
    _cw9   = bch.encode(_msg9)
    _rx9   = np.array(_cw9, dtype=int)
    _rx9[3] ^= 1           # inject one bit flip
    _rx9[11] ^= 1          # inject another
    _dec9  = bch.decode(GF2(_rx9))
    _ok9   = bool(np.all(np.array(_dec9, dtype=int) == np.array(_msg9, dtype=int)))

    mo.md(f"""
    ```python
    import galois
    import numpy as np

    # ── 1. Construct the BCH code ────────────────────────────────────────────
    bch = galois.BCH(n=15, k=7)     # specify n and k directly; t = {bch.t}
    bch = galois.BCH(n=15, d=5)     # or specify n and minimum distance d = 2t+1
    #  → bch.n = {bch.n},  bch.k = {bch.k},  bch.t = {bch.t},  bch.d = {bch.d}

    # ── 2. Create GF(2) arrays (required by encode/decode) ──────────────────
    GF2 = galois.GF2
    msg = GF2(np.random.randint(0, 2, size=bch.k))   # shape ({bch.k},)

    # ── 3. Encode ────────────────────────────────────────────────────────────
    codeword = bch.encode(msg)       # GF2 array, shape ({bch.n},)

    # ── 4. Simulate errors: convert to plain numpy, flip bits, convert back ──
    rx = np.array(codeword, dtype=int)
    rx[3]  ^= 1                      # XOR with 1 flips the bit
    rx[11] ^= 1
    rx_gf = GF2(rx)

    # ── 5. Decode ────────────────────────────────────────────────────────────
    decoded = bch.decode(rx_gf)      # GF2 array, shape ({bch.k},)

    # ⚠ IMPORTANT: decode() NEVER raises an exception, even if errors > t.
    # It silently miscorrects. In simulation, always check against ground truth:
    ok = bool(np.all(np.array(decoded, dtype=int) == np.array(msg, dtype=int)))

    # ── 6. Code properties ───────────────────────────────────────────────────
    bch.n              # {bch.n}   — codeword length
    bch.k              # {bch.k}   — message length
    bch.t              # {bch.t}   — correction power
    bch.d              # {bch.d}   — minimum Hamming distance (= 2t+1)
    bch.generator_poly # degree-{_gdeg} polynomial over GF(2)

    # ── 7. Manual syndrome ───────────────────────────────────────────────────
    rx_poly  = galois.Poly(GF2(rx))
    syndrome = rx_poly % bch.generator_poly   # all-zero ↔ valid codeword
    ```

    Live verification (2 errors injected, decoded vs original): `{_ok9}`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10  Spacecraft EDAC in Practice

    ### The radiation environment

    Spacecraft operate in harsh radiation environments: trapped protons and electrons
    in the Van Allen belts, galactic cosmic rays, and solar energetic particle events.
    High-energy particles that pass through memory cells deposit charge via ionisation,
    which can change the stored logic state — a Single-Event Upset.

    The SEU rate depends on:
    - **Orbit**: LEO inside the South Atlantic Anomaly has high trapped proton flux;
      GEO and interplanetary missions face more cosmic rays
    - **Shielding**: aluminium enclosures attenuate low-energy particles but are
      penetrated by GeV-range cosmic rays
    - **Technology node**: smaller transistors store less charge and are more sensitive

    Standards such as
    [ECSS-E-ST-10-12C](https://ecss.nl/standard/ecss-e-st-10-12c-methods-for-the-calculation-of-radiation-received-and-its-effects-15-june-2008/)
    define methods for quantifying the radiation environment and predicting SEU rates
    for a given mission.

    ### Memory scrubbing

    Even with EDAC, errors accumulate over time. A second SEU in an already-corrected
    word upgrades a 1-error event (correctable) to a 2-error event (still correctable
    for t=2), but further upsets may exceed the limit. **Scrubbing** — periodically
    reading every memory word, correcting errors, and writing the corrected value back
    — resets the error count before it can build up. Scrub periods range from seconds
    (high-radiation orbit) to hours (benign environment).

    ### Design trade-offs

    | Parameter | Typical range | Trade-off |
    |-----------|--------------|-----------|
    | Block size $n$ | 32 – 512 bits | Larger → lower header overhead; but one SEU burst hits more data |
    | Correction power $t$ | 1 – 4 | Higher → more reliable; but more parity bits and decoder complexity |
    | Scrubbing period | 1 s – 24 h | Faster → fewer accumulated errors; but more memory bus bandwidth |
    | Code rate $R = k/n$ | 0.85 – 0.98 | Lower $R$ → higher overhead; tighter for mass/power-constrained designs |

    ### BCH vs other codes

    | Code | Strengths | Weaknesses | Typical use |
    |------|-----------|------------|-------------|
    | [**BCH**](https://en.wikipedia.org/wiki/BCH_code) | Flexible $t$, proven, efficient decoder | Bit-oriented only | Space memory, NAND flash |
    | [Hamming / SECDED](https://en.wikipedia.org/wiki/Hamming_code) | Tiny overhead, trivial hardware | $t = 1$ only | Cache SRAM, low-radiation LEO |
    | [Reed–Solomon](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) | Burst-error correction, byte-oriented | More complex | CDs, DSN telemetry, QR codes |
    | [LDPC](https://en.wikipedia.org/wiki/Low-density_parity-check_code) | Near-Shannon capacity | Iterative decoder, long latency | Deep-space data links |
    | [Turbo codes](https://en.wikipedia.org/wiki/Turbo_code) | Very high performance | Complex | 3G/4G, some ESA missions |

    For protecting individual memory words where a simple, deterministic decoder with
    bounded latency is required, **BCH is the industry standard**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11  Code Rate in Real NAND Flash — The OOB Area

    The textbook BCH model presents a clear trade-off: higher correction power $t$
    requires more parity bits, which lowers the code rate $R = k/n$ and reduces the
    fraction of each block that carries useful data. In practice, NAND Flash hardware
    sidesteps this trade-off almost entirely.

    ### The physical page layout

    Every NAND Flash page is divided into two physically distinct regions:

    ```
    ┌─────────────────────────────────────────┬──────────────────┐
    │           Main area                     │   OOB / Spare    │
    │   (e.g. 8192 bytes — your data)         │  (e.g. 448 bytes)│
    └─────────────────────────────────────────┴──────────────────┘
         ← advertised storage capacity →          "free" bonus
    ```

    The **Out-Of-Band (OOB)** region (also called the *spare area*) is not counted
    in the chip's advertised capacity. It is a physically separate set of cells that
    the manufacturer includes specifically for ECC, metadata, and wear-leveling
    bookkeeping. From the user's perspective, storing ECC parity in the OOB costs
    nothing — the code rate for user data is effectively **R ≈ 1**.

    ### What goes in the OOB?

    The OOB is shared between several consumers:

    | OOB content | Written by |
    |-------------|-----------|
    | ECC parity bits | Your BCH encoder |
    | Bad block marker | Factory test / wear-leveling layer |
    | Filesystem metadata | JFFS2, UBIFS, YAFFS, … |
    | Sector sequence numbers / timestamps | Flash Translation Layer (FTL) |

    You do **not** need to fill the entire OOB with ECC parity. You use only as many
    bytes as your chosen BCH code requires; the rest are available for other purposes.

    ### Checking that your ECC fits

    The datasheet specifies two things you must verify:

    1. **Minimum required ECC strength** — the manufacturer's worst-case raw bit error
       rate implies a minimum $t$ you must meet (e.g. "4-bit correction per 512 bytes").
       Choose a code at least this strong.

    2. **OOB size** — your parity bits must physically fit. For BCH(511, 493) the
       parity is $r = 18$ bits = 2.25 bytes per 511-bit sector. Even protecting many
       sectors per page, this is typically a small fraction of the available OOB.

    ### Worked example

    Suppose a page has an 8192-byte main area and a 448-byte OOB. If the main area is
    divided into sixteen 512-byte sectors, each protected by BCH(511, 493):

    | Quantity | Value |
    |----------|-------|
    | Sectors per page | 16 |
    | Parity per sector | 18 bits = 3 bytes (rounded up) |
    | Total ECC bytes | 16 × 3 = 48 bytes |
    | OOB used for ECC | 48 / 448 = 11% |
    | OOB remaining | 400 bytes — available for metadata |

    The ECC consumes only a small slice of the OOB; the rest is free. And because
    the OOB sits outside the main area entirely, the user-visible storage efficiency
    remains 100% — not the 81% that a naïve $R = 51/63 = 0.81$ figure would suggest.

    ### Why the code rate still matters

    Even though the OOB makes the storage-efficiency argument moot, the code rate
    $R = k/n$ still matters in two ways:

    - **Decoder latency and power**: more parity bits means more syndrome computation
      and a longer Chien search, which affects read latency and energy per bit.
    - **OOB budget**: if you choose a very strong code (large $t$, many parity bits)
      for very large sectors, the parity may eventually overflow the OOB, forcing you
      to choose smaller sectors or a weaker code.

    For typical spacecraft memory protection with $t = 2$ or $t = 4$, neither
    constraint is binding and BCH fits comfortably within the OOB.
    """)
    return


@app.cell
def _(bch, mo):
    mo.md(rf"""
    ## Summary

    ### How BCH(15, 7) works end-to-end

    | Step | Operation | Detail |
    |------|-----------|--------|
    | **Write** | Encode | {bch.k}-bit message → {bch.n}-bit codeword (append {bch.n - bch.k} parity bits) |
    | **Store** | Memory | Codeword sits in SRAM/Flash; SEUs may flip bits |
    | **Read** | Receive | {bch.n}-bit (possibly corrupted) word retrieved |
    | **Check** | Syndrome | $s(x) = r(x) \bmod g(x)$  — zero = clean, nonzero = errors |
    | **Locate** | Berlekamp–Massey + Chien | Find which bit positions were flipped |
    | **Fix** | Correct | Flip the located bits back |
    | **Return** | Strip parity | Return first {bch.k} bits to the application |

    ### Key numbers for BCH({bch.n}, {bch.k})

    $$n = {bch.n},\quad k = {bch.k},\quad t = {bch.t},\quad
      d_{{\min}} = {bch.d},\quad r = {bch.n - bch.k},\quad R = {bch.k/bch.n:.3f}$$

    ### Further reading

    - [BCH code — Wikipedia](https://en.wikipedia.org/wiki/BCH_code)
    - [Single-event upset — Wikipedia](https://en.wikipedia.org/wiki/Single-event_upset)
    - [Hamming distance — Wikipedia](https://en.wikipedia.org/wiki/Hamming_distance)
    - [Error correction code — Wikipedia](https://en.wikipedia.org/wiki/Error_correction_code)
    - [galois library documentation](https://mhostetter.github.io/galois/latest/)
    - [ECSS-E-ST-10-12C — Radiation methods standard](https://ecss.nl/standard/ecss-e-st-10-12c-methods-for-the-calculation-of-radiation-received-and-its-effects-15-june-2008/)
    """)
    return


if __name__ == "__main__":
    app.run()
