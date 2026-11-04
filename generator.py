"""SN91 cascade generator — single native module (s1 of c313k105c303epi_fast).

One generator, no packed sources: the engine, its vendored chronoforge family
library and the partner family program are ordinary top-level code of this
module.  Output rows are byte-identical to the c313 exec-wrapper for every seed
and ``n``: the engine produces each row; with probability ``_Q`` (decided by a
per-row SeedSequence draw) the next partner-family row is stacked under it as a
second channel.

Name layout (to keep every former module's globals separate):
  * engine names are unchanged, except its public ``Generator`` which is
    ``_EngineGenerator`` here;
  * chronoforge names are unchanged unless they would collide, in which case
    they are prefixed ``_<cf_module>_``;
  * partner-family names are prefixed ``_pt_``; its config lives in
    ``config.json["partner"]``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

# ═══════════════ engine: runtime setup and imports ═══════════════
# Cascade V145 — native support integrity over the V144 persistence candidate.
#
# Experimental training-data generator. No measured forecasting gain is implied.


# H393: production-sandbox thread control before numerical imports.
# These must be set before the first numerical import to take effect.
import os as _h393_os
for _h393_var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS"):
    _h393_os.environ[_h393_var] = "1"
from threadpoolctl import threadpool_limits as _h393_threadpool_limits
_H393_THREAD_LIMITER = _h393_threadpool_limits(limits=1)
del _h393_var

import json
import os
from functools import lru_cache, partial
from queue import Full, Queue
from threading import Event, Lock, Thread, local
from typing import Any

from numba import njit
from scipy.signal import lfilter


# --- graft: chronoforge, inlined (logic unchanged) ---


# ═══════════════ chronoforge: cf_rng ═══════════════
# Deterministic RNG stream derivation.
#
# Every random number in the corpus comes from ``np.random.SeedSequence`` entropy
# built from four 32-bit words::
#
#     (seed_hi, seed_lo, batch_index, stage_id)
#
# ``batch_index`` is the unit of reproducibility: batch sizes follow a schedule
# that depends on the batch index alone (never on ``n_series``), so the corpus is
# prefix-stable by construction.  ``stage_id`` isolates the pipeline stages from
# each other, which is what makes weight sweeps *paired*: changing family ``f``'s
# weight leaves every other family's per-row parameter draws byte-identical.
#
# Nothing here reads the clock, the environment, ``os.urandom``, or a global RNG.


MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# ── stage identifiers ──────────────────────────────────────────────────────
STAGE_ASSIGN = 0        # length + cadence + family assignment
STAGE_CALENDAR = 1      # calendar bundle (start day-of-week / day-of-year)
STAGE_FAMILY_PARAM = 1000   # + family index: dense per-row parameter draws
STAGE_FAMILY_BULK = 5000    # + family index: group-sized innovation draws
STAGE_OBSERVE = 9000    # observation layer
STAGE_SCALE = 9001      # scale / offset
STAGE_AGGREGATE = 9002  # structured hierarchical aggregation
STAGE_SANITISE = 9003   # fallback regeneration + degeneracy repair
STAGE_COUNT = 9004      # count prior: nonnegative floor + level quantisation


def seed_words(seed: int) -> tuple[int, int]:
    """Split an arbitrary Python int seed into two 32-bit words."""
    s = int(seed) & MASK64
    return (s >> 32) & MASK32, s & MASK32


def stream(seed_hi: int, seed_lo: int, batch: int, stage: int) -> np.random.Generator:
    """Independent PCG64 stream for ``(seed, batch, stage)``."""
    ss = np.random.SeedSequence(
        [int(seed_hi) & MASK32, int(seed_lo) & MASK32,
         int(batch) & MASK32, int(stage) & MASK32]
    )
    return np.random.Generator(np.random.PCG64(ss))


def _self_test() -> None:
    """Fail loudly at import if stream derivation is not reproducible.

    We deliberately do not hard-code NumPy's internal bit-generator state (that
    would pin us to one NumPy build for no benefit).  What we *do* assert is the
    three properties the corpus relies on: same coordinates -> same draws,
    different coordinates -> different draws, and the derivation is insensitive
    to how the Python int seed is spelled.
    """
    a = stream(1234, 5678, 7, 1003).random(8)
    b = stream(1234, 5678, 7, 1003).random(8)
    if not np.array_equal(a, b):
        raise RuntimeError("chronoforge: RNG stream derivation is not reproducible")
    for coords in ((1234, 5678, 8, 1003), (1234, 5678, 7, 1004),
                   (1235, 5678, 7, 1003), (1234, 5679, 7, 1003)):
        if np.array_equal(a, stream(*coords).random(8)):
            raise RuntimeError("chronoforge: RNG streams collide across coordinates")
    hi, lo = seed_words(-1)
    if (hi, lo) != (MASK32, MASK32):
        raise RuntimeError("chronoforge: seed word split is wrong")
    hi, lo = seed_words((1 << 63) + 12345)
    if (hi << 32 | lo) != ((1 << 63) + 12345):
        raise RuntimeError("chronoforge: seed word split loses bits")
    # zero seed must still produce a usable stream
    if not np.isfinite(stream(0, 0, 0, 0).random(4)).all():
        raise RuntimeError("chronoforge: zero seed produced non-finite draws")


_self_test()


# ═══════════════ chronoforge: cf_spectral ═══════════════
# FFT-based Gaussian-process synthesis.
#
# Every stationary GP in the corpus is drawn by sampling complex white noise in
# the frequency domain, shaping it by ``sqrt(S(f))`` and taking one batched
# inverse real FFT on a ``2L`` grid (the 2x embedding removes circular
# wrap-around; we crop back to ``L``).  Cost is O(L log L) per row and the whole
# batch goes through a single ``irfft`` call.
#
# No Cholesky, no ``multivariate_normal``, no BLAS GEMM: that is deliberate.  It
# is asymptotically cheaper than the O(L^3) factorisation route *and* it removes
# the threaded-BLAS reduction-order hazard, which matters because the producer is
# multi-threaded and the corpus digest must be byte-identical at any thread count.


try:
    from scipy.special import gammaincinv as _gammaincinv
    HAVE_SCIPY_SPECIAL = True
except Exception:  # pragma: no cover
    HAVE_SCIPY_SPECIAL = False
    _gammaincinv = None

TWO_PI = 2.0 * np.pi


@lru_cache(maxsize=16)
def freq_grid(L: int) -> np.ndarray:
    """rfft frequency grid (cycles/sample) for the 2L circulant embedding."""
    f = np.fft.rfftfreq(2 * L, d=1.0)
    f.flags.writeable = False
    return f


@lru_cache(maxsize=16)
def time_grid(L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)
    t.flags.writeable = False
    return t


# ───────────────────────────── PSD primitives ──────────────────────────────
# All return a (n, nf) array up to an arbitrary positive constant; the sampler
# normalises each realised row to unit standard deviation anyway, so only the
# *shape* of S(f) matters.

def psd_matern(f: np.ndarray, ell: np.ndarray, nu: float) -> np.ndarray:
    """Matern-nu spectral density in 1-D.

    The half-integer orders we actually use give integer exponents, so they are
    evaluated by reciprocal multiplication rather than ``np.power`` — the same
    numbers, several times cheaper on a (n, L+1) grid.
    """
    w = TWO_PI * f[None, :] * ell
    d = 1.0 + (w * w) / (2.0 * nu)
    if nu == 0.5:
        return 1.0 / d
    if nu == 1.5:
        return 1.0 / (d * d)
    if nu == 2.5:
        return 1.0 / (d * d * d)
    return d ** (-(nu + 0.5))


def psd_rbf(f: np.ndarray, ell: np.ndarray) -> np.ndarray:
    a = 2.0 * np.pi ** 2 * (ell ** 2) * (f[None, :] ** 2)
    return np.exp(-np.minimum(a, 700.0))


def psd_rq(f: np.ndarray, ell: np.ndarray, alpha: np.ndarray, J: int = 8) -> np.ndarray:
    """Rational-quadratic as a Gamma scale-mixture of RBF densities."""
    q = (np.arange(J, dtype=np.float64) + 0.5) / J
    if HAVE_SCIPY_SPECIAL:
        g = _gammaincinv(np.maximum(alpha, 1e-3), q[None, :])
    else:  # Wilson-Hilferty approximation of the Gamma quantile
        a = np.maximum(alpha, 1e-3)
        zq = np.sqrt(2.0) * _erfinv_approx(2.0 * q - 1.0)[None, :]
        g = a * (1.0 - 1.0 / (9.0 * a) + zq / np.sqrt(9.0 * a)) ** 3
    g = np.maximum(g, 1e-6)
    out = np.zeros((ell.shape[0], f.shape[0]))
    for j in range(J):
        ell_j = ell[:, 0] * np.sqrt(alpha[:, 0] / g[:, j])
        out += psd_rbf(f, ell_j[:, None])
    return out / J


def _erfinv_approx(x: np.ndarray) -> np.ndarray:
    a = 0.147
    ln = np.log(np.maximum(1.0 - x * x, 1e-12))
    term = 2.0 / (np.pi * a) + ln / 2.0
    return np.sign(x) * np.sqrt(np.sqrt(term ** 2 - ln / a) - term)


def psd_periodic(f: np.ndarray, f0: np.ndarray, n_harm: int,
                 decay: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Comb of Gaussian peaks at harmonics of f0.

    ``width`` broadens the peaks, which is exactly the PSD-domain effect of
    multiplying the periodic kernel by an RBF envelope (locally periodic).
    """
    out = np.zeros((f0.shape[0], f.shape[0]))
    for k in range(1, n_harm + 1):
        centre = f0 * k
        amp = np.exp(-decay * (k - 1))
        out += amp * np.exp(-0.5 * ((f[None, :] - centre) / width) ** 2)
    return out


def psd_spectral_mixture(f: np.ndarray, centres: np.ndarray, widths: np.ndarray,
                         weights: np.ndarray) -> np.ndarray:
    """Sum of Q Gaussian peaks at arbitrary (non-integer-period) centres."""
    out = np.zeros((centres.shape[0], f.shape[0]))
    Q = centres.shape[1]
    for q in range(Q):
        c = centres[:, q:q + 1]
        w = np.maximum(widths[:, q:q + 1], 1e-9)
        out += weights[:, q:q + 1] * np.exp(-0.5 * ((f[None, :] - c) / w) ** 2)
    return out


def psd_pink(f: np.ndarray, beta: np.ndarray, f_break: np.ndarray) -> np.ndarray:
    """1/f^beta with a low-frequency spectral break (flat below ``f_break``)."""
    ff = np.maximum(f[None, :], 1e-12)
    return 1.0 / (f_break ** beta + ff ** beta)


# ─────────────────────────────── the sampler ───────────────────────────────

def gp_from_psd(rng: np.random.Generator, psd: np.ndarray, L: int) -> np.ndarray:
    """Draw a stationary Gaussian field with spectral density ``psd``.

    ``psd`` has shape (n, L + 1) on the 2L rfft grid.  Rows are returned with
    zero mean and unit standard deviation.
    """
    n, nf = psd.shape
    amp = np.sqrt(np.maximum(psd, 0.0))
    re = rng.standard_normal((n, nf))
    im = rng.standard_normal((n, nf))
    np.multiply(re, amp, out=re)
    np.multiply(im, amp, out=im)
    im[:, 0] = 0.0
    im[:, -1] = 0.0
    spec = np.empty((n, nf), dtype=np.complex128)
    spec.real = re
    spec.imag = im
    x = np.ascontiguousarray(np.fft.irfft(spec, n=2 * L, axis=1)[:, :L])
    x -= x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    np.maximum(s, 1e-300, out=s)
    x /= s
    return x


def matern_gp(rng: np.random.Generator, n: int, L: int, ell: np.ndarray,
              nu: float = 1.5) -> np.ndarray:
    """Convenience: unit-variance Matern-nu field with per-row lengthscale."""
    f = freq_grid(L)
    ell = np.asarray(ell, dtype=np.float64).reshape(n, 1)
    return gp_from_psd(rng, psd_matern(f, ell, nu), L)


def pink_gp(rng: np.random.Generator, n: int, L: int, beta: np.ndarray,
            f_break: np.ndarray | float = 1.0 / 4096.0) -> np.ndarray:
    f = freq_grid(L)
    beta = np.asarray(beta, dtype=np.float64).reshape(n, 1)
    fb = np.full((n, 1), f_break) if np.isscalar(f_break) else np.asarray(f_break).reshape(n, 1)
    return gp_from_psd(rng, psd_pink(f, beta, fb), L)


def psd_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PSD convolution == kernel product.  Both inputs (n, nf), same n."""
    n, nf = a.shape
    m = 1
    while m < 2 * nf:
        m <<= 1
    fa = np.fft.rfft(a, n=m, axis=1)
    fb = np.fft.rfft(b, n=m, axis=1)
    c = np.fft.irfft(fa * fb, n=m, axis=1)[:, :nf]
    return np.maximum(c, 0.0)
