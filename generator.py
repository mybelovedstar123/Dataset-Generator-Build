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


# ═══════════════ chronoforge: cf_kernels ═══════════════
# Sequential recursions that NumPy cannot vectorise, as numba kernels.
#
# Every kernel takes *per-row parameter arrays* and writes into a caller-allocated
# ``(n, L)`` output, so one call replaces ``n`` Python-level calls.  All are
# compiled with ``fastmath=False`` and ``parallel=False`` (no reduction-order
# variation), ``nogil=True`` (so the producer threads actually overlap) and
# ``cache=False`` (the sandbox must not write files).
#
# Randomness never originates inside a kernel: every stochastic kernel consumes
# pre-drawn uniform / normal / gamma variates produced by a seeded NumPy
# ``Generator``.  That keeps the corpus a pure function of ``(seed, batch)``
# regardless of numba's internal thread-local RNG state.


import math
import threading


try:  # pragma: no cover - exercised by whichever branch the env provides
    from numba import njit as _numba_njit

    HAVE_NUMBA = True
except Exception:  # numba unavailable -> pure-Python fallback (slow but correct)
    HAVE_NUMBA = False
    _numba_njit = None


def _kernel(fn):
    if HAVE_NUMBA:
        return _numba_njit(cache=False, fastmath=False, parallel=False, nogil=True)(fn)
    return fn


def _inline(fn):
    if HAVE_NUMBA:
        return _numba_njit(cache=False, fastmath=False, parallel=False,
                           nogil=True, inline="always")(fn)
    return fn


# ─────────────────────────── count sampling helper ─────────────────────────

@_inline
def _pois(lam, u, z):
    """Poisson draw from a pre-drawn uniform ``u`` and standard normal ``z``.

    Exact inversion below lambda=30; a continuity-corrected normal above it
    (relative error < 1e-3 there, and the tail behaviour is what matters).
    """
    if lam <= 0.0:
        return 0.0
    if lam < 30.0:
        p = math.exp(-lam)
        s = p
        k = 0
        while u > s and k < 400:
            k += 1
            p *= lam / k
            s += p
        return float(k)
    v = lam + math.sqrt(lam) * z
    if v < 0.0:
        v = 0.0
    return math.floor(v + 0.5)


@_kernel
def k_poisson(lam, u, z, out):
    n, L = lam.shape
    for i in range(n):
        for t in range(L):
            out[i, t] = _pois(lam[i, t], u[i, t], z[i, t])


# ───────────────────────────────── ARMA ────────────────────────────────────

@_kernel
def k_arma(phi, p_ord, theta, q_ord, e, out):
    """Per-row ARMA(p, q) filter.  ``out`` must be zero-initialised."""
    n, L = e.shape
    for i in range(n):
        p = p_ord[i]
        q = q_ord[i]
        for t in range(L):
            v = e[i, t]
            for j in range(q):
                k = t - 1 - j
                if k >= 0:
                    v += theta[i, j] * e[i, k]
            for j in range(p):
                k = t - 1 - j
                if k >= 0:
                    v += phi[i, j] * out[i, k]
            if v > 1.0e150:
                v = 1.0e150
            elif v < -1.0e150:
                v = -1.0e150
            out[i, t] = v


@_kernel
def k_seasonal_int(x, m, out):
    """Inverse of the seasonal difference (1 - B^m), per row."""
    n, L = x.shape
    for i in range(n):
        mm = m[i]
        if mm < 1:
            mm = 1
        for t in range(L):
            if t < mm:
                out[i, t] = x[i, t]
            else:
                out[i, t] = out[i, t - mm] + x[i, t]


@_kernel
def k_setar(thr, c_lo, phi_lo, c_hi, phi_hi, e, out):
    """Two-regime self-exciting threshold AR(1)."""
    n, L = e.shape
    for i in range(n):
        x = e[i, 0]
        out[i, 0] = x
        for t in range(1, L):
            if x < thr[i]:
                v = c_lo[i] + phi_lo[i] * x
            else:
                v = c_hi[i] + phi_hi[i] * x
            v += e[i, t]
            if v > 1.0e150:
                v = 1.0e150
            elif v < -1.0e150:
                v = -1.0e150
            out[i, t] = v
            x = v


@_kernel
def k_ar1_tv(phi, mu, e, out):
    """AR(1) around a time-varying mean: x_t = mu_t + phi (x_{t-1} - mu_{t-1}) + e_t."""
    n, L = e.shape
    for i in range(n):
        d = e[i, 0]
        out[i, 0] = mu[i, 0] + d
        for t in range(1, L):
            d = phi[i] * d + e[i, t]
            if d > 1.0e150:
                d = 1.0e150
            elif d < -1.0e150:
                d = -1.0e150
            out[i, t] = mu[i, t] + d


# ───────────────────────────────── GARCH ───────────────────────────────────

@_kernel
def k_garch(omega, alpha, gamma, beta, z, out_r, out_s):
    """GJR-GARCH(1,1): sigma^2_t = w + (a + g*1[e<0]) e^2_{t-1} + b sigma^2_{t-1}."""
    n, L = z.shape
    for i in range(n):
        denom = 1.0 - alpha[i] - 0.5 * gamma[i] - beta[i]
        if denom < 1.0e-4:
            denom = 1.0e-4
        s2 = omega[i] / denom
        eprev = 0.0
        for t in range(L):
            ind = 1.0 if eprev < 0.0 else 0.0
            s2 = omega[i] + (alpha[i] + gamma[i] * ind) * eprev * eprev + beta[i] * s2
            if s2 > 1.0e120:
                s2 = 1.0e120
            if s2 < 1.0e-30:
                s2 = 1.0e-30
            sd = math.sqrt(s2)
            e = sd * z[i, t]
            out_r[i, t] = e
            out_s[i, t] = sd
            eprev = e


# ───────────────────────────────── Hawkes ──────────────────────────────────

@_kernel
def k_hawkes(mu, decays, weights, u, z, out_lam, out_n):
    """Discrete-time self-exciting process with K exponential kernels.

    ``decays[i, k]`` is exp(-1/tau_k); ``weights[i, k]`` the branching weight of
    component k.  Emits both the conditional intensity and the counts.
    """
    n, L = mu.shape
    K = decays.shape[1]
    s = np.zeros(K, dtype=np.float64)
    for i in range(n):
        for k in range(K):
            s[k] = 0.0
        for t in range(L):
            lam = mu[i, t]
            for k in range(K):
                lam += s[k]
            if lam < 0.0:
                lam = 0.0
            if lam > 1.0e8:
                lam = 1.0e8
            cnt = _pois(lam, u[i, t], z[i, t])
            out_lam[i, t] = lam
            out_n[i, t] = cnt
            for k in range(K):
                s[k] = decays[i, k] * (s[k] + weights[i, k] * cnt)


# ───────────────────────── closed-loop resource control ────────────────────

@_kernel
def k_feedback(d, th_up, th_dn, k_up, k_dn, gam, lag, c0, mode, out, out_cap):
    """Autoscaling controller.

    mode 0 -> utilisation min(d/c, 1); 1 -> capacity c_t; 2 -> queue backlog.
    """
    n, L = d.shape
    for i in range(n):
        c = c0[i]
        if c <= 0.0:
            c = 1.0
        uc = 0
        dc = 0
        pend = 0
        pdir = 0
        q = 0.0
        m = mode[i]
        for t in range(L):
            dt = d[i, t]
            if c < 1.0e-9:
                c = 1.0e-9
            r = dt / c
            if r > th_up[i]:
                uc += 1
                dc = 0
            elif r < th_dn[i]:
                dc += 1
                uc = 0
            else:
                uc = 0
                dc = 0
            if pend > 0:
                pend -= 1
                if pend == 0:
                    if pdir > 0:
                        c = c * (1.0 + gam[i])
                    else:
                        c = c / (1.0 + gam[i])
                    if c > 1.0e12:
                        c = 1.0e12
                    if c < 1.0e-9:
                        c = 1.0e-9
            else:
                if uc >= k_up[i]:
                    pend = lag[i]
                    pdir = 1
                    uc = 0
                elif dc >= k_dn[i]:
                    pend = lag[i]
                    pdir = -1
                    dc = 0
            out_cap[i, t] = c
            if m == 0:
                v = dt / c
                if v > 1.0:
                    v = 1.0
                if v < 0.0:
                    v = 0.0
                out[i, t] = v
            elif m == 1:
                out[i, t] = c
            else:
                q = q + dt - c
                if q < 0.0:
                    q = 0.0
                if q > 1.0e12:
                    q = 1.0e12
                out[i, t] = q


@_kernel
def k_counter_reset_cal(inc, reset, out):
    """Monotone accumulation with resets flagged in ``reset`` (0/1)."""
    n, L = inc.shape
    for i in range(n):
        acc = 0.0
        for t in range(L):
            if reset[i, t] != 0:
                acc = 0.0
            acc += inc[i, t]
            if acc > 1.0e14:
                acc = 1.0e14
            out[i, t] = acc


# ─────────────────────────── epidemic renewal ──────────────────────────────

@_kernel
def k_renewal(w, nw, rt, imports, gam, u, z, i0, out):
    """I_t ~ NB( R_t * sum_s w_s I_{t-s} + imports_t , k ).

    ``gam[i, t]`` is a pre-drawn Gamma(k_i, 1) variate divided by k_i by the
    caller, so ``lam * gam`` is the gamma-mixed Poisson mean (i.e. negative
    binomial with dispersion k_i).
    """
    n, L = rt.shape
    W = w.shape[1]
    for i in range(n):
        taps = nw[i]
        if taps > W:
            taps = W
        for t in range(L):
            conv = 0.0
            for s in range(taps):
                k = t - 1 - s
                if k >= 0:
                    conv += w[i, s] * out[i, k]
                else:
                    conv += w[i, s] * i0[i]
            lam = rt[i, t] * conv + imports[i, t]
            if lam < 0.0:
                lam = 0.0
            if lam > 1.0e9:
                lam = 1.0e9
            out[i, t] = _pois(lam * gam[i, t], u[i, t], z[i, t])


# ─────────────────────── chaotic / delay-differential ──────────────────────

@_kernel
def k_rk4_3d(system, par, dt, state0, sub, out):
    """RK4 integration of Lorenz / Rossler / Chua / Hindmarsh-Rose.

    ``out`` has shape (n, L, 3).  ``sub`` is the number of RK4 sub-steps taken
    per emitted sample.
    """
    n = out.shape[0]
    L = out.shape[1]
    k1 = np.zeros(3, dtype=np.float64)
    k2 = np.zeros(3, dtype=np.float64)
    k3 = np.zeros(3, dtype=np.float64)
    k4 = np.zeros(3, dtype=np.float64)
    tmp = np.zeros(3, dtype=np.float64)
    st = np.zeros(3, dtype=np.float64)
    for i in range(n):
        st[0] = state0[i, 0]
        st[1] = state0[i, 1]
        st[2] = state0[i, 2]
        sysid = system[i]
        h = dt[i]
        ns = sub[i]
        a = par[i, 0]
        b = par[i, 1]
        c = par[i, 2]
        d = par[i, 3]
        for t in range(L):
            for _ in range(ns):
                _deriv3(sysid, st, a, b, c, d, k1)
                for j in range(3):
                    tmp[j] = st[j] + 0.5 * h * k1[j]
                _deriv3(sysid, tmp, a, b, c, d, k2)
                for j in range(3):
                    tmp[j] = st[j] + 0.5 * h * k2[j]
                _deriv3(sysid, tmp, a, b, c, d, k3)
                for j in range(3):
                    tmp[j] = st[j] + h * k3[j]
                _deriv3(sysid, tmp, a, b, c, d, k4)
                for j in range(3):
                    st[j] = st[j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                    if st[j] > 1.0e6:
                        st[j] = 1.0e6
                    elif st[j] < -1.0e6:
                        st[j] = -1.0e6
            out[i, t, 0] = st[0]
            out[i, t, 1] = st[1]
            out[i, t, 2] = st[2]


@_inline
def _deriv3(sysid, s, a, b, c, d, o):
    x = s[0]
    y = s[1]
    z = s[2]
    if sysid == 0:  # Lorenz
        o[0] = a * (y - x)
        o[1] = x * (b - z) - y
        o[2] = x * y - c * z
    elif sysid == 1:  # Rossler
        o[0] = -y - z
        o[1] = x + a * y
        o[2] = b + z * (x - c)
    elif sysid == 2:  # Chua (cubic nonlinearity)
        g = -a * x + b * x * x * x
        o[0] = c * (y - x - g)
        o[1] = x - y + z
        o[2] = -d * y
    else:  # Hindmarsh-Rose bursting neuron
        o[0] = y - a * x * x * x + b * x * x - z + d
        o[1] = 1.0 - c * x * x - y
        o[2] = 0.006 * (4.0 * (x + 1.6) - z)


@_kernel
def k_mackey_glass(beta, gamma_, nexp, delay, hist, out):
    """Mackey-Glass delay differential equation, 4 sub-steps per sample.

    ``hist[i, :]`` is a pre-filled circular history buffer (length >= delay+2).
    """
    n, L = out.shape
    B = hist.shape[1]
    sub = 4
    h = 1.0 / sub
    for i in range(n):
        D = delay[i]
        if D < 1:
            D = 1
        if D > B - 2:
            D = B - 2
        pos = B - 1
        x = hist[i, pos]
        for t in range(L):
            for _ in range(sub):
                idx = pos - D
                while idx < 0:
                    idx += B
                xd = hist[i, idx]
                den = 1.0 + math.pow(abs(xd), nexp[i])
                f = beta[i] * xd / den - gamma_[i] * x
                xm = x + 0.5 * h * f
                idx2 = idx + 1
                if idx2 >= B:
                    idx2 -= B
                xd2 = hist[i, idx2]
                den2 = 1.0 + math.pow(abs(xd2), nexp[i])
                f2 = beta[i] * xd2 / den2 - gamma_[i] * xm
                x = x + h * f2
                if x > 1.0e6:
                    x = 1.0e6
                elif x < -1.0e6:
                    x = -1.0e6
                pos += 1
                if pos >= B:
                    pos = 0
                hist[i, pos] = x
            out[i, t] = x


# ───────────────────────── quasi-periodic physiology ───────────────────────

@_kernel
def k_template(phase, tmpl, tid, out):
    """Linear-interpolated lookup of a per-row waveform template at ``phase``."""
    n, L = phase.shape
    S = tmpl.shape[1]
    for i in range(n):
        r = tid[i]
        for t in range(L):
            ph = phase[i, t]
            ph = ph - math.floor(ph)
            x = ph * S
            j = int(x)
            if j >= S:
                j = S - 1
            fr = x - j
            j2 = j + 1
            if j2 >= S:
                j2 = 0
            out[i, t] = tmpl[r, j] * (1.0 - fr) + tmpl[r, j2] * fr


# ─────────────────────────────── warm-up ───────────────────────────────────

_WARM_LOCK = threading.Lock()
_WARMED = False


def warmup() -> None:
    """Force numba compilation before the streaming loop starts."""
    global _WARMED
    if _WARMED or not HAVE_NUMBA:
        _WARMED = True
        return
    with _WARM_LOCK:
        if _WARMED:
            return
        n, L = 2, 3
        f = np.zeros((n, L))
        i32 = np.zeros(n, dtype=np.int64)
        one = np.ones(n)

        out = np.zeros((n, L))
        k_poisson(np.ones((n, L)), np.full((n, L), 0.5), f.copy(), out)
        k_arma(np.zeros((n, 2)), np.ones(n, dtype=np.int64), np.zeros((n, 2)),
               np.ones(n, dtype=np.int64), np.ones((n, L)), np.zeros((n, L)))
        k_seasonal_int(np.ones((n, L)), np.ones(n, dtype=np.int64), np.zeros((n, L)))
        k_setar(f.copy()[:, 0].copy(), one * 0, one * 0.5, one * 0, one * 0.5,
                np.ones((n, L)), np.zeros((n, L)))
        k_ar1_tv(one * 0.5, np.zeros((n, L)), np.ones((n, L)), np.zeros((n, L)))
        k_garch(one * 0.01, one * 0.05, one * 0.05, one * 0.85, np.ones((n, L)),
                np.zeros((n, L)), np.zeros((n, L)))
        k_hawkes(np.ones((n, L)) * 0.1, np.full((n, 4), 0.5), np.full((n, 4), 0.05),
                 np.full((n, L), 0.5), np.zeros((n, L)), np.zeros((n, L)),
                 np.zeros((n, L)))
        k_feedback(np.ones((n, L)), one * 0.8, one * 0.3,
                   np.full(n, 3, dtype=np.int64), np.full(n, 5, dtype=np.int64),
                   one * 0.3, np.full(n, 2, dtype=np.int64), one,
                   i32.copy(), np.zeros((n, L)), np.zeros((n, L)))
        k_counter_reset_cal(np.ones((n, L)), np.zeros((n, L), dtype=np.int8),
                            np.zeros((n, L)))
        k_renewal(np.full((n, 2), 0.5), np.full(n, 2, dtype=np.int64),
                  np.ones((n, L)), np.zeros((n, L)), np.ones((n, L)),
                  np.full((n, L), 0.5), np.zeros((n, L)), one, np.zeros((n, L)))
        k_rk4_3d(i32.copy(), np.full((n, 4), 1.0), one * 0.01,
                 np.ones((n, 3)), np.full(n, 1, dtype=np.int64), np.zeros((n, L, 3)))
        k_mackey_glass(one * 0.2, one * 0.1, one * 10.0,
                       np.full(n, 4, dtype=np.int64), np.ones((n, 16)),
                       np.zeros((n, L)))
        k_template(np.linspace(0, 1, n * L).reshape(n, L), np.ones((2, 8)),
                   i32.copy(), np.zeros((n, L)))
        _WARMED = True


# ═══════════════ chronoforge: cf_prims ═══════════════
# Vectorised building blocks shared by the family modules.
#
# Everything here is written to operate on a whole *group* of rows at once with
# per-row parameters carried as ``(n, 1)`` columns.  There are no per-series
# Python loops in the hot path; where a loop is unavoidable it runs over a small
# fixed number of *slots* (event slots, harmonics, mixture components), never over
# rows.


try:
    from scipy.ndimage import maximum_filter1d as _max_filter
    from scipy.ndimage import minimum_filter1d as _min_filter
    _HAVE_NDIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_NDIMAGE = False
    _max_filter = _min_filter = None

_cf_prims_TWO_PI = 2.0 * np.pi


# ─────────────────────────────── draw helpers ──────────────────────────────

def logu(rng: np.random.Generator, lo: float, hi: float, size) -> np.ndarray:
    """Log-uniform draw."""
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))


def col(x: np.ndarray) -> np.ndarray:
    """Reshape a length-n vector into an (n, 1) broadcast column."""
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def categorical(rng: np.random.Generator, probs, size: int) -> np.ndarray:
    """Vectorised categorical draw; returns int64 indices."""
    p = np.asarray(probs, dtype=np.float64)
    c = np.cumsum(p / p.sum())
    u = rng.random(size)
    return np.searchsorted(c, u, side="right").clip(0, len(p) - 1).astype(np.int64)


def bernoulli(rng: np.random.Generator, p, size: int) -> np.ndarray:
    return rng.random(size) < p


def safe_std(x: np.ndarray) -> np.ndarray:
    """Row standard deviation, floored so no downstream division can overflow.

    The floor is relative as well as absolute: a *nearly* constant row (std far
    below its own magnitude) would otherwise be amplified to infinity.  Rows that
    trip the relative floor are degenerate by construction and get caught by the
    sanitiser's regeneration guard.
    """
    s = x.std(axis=1, keepdims=True)
    m = np.abs(x).max(axis=1, keepdims=True)
    return np.maximum(s, np.maximum(m * 1e-12, 1e-300))


def unit_std(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std rows (never in place)."""
    x = x - x.mean(axis=1, keepdims=True)
    return x / safe_std(x)


# ───────────────────────────── linear processes ────────────────────────────

def ar1(rng: np.random.Generator, n: int, L: int, phi: np.ndarray,
        sigma: np.ndarray | float = 1.0) -> np.ndarray:
    """AR(1) with per-row phi.  Returns (n, L)."""
    e = rng.standard_normal((n, L))
    phi = np.asarray(phi, dtype=np.float64).reshape(n, 1)
    e *= np.sqrt(np.maximum(1.0 - phi ** 2, 1e-6))
    out = np.zeros((n, L))
    k_arma(phi.copy(), np.ones(n, dtype=np.int64),
             np.zeros((n, 1)), np.zeros(n, dtype=np.int64), e, out)
    if np.isscalar(sigma):
        if sigma != 1.0:
            out *= sigma
    else:
        out *= np.asarray(sigma, dtype=np.float64).reshape(n, 1)
    return out


def ou(rng: np.random.Generator, n: int, L: int, tau: np.ndarray) -> np.ndarray:
    """Discretised Ornstein-Uhlenbeck with correlation time ``tau`` samples."""
    phi = np.exp(-1.0 / np.maximum(np.asarray(tau, dtype=np.float64).reshape(n, 1), 1e-3))
    return ar1(rng, n, L, np.clip(phi, 0.0, 0.99999))


def local_linear_trend(rng: np.random.Generator, n: int, L: int,
                       sig_level: np.ndarray, sig_slope: np.ndarray,
                       damp: np.ndarray | None = None) -> np.ndarray:
    """Local-linear-trend state space; ``damp`` < 1 gives a damped trend."""
    es = rng.standard_normal((n, L)) * np.asarray(sig_slope).reshape(n, 1)
    el = rng.standard_normal((n, L)) * np.asarray(sig_level).reshape(n, 1)
    if damp is None:
        slope = np.cumsum(es, axis=1)
    else:
        d = np.asarray(damp, dtype=np.float64).reshape(n, 1)
        slope = np.zeros((n, L))
        k_arma(d.copy(), np.ones(n, dtype=np.int64), np.zeros((n, 1)),
                 np.zeros(n, dtype=np.int64), es, slope)
    lag = np.zeros((n, L))
    lag[:, 1:] = slope[:, :-1]
    return np.cumsum(lag + el, axis=1)


# ─────────────────────────── segmentation helpers ──────────────────────────

def segment_map(dwell: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Map cumulative dwell times to a per-sample segment id.

    ``dwell`` is (n, M) positive lengths.  Returns ``(seg_id, seg_start)`` both
    (n, L) int64 / float64, computed with bincount + cumsum (no per-row loop).
    """
    n, M = dwell.shape
    edges = np.cumsum(np.maximum(dwell, 1.0), axis=1)
    pos = np.floor(edges).astype(np.int64)
    valid = (pos > 0) & (pos < L)
    rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, M))
    flat = (rows[valid] * L + pos[valid])
    mark = np.bincount(flat, minlength=n * L).reshape(n, L).astype(np.int64)
    seg_id = np.cumsum(mark, axis=1)
    # segment start index for each sample
    t = np.arange(L, dtype=np.float64)[None, :]
    starts = np.where(mark > 0, t, 0.0)
    seg_start = np.maximum.accumulate(starts, axis=1)
    return seg_id, seg_start


def gather_levels(levels: np.ndarray, seg_id: np.ndarray) -> np.ndarray:
    """levels (n, M) gathered at per-sample segment ids (n, L)."""
    M = levels.shape[1]
    idx = np.clip(seg_id, 0, M - 1)
    return np.take_along_axis(levels, idx, axis=1)


def run_mask(n: int, L: int, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Boolean mask covering runs [s, s+len) given ragged start/length lists.

    ``starts`` / ``lengths`` are (n, E) arrays; non-positive lengths are ignored.
    """
    E = starts.shape[1]
    s = np.clip(starts, 0, L - 1).astype(np.int64)
    e = np.clip(starts + np.maximum(lengths, 0), 0, L).astype(np.int64)
    live = e > s
    rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, E))
    delta = np.zeros((n, L + 1), dtype=np.int64)
    np.add.at(delta, (rows[live], s[live]), 1)
    np.add.at(delta, (rows[live], e[live]), -1)
    return np.cumsum(delta[:, :L], axis=1) > 0


def event_positions(rng: np.random.Generator, n: int, L: int, count: np.ndarray,
                    E: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform event positions with a per-row count, padded to E slots."""
    pos = rng.integers(0, L, size=(n, E)).astype(np.int64)
    live = np.arange(E)[None, :] < np.asarray(count).reshape(n, 1)
    return pos, live


def bursty_gaps(rng: np.random.Generator, n: int, L: int, rate: np.ndarray,
                mean_len: np.ndarray, E: int = 24) -> np.ndarray:
    """Boolean gap mask from a bursty (LogN-length) outage process."""
    cnt = rng.poisson(np.maximum(np.asarray(rate).reshape(n, 1) * L, 0.0), size=(n, E))
    cnt = (cnt > 0).astype(np.int64)
    starts = rng.integers(0, L, size=(n, E))
    lens = np.exp(rng.normal(np.log(np.maximum(np.asarray(mean_len).reshape(n, 1), 1.0)),
                             0.9, size=(n, E)))
    lens = np.where(cnt > 0, lens, 0.0)
    return run_mask(n, L, starts, lens)


def locf(x: np.ndarray, gap: np.ndarray) -> np.ndarray:
    """Last-observation-carried-forward over a boolean gap mask."""
    n, L = x.shape
    idx = np.where(gap, -1, np.arange(L)[None, :])
    idx = np.maximum.accumulate(idx, axis=1)
    idx = np.maximum(idx, 0)
    return np.take_along_axis(x, idx, axis=1)


# ─────────────────────────── periodic profile shapes ───────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def phase_of(L: int, period: np.ndarray, phase0: np.ndarray,
             drift: np.ndarray | None = None,
             period_drift: np.ndarray | None = None) -> np.ndarray:
    """Cycle phase in [0,1) with optional slow phase drift / period drift."""
    t = time_grid(L)[None, :]
    p = np.maximum(np.asarray(period, dtype=np.float64).reshape(-1, 1), 1e-6)
    if period_drift is None:
        ph = t / p
    else:
        d = np.asarray(period_drift, dtype=np.float64).reshape(-1, 1)
        # instantaneous frequency 1/p * (1 + d * t/L); phase = integral
        ph = (t / p) * (1.0 + d * t / (2.0 * L))
    ph = ph + np.asarray(phase0, dtype=np.float64).reshape(-1, 1)
    if drift is not None:
        ph = ph + np.asarray(drift, dtype=np.float64).reshape(-1, 1) * (t / L)
    return ph


def warp_phase(ph: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Circle map theta -> theta + kappa sin(theta), applied on the unit cycle."""
    k = np.asarray(kappa, dtype=np.float64).reshape(-1, 1)
    return ph + (k / _cf_prims_TWO_PI) * np.sin(_cf_prims_TWO_PI * ph)


def profile_harmonic(rng, n, L, ph, H=5):
    alpha = rng.uniform(0.7, 1.8, size=(n, 1))
    nh = rng.integers(2, H + 1, size=(n, 1))
    out = np.zeros((n, L))
    for h in range(1, H + 1):
        amp = rng.standard_normal((n, 1)) * (h ** -alpha)
        psi = rng.uniform(0.0, _cf_prims_TWO_PI, size=(n, 1))
        out += np.where(h <= nh, 1.0, 0.0) * amp * np.cos(_cf_prims_TWO_PI * h * ph + psi)
    return out


def profile_trapezoid(rng, n, L, ph):
    a = rng.uniform(0.20, 0.35, size=(n, 1))
    b = rng.uniform(0.68, 0.92, size=(n, 1))
    r1 = rng.uniform(0.01, 0.06, size=(n, 1))
    r2 = rng.uniform(0.02, 0.10, size=(n, 1))
    f = ph - np.floor(ph)
    y = _sigmoid((f - a) / r1) - _sigmoid((f - b) / r2)
    notch = rng.random((n, 1)) < 0.28
    nc = rng.uniform(0.42, 0.58, size=(n, 1))
    nw = rng.uniform(0.02, 0.06, size=(n, 1))
    nd = rng.uniform(0.10, 0.35, size=(n, 1))
    y = y - np.where(notch, nd, 0.0) * np.exp(-0.5 * ((f - nc) / nw) ** 2)
    tilt = rng.uniform(-0.25, 0.25, size=(n, 1))
    return y * (1.0 + tilt * (f - 0.5))


def profile_double_peak(rng, n, L, ph):
    m1 = rng.uniform(0.25, 0.40, size=(n, 1))
    m2 = rng.uniform(0.62, 0.80, size=(n, 1))
    k1 = rng.uniform(8.0, 60.0, size=(n, 1))
    k2 = rng.uniform(6.0, 40.0, size=(n, 1))
    h2 = rng.uniform(0.4, 1.6, size=(n, 1))
    c = np.cos(_cf_prims_TWO_PI * (ph - m1))
    d = np.cos(_cf_prims_TWO_PI * (ph - m2))
    return np.exp(k1 * (c - 1.0)) + h2 * np.exp(k2 * (d - 1.0))


def profile_free_periodic(rng, n, L, ph, H=16):
    out = np.zeros((n, L))
    decay = rng.uniform(0.8, 2.0, size=(n, 1))
    nh = rng.integers(8, H + 1, size=(n, 1))
    for h in range(1, H + 1):
        w = float(h) ** (-1.0)
        amp = rng.standard_normal((n, 1)) * (w ** decay)
        psi = rng.uniform(0.0, _cf_prims_TWO_PI, size=(n, 1))
        out += np.where(h <= nh, 1.0, 0.0) * amp * np.cos(_cf_prims_TWO_PI * h * ph + psi)
    return out


SHAPE_PROBS = (0.40, 0.30, 0.15, 0.10, 0.05)  # harmonic / trapezoid / double / warped / freeGP


def seasonal_profile(rng: np.random.Generator, n: int, L: int, period: np.ndarray,
                     cfg: dict | None = None, allow_drift: bool = True) -> np.ndarray:
    """Unit-std periodic profile at ``period`` samples, one row per series.

    Shapes: harmonic series, business-hours trapezoid, double von-Mises peak,
    circle-map-warped variant, free periodic GP.  Optional slow phase drift and
    genuine period drift (a phase accumulator over a varying instantaneous
    frequency), which no fixed-grid seasonal model can express.
    """
    sea = (cfg or {}).get("seasonality", {})
    shape_probs = sea.get("shape_mix", SHAPE_PROBS)
    p_phase = float(sea.get("phase_drift_rate", 0.35))
    p_period = float(sea.get("period_drift_rate", 0.12))
    period = np.asarray(period, dtype=np.float64).reshape(n, 1)
    phase0 = rng.random((n, 1))
    drift = np.where(rng.random((n, 1)) < p_phase,
                     rng.uniform(0.02, 0.10, size=(n, 1)) * np.sign(rng.standard_normal((n, 1))),
                     0.0) if allow_drift else None
    pdrift = np.where(rng.random((n, 1)) < p_period,
                      rng.uniform(0.005, 0.03, size=(n, 1)) * np.sign(rng.standard_normal((n, 1))),
                      0.0) if allow_drift else None
    ph = phase_of(L, period, phase0, drift, pdrift)

    kind = categorical(rng, shape_probs, n)
    kappa = rng.uniform(-0.6, 0.6, size=(n, 1))
    ph_w = warp_phase(ph, kappa)

    out = np.zeros((n, L))
    for k, fn, use_warp in (
        (0, profile_harmonic, False),
        (1, profile_trapezoid, False),
        (2, profile_double_peak, False),
        (3, profile_harmonic, True),
        (4, profile_free_periodic, False),
    ):
        m = kind == k
        if not m.any():
            continue
        sub = int(m.sum())
        out[m] = fn(rng, sub, L, (ph_w if use_warp else ph)[m])
    return unit_std(out)


# ────────────────────────────── misc processes ─────────────────────────────

def hawkes(rng: np.random.Generator, n: int, L: int, mu: np.ndarray,
           branching: np.ndarray, tau: np.ndarray,
           power_law: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time self-exciting arrivals; returns (intensity, counts).

    ``power_law`` rows use four exponential components geometrically spaced over
    three decades, which approximates a genuine long-memory power-law kernel.
    """
    K_COMP = 4
    tau = np.asarray(tau, dtype=np.float64).reshape(n, 1)
    br = np.clip(np.asarray(branching, dtype=np.float64).reshape(n, 1), 0.0, 0.95)
    taus = np.repeat(tau, K_COMP, axis=1)
    wts = np.zeros((n, K_COMP))
    if power_law is None:
        power_law = np.zeros(n, dtype=bool)
    pl = np.asarray(power_law).reshape(n, 1)
    mult = np.array([1.0, 4.6416, 21.544, 100.0])[None, :]
    taus = np.where(pl, tau * mult, taus)
    w_pl = np.array([0.55, 0.25, 0.13, 0.07])[None, :]
    w_ex = np.array([1.0, 0.0, 0.0, 0.0])[None, :]
    wts = np.where(pl, w_pl, w_ex)
    taus = np.maximum(taus, 1.0)
    decays = np.exp(-1.0 / taus)
    # branching ratio n = sum_k alpha_k * tau_k  -> normalise
    norm = np.sum(wts * taus, axis=1, keepdims=True)
    alpha = wts * br / np.maximum(norm, 1e-9)
    lam = np.zeros((n, L))
    cnt = np.zeros((n, L))
    u = rng.random((n, L))
    z = rng.standard_normal((n, L))
    k_hawkes(np.ascontiguousarray(mu), np.ascontiguousarray(decays),
               np.ascontiguousarray(alpha), u, z, lam, cnt)
    return lam, cnt


def decay_convolve(x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Causal exponential smoothing with per-row timescale (via k_arma)."""
    n, L = x.shape
    phi = np.exp(-1.0 / np.maximum(np.asarray(tau, dtype=np.float64).reshape(n, 1), 1e-3))
    out = np.zeros((n, L))
    k_arma(np.ascontiguousarray(phi), np.ones(n, dtype=np.int64),
             np.zeros((n, 1)), np.zeros(n, dtype=np.int64),
             np.ascontiguousarray(x), out)
    return out


GAMMA_NORMAL_SHAPE = 60.0


def gamma_shape_mix(rng: np.random.Generator, n: int, L: int,
                    k: np.ndarray) -> np.ndarray:
    """Gamma(k, 1/k) variates (unit mean) with a per-row shape.

    Above ``GAMMA_NORMAL_SHAPE`` the Wilson-Hilferty cube-root transform of a
    normal is accurate to better than 1e-3 in the body and far cheaper than a
    rejection sampler; below it we draw exactly.
    """
    k = np.maximum(np.asarray(k, dtype=np.float64).reshape(n, 1), 1e-3)
    big = k >= GAMMA_NORMAL_SHAPE
    if big.all():
        z = rng.standard_normal((n, L))
        c = 1.0 / (9.0 * k)
        return np.maximum((1.0 - c + z * np.sqrt(c)) ** 3, 0.0)
    if not big.any():
        return rng.gamma(np.broadcast_to(k, (n, L))) / k
    out = rng.gamma(np.broadcast_to(k, (n, L))) / k
    idx = np.nonzero(big[:, 0])[0]
    z = rng.standard_normal((idx.size, L))
    c = 1.0 / (9.0 * k[idx])
    out[idx] = np.maximum((1.0 - c + z * np.sqrt(c)) ** 3, 0.0)
    return out


POISSON_NORMAL_LAMBDA = 200.0


def nb_counts(rng: np.random.Generator, mean: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Negative-binomial counts via a Gamma-mixed Poisson (var = mu + mu^2/k).

    Above ``POISSON_NORMAL_LAMBDA`` the Poisson layer is replaced by a
    continuity-corrected normal (relative error < 1e-3 there, and the
    over-dispersion that actually shapes the series comes from the Gamma mix).
    """
    n, L = mean.shape
    lam = np.minimum(np.maximum(mean, 0.0) * gamma_shape_mix(rng, n, L, k), 1e8)
    big = lam > POISSON_NORMAL_LAMBDA
    if not big.any():
        return rng.poisson(lam).astype(np.float64)
    small = rng.poisson(np.where(big, 0.0, lam)).astype(np.float64)
    z = rng.standard_normal((n, L))
    approx = np.maximum(np.floor(lam + np.sqrt(lam) * z + 0.5), 0.0)
    return np.where(big, approx, small)


def round_tick(x: np.ndarray, tick: np.ndarray) -> np.ndarray:
    tick = np.maximum(np.asarray(tick, dtype=np.float64).reshape(-1, 1), 1e-300)
    return np.round(x / tick) * tick


def human_tick(scale: np.ndarray, rng: np.random.Generator, n: int,
               rel_lo=1e-3, rel_hi=0.2) -> np.ndarray:
    """A round human tick {1,2,5}x10^k sized relative to a series scale.

    ``rel_lo`` / ``rel_hi`` may be scalars or (n, 1) columns.  Ticks are round
    human values because that is what real reporting granularity looks like, and
    a *recurring* tick is learnable in a way an arbitrary quantiser is not.
    """
    lo = np.broadcast_to(np.asarray(rel_lo, dtype=np.float64).reshape(-1, 1), (n, 1))
    hi = np.maximum(np.broadcast_to(
        np.asarray(rel_hi, dtype=np.float64).reshape(-1, 1), (n, 1)), lo * 1.001)
    rel = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n, 1)))
    raw = np.maximum(np.asarray(scale).reshape(n, 1), 1e-300) * rel
    ex = np.floor(np.log10(raw))
    mant = raw / (10.0 ** ex)
    mant = np.where(mant < 1.5, 1.0, np.where(mant < 3.5, 2.0, 5.0))
    return mant * (10.0 ** ex)


def rolling_agg(x: np.ndarray, k: int, mode: int) -> np.ndarray:
    """Causal rolling aggregate over the last ``k`` samples (mode 0/1/2/3 =
    mean/sum/max/min)."""
    n, L = x.shape
    if k <= 1:
        return x
    if mode <= 1:
        c = np.cumsum(x, axis=1)
        pad = np.zeros((n, 1))
        cp = np.concatenate([pad, c], axis=1)
        idx = np.maximum(np.arange(L) - k + 1, 0)
        s = c - cp[:, idx]
        cnt = np.minimum(np.arange(L) + 1, k)[None, :].astype(np.float64)
        return s / cnt if mode == 0 else s
    origin = (k - 1) - k // 2
    if _HAVE_NDIMAGE:
        f = _max_filter if mode == 2 else _min_filter
        return f(x, size=k, axis=1, mode="nearest", origin=origin)
    acc = x.copy()
    for j in range(1, k):
        shifted = np.empty_like(x)
        shifted[:, j:] = x[:, :-j]
        shifted[:, :j] = x[:, :1]
        if mode == 2:
            np.maximum(acc, shifted, out=acc)
        else:
            np.minimum(acc, shifted, out=acc)
    return acc


# ─────────────────────────── per-row capability flags ──────────────────────

FLAG_DEFAULTS = {
    "scale_mode": 0,      # 0 free (standardise + offset/scale) | 1 positive | 2 as-is
    "offset_pref": 0,     # 0 mixture | 1 large offset | 2 zero anchor | 3 sign crossing
    "integer": False,     # values already lie on a discrete lattice
    "count": False,       # genuine integer counts (set by the batch builder)
    "bounded": False,     # has hard bounds that artefacts must not break
    "positive": False,    # non-negative by construction
    "quant_boost": 1.0,   # multiplier on the quantisation stage probability
    "quant_rel_hi": 0.2,  # upper bound on tick/std for the quantiser
    "obs_boost": 1.0,     # multiplier on every observation-artefact rate
    "allow_agg": True,    # may participate in structured hierarchical aggregation
}

_FLAG_DTYPE = {
    "scale_mode": np.int8, "offset_pref": np.int8, "integer": bool,
    "count": bool,
    "bounded": bool, "positive": bool, "quant_boost": np.float64,
    "quant_rel_hi": np.float64, "obs_boost": np.float64, "allow_agg": bool,
}


def new_flags(n: int, **over) -> dict:
    """Per-row capability flags with family overrides."""
    f = {}
    for k, v in FLAG_DEFAULTS.items():
        f[k] = np.full(n, over.get(k, v), dtype=_FLAG_DTYPE[k])
    for k in over:
        if k not in FLAG_DEFAULTS:
            raise KeyError(f"unknown row flag {k!r}")
    return f


def complete_dwell_table(rng, dwell, L, median, sigma, minimum=1.0,
                         heavy=None, pareto_a=None):
    """Extend only exhausted renewal tables; never tile or freeze the last slot.

    Called after the legacy builder has drawn its carriers/noise, so extending
    a table cannot move any of those draws. median/sigma have one column for
    iid durations or two columns for an alternating process. The existing
    floating dwell law and floor(cumsum) event locations are preserved.
    Returns affected original row indices, their completed table, and the first
    invalid legacy sample. Non-exhausted rows consume no extension randomness.
    """
    end = np.floor(np.cumsum(np.maximum(dwell, 1.0), axis=1)[:, -1]).astype(np.int64)
    rows = np.flatnonzero(end < L)
    if rows.size == 0:
        return rows, dwell[:0], end[:0]
    table = dwell[rows].copy()
    med = np.asarray(median, dtype=np.float64).reshape(dwell.shape[0], -1)[rows]
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64),
                          (dwell.shape[0], med.shape[1]))[rows]
    if med.shape[1] not in (1, 2) or np.any(med <= 0) or np.any(sig < 0):
        raise ValueError('invalid native renewal duration parameters')
    # Every appended duration is >= minimum >= 1. Thus even the shortest
    # possible draws terminate after at most L extra slots; no probabilistic cap.
    while np.any(np.floor(np.cumsum(table, axis=1)[:, -1]) < L):
        width = min(64, max(1, int(np.ceil(L / max(minimum, 1.0))) + 1 - table.shape[1]))
        columns = (table.shape[1] + np.arange(width)) % med.shape[1]
        centre = med[:, columns]
        sd = sig[:, columns]
        extra = np.exp(rng.normal(np.log(np.maximum(centre, 1.0)), sd))
        if heavy is not None:
            u = np.maximum(rng.random(extra.shape), 1e-9)
            shape = np.asarray(pareto_a, dtype=np.float64)[rows, None]
            tail = centre * np.power(u, -1.0 / shape)
            use = ((rng.random(extra.shape) < 0.15)
                   & np.asarray(heavy, dtype=bool)[rows, None])
            extra = np.where(use, tail, extra)
        table = np.concatenate((table, np.maximum(extra, minimum)), axis=1)
    return rows, table, end[rows]
