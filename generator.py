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
