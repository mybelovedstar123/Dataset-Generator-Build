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


# ═══════════════ chronoforge: cf_cadence ═══════════════
# Cadence, derived periods, and the length ladder.
#
# We do not carry a hard-coded list of integer seasonal periods.  We sample a
# *cadence* (the wall-clock spacing between samples, matched to the pool's
# frequency map) and then derive every period physically::
#
#     P_day   = 86400 / c            P_week  = 7   * P_day
#     P_half  = P_day / 2            P_month = 30.436875 * P_day
#     P_third = P_day / 3            P_year  = 365.2425  * P_day
#
# Periods are used as *floats*.  At a daily cadence ``P_month = 30.436875`` is
# genuinely non-integer, which produces the slow phase drift an integer-30 model
# cannot reproduce; at an hourly cadence ``P_year = 8765.8`` exceeds the window
# and correctly enters as a slow trend rather than a cycle.
#
# Length is drawn conditional on the cadence class: short evaluation contexts in
# the real pool come from *daily* feeds, so our short training series carry daily
# structure rather than truncated hourly structure.  Every length is an exact
# multiple of 32 (the trainer buckets by ``p = L // 32`` and discards the
# remainder, so a non-multiple silently throws away token budget).


# Cadences (seconds between samples) matched to the pool's FREQ_MAP.
FINE_SECONDS = np.array([30, 60, 150, 300, 360, 600, 900, 1800, 3600], dtype=np.float64)
COARSE_SECONDS = np.array([28800, 86400], dtype=np.float64)

SECONDS_PER_DAY = 86400.0

# Derived-period multipliers on P_day.
MULT_HALF = 0.5
MULT_THIRD = 1.0 / 3.0
MULT_WEEK = 7.0
MULT_BIZWEEK = 5.0
MULT_MONTH = 30.436875
MULT_QUARTER = 91.310625
MULT_YEAR = 365.2425


class CadenceDraw:
    """Per-row cadence for one batch (dense, batch-sized)."""

    __slots__ = ("seconds", "p_day", "is_coarse", "length", "n")

    def __init__(self, seconds: np.ndarray, is_coarse: np.ndarray, length: np.ndarray):
        self.seconds = seconds
        self.p_day = SECONDS_PER_DAY / seconds
        self.is_coarse = is_coarse
        self.length = length
        self.n = seconds.shape[0]

    def take(self, rows: np.ndarray) -> "CadenceDraw":
        return CadenceDraw(self.seconds[rows], self.is_coarse[rows], self.length[rows])

    # ── derived periods, as (n, 1) float columns ──
    def period(self, name: str) -> np.ndarray:
        pd = self.p_day.reshape(-1, 1)
        return pd * _MULT[name]


_MULT = {
    "day": 1.0,
    "half": MULT_HALF,
    "third": MULT_THIRD,
    "week": MULT_WEEK,
    "bizweek": MULT_BIZWEEK,
    "month": MULT_MONTH,
    "quarter": MULT_QUARTER,
    "year": MULT_YEAR,
}


def draw_cadence(rng: np.random.Generator, n: int, p_coarse: np.ndarray,
                 cfg: dict) -> CadenceDraw:
    """Draw a cadence and a length for every row of the batch."""
    fine_s = np.asarray(cfg["cadence"]["fine_seconds"], dtype=np.float64)
    coarse_s = np.asarray(cfg["cadence"]["coarse_seconds"], dtype=np.float64)
    fine_w = np.asarray(cfg["cadence"]["fine_weights"], dtype=np.float64)
    coarse_w = np.asarray(cfg["cadence"]["coarse_weights"], dtype=np.float64)

    is_coarse = rng.random(n) < np.asarray(p_coarse, dtype=np.float64)
    fi = categorical(rng, fine_w, n)
    ci = categorical(rng, coarse_w, n)
    seconds = np.where(is_coarse, coarse_s[ci], fine_s[fi])

    fine_lad = cfg["length_ladder"]["fine"]
    coarse_lad = cfg["length_ladder"]["coarse"]
    fl = np.asarray(fine_lad["lengths"], dtype=np.int64)
    fw = np.asarray(fine_lad["weights"], dtype=np.float64)
    cl = np.asarray(coarse_lad["lengths"], dtype=np.int64)
    cw = np.asarray(coarse_lad["weights"], dtype=np.float64)
    length = np.where(is_coarse, cl[categorical(rng, cw, n)],
                      fl[categorical(rng, fw, n)])
    return CadenceDraw(seconds.astype(np.float64), is_coarse, length.astype(np.int64))


# ─────────────────────── which periods are active ──────────────────────────

ACTIVE_KEYS = ("day", "half", "third", "week", "bizweek", "month", "quarter", "free")


def active_periods(rng: np.random.Generator, cad: CadenceDraw, cfg: dict,
                   L: int) -> dict[str, np.ndarray]:
    """Cadence-conditional Bernoulli activation of each derived period.

    Returns a dict ``key -> (n, 1) period length in samples`` where inactive
    rows carry ``0.0``.  Also emits a ``free`` entry: a continuous period drawn
    log-uniformly in ``[6, L/3]``, incommensurate with everything else.  That is
    the escape hatch which stops the model over-fitting to a fixed grid.
    """
    n = cad.n
    tbl_fine = cfg["seasonality"]["active_fine"]
    tbl_coarse = cfg["seasonality"]["active_coarse"]
    coarse = cad.is_coarse.reshape(n, 1)
    out: dict[str, np.ndarray] = {}
    for key in ACTIVE_KEYS:
        pf = float(tbl_fine.get(key, 0.0))
        pc = float(tbl_coarse.get(key, 0.0))
        p = np.where(coarse, pc, pf)
        on = rng.random((n, 1)) < p
        if key == "free":
            per = np.exp(rng.uniform(np.log(6.0), np.log(max(L / 3.0, 12.0)), size=(n, 1)))
        else:
            per = cad.period(key)
        # a period must fit at least ~2.2 cycles in the window to be learnable
        on = on & (per >= 2.0) & (per <= L / 2.2)
        out[key] = np.where(on, per, 0.0)
    return out


def diurnal_period(cad: CadenceDraw, min_samples: float = 3.0) -> np.ndarray:
    """P_day where a diurnal cycle is resolvable, else 0."""
    pd = cad.p_day.reshape(-1, 1)
    return np.where(pd >= min_samples, pd, 0.0)


def multi_seasonal(rng: np.random.Generator, n: int, L: int, cad: CadenceDraw,
                   cfg: dict) -> np.ndarray:
    """Sum of shaped profiles over every period that is active for the row.

    Amplitudes are log-normal, so one component usually dominates; 45% of rows
    additionally get their seasonal amplitude modulated by a slow latent, which
    is what makes seasonality non-stationary in real feeds.
    """
    act = active_periods(rng, cad, cfg, L)
    out = np.zeros((n, L))
    any_on = np.zeros((n, 1), dtype=bool)
    for key in ACTIVE_KEYS:
        per = act[key]
        on = per > 0
        if not on.any():
            continue
        prof = seasonal_profile(rng, n, L, np.where(on, per, 1e9), cfg)
        amp = np.exp(rng.normal(0.0, 0.7, size=(n, 1)))
        out += np.where(on, amp * prof, 0.0)
        any_on |= on

    rate = float(cfg["seasonality"].get("amplitude_modulation_rate", 0.45))
    mod = rng.random((n, 1)) < rate
    if mod.any():
        slow = matern_gp(rng, n, L, np.exp(rng.uniform(
            np.log(L / 24.0), np.log(L / 3.0), n)), nu=1.5)
        env = np.clip(1.0 + 0.7 * slow, 0.05, 4.0)
        out = np.where(mod, out * env, out)

    fallback = unit_std(np.cumsum(rng.standard_normal((n, L)), axis=1))
    return np.where(any_on, unit_std(out), fallback)


# ═══════════════ chronoforge: cf_calendars ═══════════════
# Broadcast calendar arrays — no pandas, no Python loops.
#
# For every row we draw a starting day-of-week and day-of-year, then derive::
#
#     day_index[i, t] = floor(t / P_day[i])
#     dow[i, t]       = (start_dow[i] + day_index) % 7
#     doy[i, t]       = (start_doy[i] + day_index) % 365
#
# The calendar layer exists because the losing domains are administrative.  Real
# public-health and hospital feeds do not carry a +-0.12 additive log day-of-week
# offset: they carry a 2-5x *multiplicative* factor, a Monday catch-up spike,
# holiday collapses with a compensating spike on the next working day, batched
# multi-day releases, and revisions.  All of that lives here.


DAYS_IN_YEAR = 365
MAX_YEARS = 16


class Calendar:
    """Lazily materialised calendar arrays for one group of rows."""

    __slots__ = ("n", "L", "p_day", "start_dow", "start_doy", "_day_index",
                 "_dow", "_doy")

    def __init__(self, n: int, L: int, p_day: np.ndarray,
                 start_dow: np.ndarray, start_doy: np.ndarray):
        self.n = n
        self.L = L
        self.p_day = np.asarray(p_day, dtype=np.float64).reshape(n, 1)
        self.start_dow = np.asarray(start_dow, dtype=np.int64).reshape(n, 1)
        self.start_doy = np.asarray(start_doy, dtype=np.int64).reshape(n, 1)
        self._day_index = None
        self._dow = None
        self._doy = None

    @property
    def day_index(self) -> np.ndarray:
        if self._day_index is None:
            t = np.arange(self.L, dtype=np.float64)[None, :]
            self._day_index = np.floor(t / np.maximum(self.p_day, 1e-9)).astype(np.int64)
        return self._day_index

    @property
    def dow(self) -> np.ndarray:
        if self._dow is None:
            self._dow = (self.start_dow + self.day_index) % 7
        return self._dow

    @property
    def doy(self) -> np.ndarray:
        if self._doy is None:
            self._doy = (self.start_doy + self.day_index) % DAYS_IN_YEAR
        return self._doy

    @property
    def is_weekend(self) -> np.ndarray:
        return self.dow >= 5

    def take(self, rows: np.ndarray) -> "Calendar":
        return Calendar(len(rows), self.L, self.p_day[rows, 0],
                        self.start_dow[rows, 0], self.start_doy[rows, 0])


def draw_calendar(rng: np.random.Generator, n: int, L: int,
                  p_day: np.ndarray) -> Calendar:
    return Calendar(n, L, p_day,
                    rng.integers(0, 7, size=n),
                    rng.integers(0, DAYS_IN_YEAR, size=n))


# ───────────────────────── multiplicative factors ──────────────────────────

def dow_factors(rng: np.random.Generator, n: int, cfg: dict) -> np.ndarray:
    """Seven multiplicative day-of-week factors, geometric mean normalised to 1.

    Weekend ~ LogN(log 0.45, 0.35) -> a 0.2-0.9x weekend; Monday carries a
    catch-up multiplier.  These are an order of magnitude stronger than the
    additive log offsets the incumbent field uses, and they are what the data
    actually shows.
    """
    c = cfg["calendar"]
    f = np.empty((n, 7))
    f[:, 0] = np.exp(rng.normal(np.log(c["monday_factor"]), c["monday_sigma"], size=n))
    for d in (1, 2, 3, 4):
        f[:, d] = np.exp(rng.normal(0.0, c["weekday_sigma"], size=n))
    for d in (5, 6):
        f[:, d] = np.exp(rng.normal(np.log(c["weekend_factor"]),
                                    c["weekend_sigma"], size=n))
    # Sat != Sun
    f[:, 6] *= np.exp(rng.normal(0.0, 0.22, size=n))
    g = np.exp(np.mean(np.log(np.maximum(f, 1e-6)), axis=1, keepdims=True))
    return f / g


def apply_dow(fac: np.ndarray, cal: Calendar) -> np.ndarray:
    """Gather the 7 factors onto the (n, L) grid."""
    return np.take_along_axis(fac, cal.dow, axis=1)


def holiday_factor(rng: np.random.Generator, cal: Calendar, cfg: dict
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative holiday collapse plus a next-working-day compensation.

    Returns ``(factor, comp)`` on the (n, L) grid.  Holidays are 8-13 fixed
    day-of-year anchors plus 2-4 moving ones (an Easter-like offset from a drawn
    anchor), each with a +-1 day shoulder.
    """
    n, L = cal.n, cal.L
    c = cfg["calendar"]
    n_fixed = int(c["n_fixed_holidays"])
    n_moving = int(c["n_moving_holidays"])
    total = n_fixed + n_moving
    anchors = rng.integers(0, 365, size=(n, total))
    live = np.arange(total)[None, :] < rng.integers(
        c["holiday_count_lo"], c["holiday_count_hi"] + 1, size=(n, 1))
    depth = np.exp(rng.normal(np.log(c["holiday_factor"]), c["holiday_sigma"],
                              size=(n, total)))
    depth = np.clip(depth, 0.02, 1.0)

    doy = cal.doy
    # A daily-cadence window spans ~11 virtual years, so a *moving* feast really
    # does land on a different day-of-year each year.  Fixed anchors do not.
    year = np.minimum(cal.day_index // DAYS_IN_YEAR, MAX_YEARS - 1)
    year_shift = rng.integers(-25, 26, size=(n, MAX_YEARS))
    shift = np.take_along_axis(year_shift, year, axis=1)

    # Fixed anchors depend only on day-of-year, so they are resolved once on a
    # (n, 365) table and gathered; only the moving feasts need the full grid.
    grid = np.arange(DAYS_IN_YEAR, dtype=np.int64)[None, :]
    tab = np.ones((n, DAYS_IN_YEAR))
    tab_hit = np.zeros((n, DAYS_IN_YEAR), dtype=bool)
    for j in range(n_fixed):
        if not live[:, j].any():
            continue
        d = np.abs(((grid - anchors[:, j:j + 1] + 182) % 365) - 182)
        on = (d <= 1) & live[:, j:j + 1]
        f = np.where(d == 0, depth[:, j:j + 1], 0.5 * (1.0 + depth[:, j:j + 1]))
        tab = np.where(on, np.minimum(tab, f), tab)
        tab_hit |= on
    fac = np.take_along_axis(tab, doy, axis=1)
    hit = np.take_along_axis(tab_hit, doy, axis=1)

    for j in range(n_fixed, total):
        if not live[:, j].any():
            continue
        d = np.abs(((doy - (anchors[:, j:j + 1] + shift) + 182) % 365) - 182)
        on = (d <= 1) & live[:, j:j + 1]
        # shoulder days are milder than the holiday itself
        f = np.where(d == 0, depth[:, j:j + 1], 0.5 * (1.0 + depth[:, j:j + 1]))
        fac = np.where(on, np.minimum(fac, f), fac)
        hit |= on

    comp = np.zeros((n, L))
    if L > 1:
        # compensation lands on the first sample after the holiday block ends
        boundary = np.zeros((n, L), dtype=bool)
        boundary[:, 1:] = hit[:, :-1] & (~hit[:, 1:])
        amp = rng.uniform(c["holiday_comp_lo"], c["holiday_comp_hi"], size=(n, 1)) - 1.0
        span = np.maximum(cal.p_day, 1.0)
        comp = np.where(boundary, amp, 0.0)
        # spread the catch-up over one "day" of samples
        if float(np.max(span)) > 1.5:
            comp = decay_convolve(comp, np.maximum(span[:, 0] * 0.4, 1.0))
    return fac, comp


def month_boundary(cal: Calendar) -> np.ndarray:
    """True on the first sample of each 30/31-day month block."""
    m = (cal.doy // 30)
    b = np.zeros_like(m, dtype=bool)
    b[:, 1:] = m[:, 1:] != m[:, :-1]
    return b


def business_day_index(cal: Calendar) -> np.ndarray:
    """Cumulative count of business days, i.e. the business-day observation grid.

    Econ/fin daily feeds publish on business days only, so their effective weekly
    period is 5 rather than 7 and their release calendar advances only on
    weekdays.  Ubiquitous in the pool and modelled by nobody.
    """
    di = cal.day_index
    new_day = np.zeros_like(di, dtype=bool)
    new_day[:, 0] = True
    new_day[:, 1:] = di[:, 1:] != di[:, :-1]
    return np.cumsum((new_day & (~cal.is_weekend)).astype(np.int64), axis=1)


def dst_shift(rng: np.random.Generator, cal: Calendar, rate: float) -> np.ndarray:
    """A one-hour daily-phase jump at two day-of-year anchors (civil-time feeds).

    Returned in *cycles*: one hour of a 24-hour day is 1/24 of the daily phase,
    whatever the sampling cadence.

    Deliberately NOT applied to the meteorological families: that source requests
    UTC, so an ERA5 series has no DST discontinuity and a spurious one would be a
    mismatch rather than a prior.
    """
    n, L = cal.n, cal.L
    on = rng.random((n, 1)) < rate
    a1 = rng.integers(60, 120, size=(n, 1))
    a2 = rng.integers(270, 330, size=(n, 1))
    inside = (cal.doy >= a1) & (cal.doy < a2)
    return np.where(on & inside, 1.0 / 24.0, 0.0)


def batch_release(rng: np.random.Generator, x: np.ndarray, cal: Calendar,
                  period_days: np.ndarray) -> np.ndarray:
    """Accumulate over a k-day batch and release the whole sum on one sample.

    Everything between releases is an exact zero.  This is how a large fraction
    of public-health feeds actually report and it destroys naive persistence.
    """
    n, L = x.shape
    blk = (cal.day_index // np.maximum(period_days.reshape(n, 1), 1))
    boundary = np.zeros((n, L), dtype=bool)
    boundary[:, 1:] = blk[:, 1:] != blk[:, :-1]
    boundary[:, 0] = True
    cum = np.cumsum(x, axis=1)
    # A_t is the total accumulated strictly before t; S_t is A at the most recent
    # boundary at or before t.  Shifting S by one sample gives A at the PREVIOUS
    # boundary, so each release reports exactly the block that just closed.
    a = cum - x
    s = np.maximum.accumulate(np.where(boundary, a, 0.0), axis=1)
    prev = np.zeros_like(s)
    prev[:, 1:] = s[:, :-1]
    return np.where(boundary, a - prev, 0.0)


def revision_ramp(rng: np.random.Generator, x: np.ndarray, n_last: np.ndarray,
                  depth: np.ndarray) -> np.ndarray:
    """Systematically under-report the most recent k samples, ramping to truth."""
    n, L = x.shape
    k = np.maximum(np.asarray(n_last).reshape(n, 1), 1)
    age = (L - 1) - np.arange(L)[None, :]
    w = np.clip(1.0 - age / k, 0.0, 1.0)
    d = np.asarray(depth).reshape(n, 1)
    return x * (1.0 - w * d)


# ═══════════════ chronoforge: cf_observe ═══════════════
# Observation layer, scale/offset, structured aggregation, sanitiser.
#
# One shared, family-gated stage runs after every family has produced its rows.
# The stages are applied in a fixed order and each one is gated by the per-row
# capability flags the family returned, so a bounded process never gets an outlier
# spike outside its bounds and an integer count is never rounded onto a
# non-integer tick.  Every stage operates on the *selected rows only* — the gate
# is an index set, not a mask over a full-batch computation.
#
# Two deliberate omissions relative to the competitive field:
#
# * **no global time reversal.**  It is applied there to symmetric families; the
#   gain is marginal and it mis-teaches causality on anything with an asymmetric
#   response, which is most of what we generate.
# * **no tail-concentrated regime break.**  Injecting breaks into the last
#   64-1024 samples of a fixed fraction of series teaches a positional artefact
#   over patch index and inflates predictive width everywhere.  Our breaks carry a
#   uniform hazard modulated by *observable* volatility precursors instead.
#
# The rounding stage is the third departure, and the most consequential: it fires
# only when the series' standard deviation is at least ``round_min_ticks`` ticks.
# Blind rounding of unit-scale continuous families is what collapses a nominal
# AR(2)/GP/1-f corpus onto a three-to-five-level staircase.


MAX_PEAK = 1.0e12


def _rows(mask):
    return np.nonzero(mask)[0]


# ═════════════════════════════ observation layer ═══════════════════════════

def apply_observation(rng, y, flags, cfg):
    n, L = y.shape
    o = cfg["observation"]
    boost = flags["obs_boost"]
    integer = flags["integer"]
    bounded = flags["bounded"]
    positive = flags["positive"]

    def gate(rate):
        return rng.random(n) < np.clip(rate * boost, 0.0, 1.0)

    # ── 1. block aggregation ────────────────────────────────────────────────
    # Real feeds are frequently a rolling-window statistic of a finer signal: a
    # max-aggregate has an extreme-value marginal, a mean-aggregate a smoothed
    # one, and neither looks like the underlying process.  We aggregate causally
    # at constant length so the emitted series stays a multiple of 32 samples.
    lo_b = y.min(axis=1, keepdims=True)
    hi_b = y.max(axis=1, keepdims=True)

    sel = gate(o["block_aggregation_rate"])
    ks = rng.integers(2, 13, n)
    modes = rng.integers(0, 4, n)
    modes = np.where(integer & (modes == 0), 1, modes)   # mean breaks a lattice
    modes = np.where(bounded & (modes == 1), 2, modes)   # sum breaks a hard bound
    if sel.any():
        for k in range(2, 13):
            for mode in range(4):
                idx = _rows(sel & (ks == k) & (modes == mode))
                if idx.size:
                    y[idx] = rolling_agg(y[idx], k, mode)

    # ── 2. quantisation to a round human tick ───────────────────────────────
    qb = flags["quant_boost"]
    idx = _rows((rng.random(n) < np.clip(o["quantise_rate"] * qb, 0.0, 1.0))
                & (~integer))
    if idx.size:
        m = idx.size
        sub = y[idx]
        std = safe_std(sub)
        tick = human_tick(std[:, 0], rng, m, 1e-3, flags["quant_rel_hi"][idx])
        q = np.round(sub / tick) * tick
        lg_sel = rng.random(m) < o["log_grid_share"]
        if lg_sel.any():
            j = _rows(lg_sel)
            step = np.maximum(logu(rng, 0.01, 0.25, (j.size, 1)), 1e-6)
            base = np.maximum(np.abs(sub[j]), 1e-300)
            q[j] = np.sign(sub[j]) * np.exp(np.round(np.log(base) / step) * step)
        y[idx] = q
        flags["integer"][idx] = True

    # ── 3. one-sided censoring at a round value ─────────────────────────────
    idx = _rows(gate(o["censor_rate"]))
    if idx.size:
        m = idx.size
        sub = y[idx]
        std = safe_std(sub)
        mean = sub.mean(axis=1, keepdims=True)
        up = rng.random((m, 1)) < 0.5
        lvl = mean + np.where(up, 1.0, -1.0) * logu(rng, 0.8, 2.6, (m, 1)) * std
        tick = human_tick(std[:, 0], rng, m, 0.05, 0.5)
        # an integer feed is clipped at an integer, not at an arbitrary tick
        tick = np.where(integer[idx].reshape(m, 1), np.maximum(np.round(tick), 1.0), tick)
        lvl = np.round(lvl / tick) * tick
        y[idx] = np.where(up, np.minimum(sub, lvl), np.maximum(sub, lvl))

    # ── 4. staleness (bursty last-observation-carried-forward) ──────────────
    idx = _rows(gate(o["staleness_rate"]))
    if idx.size:
        m = idx.size
        gap = bursty_gaps(rng, m, L, logu(rng, 1e-4, 4e-3, (m, 1)),
                          logu(rng, 2.0, 60.0, (m, 1)))
        y[idx] = locf(y[idx], gap)

    # ── 5. missing-as-zero ──────────────────────────────────────────────────
    idx = _rows(gate(o["missing_zero_rate"]) & (~bounded))
    if idx.size:
        m = idx.size
        gap = bursty_gaps(rng, m, L, logu(rng, 1e-4, 3e-3, (m, 1)),
                          logu(rng, 2.0, 40.0, (m, 1)))
        sub = y[idx]
        sub[gap] = 0.0
        y[idx] = sub

    # ── 6. isolated outliers ────────────────────────────────────────────────
    idx = _rows(gate(o["outlier_rate"]) & (~bounded))
    if idx.size:
        m = idx.size
        sub = y[idx]
        thin = sub[:, ::8]
        med = np.median(thin, axis=1, keepdims=True)
        mad = np.maximum(np.median(np.abs(thin - med), axis=1, keepdims=True), 1e-12)
        E = 5
        pos = rng.integers(0, L, size=(m, E))
        live = np.arange(E)[None, :] < rng.integers(1, E + 1, size=(m, 1))
        sign = np.where(rng.random((m, 1)) < 0.5, 1.0, -1.0)
        amp = logu(rng, 6.0, 60.0, (m, E)) * mad * sign
        amp = np.where(integer[idx].reshape(m, 1), np.round(amp), amp)
        rows = np.broadcast_to(np.arange(m)[:, None], (m, E))
        np.add.at(sub, (rows[live], pos[live]), amp[live])
        y[idx] = sub

    # ── 7. instrument drift with abrupt recalibration ───────────────────────
    idx = _rows(gate(o["drift_recal_rate"]) & (~bounded) & (~integer))
    if idx.size:
        m = idx.size
        M = 8
        seg = rng.integers(1, L, size=(m, M))
        rows = np.broadcast_to(np.arange(m, dtype=np.int64)[:, None], (m, M))
        mark = np.bincount((rows * L + seg).ravel(), minlength=m * L).reshape(m, L)
        t = np.arange(L, dtype=np.float64)[None, :]
        start = np.maximum.accumulate(np.where(mark > 0, t, 0.0), axis=1)
        slope = rng.normal(0.0, 1.0, size=(m, 1)) * logu(rng, 1e-5, 3e-3, (m, 1))
        sub = y[idx]
        std = safe_std(sub)
        bias = slope * (t - start) * std
        mult = rng.random((m, 1)) < 0.5
        y[idx] = np.where(mult,
                          sub * (1.0 + np.clip(bias / std, -0.9, 3.0)),
                          sub + bias)

    # ── 8. scale-guarded rounding ───────────────────────────────────────────
    idx = _rows(gate(o["round_rate"]) & (~integer) & (~bounded))
    if idx.size:
        m = idx.size
        sub = y[idx]
        std = safe_std(sub)
        tick = human_tick(std[:, 0], rng, m, 1e-4, 1.0 / o["round_min_ticks"])
        # a share of these land on the literal integer lattice, which is what a
        # natural-unit feed (people, packets, requests) actually looks like
        unit = (rng.random((m, 1)) < o["integer_tick_share"]) & \
            (std >= o["round_min_ticks"])
        tick = np.where(unit, 1.0, tick)
        ok = (std >= (o["round_min_ticks"] * tick))
        y[idx] = np.where(ok, np.round(sub / tick) * tick, sub)
        flags["integer"][idx] = flags["integer"][idx] | ok[:, 0]

    idx = _rows(positive)
    if idx.size:
        np.maximum(y[idx], 0.0, out=y[idx])
    # A family that declared hard bounds keeps them: the artefact stages must not
    # move probability mass off a boundary we deliberately put there.
    idx = _rows(bounded)
    if idx.size:
        y[idx] = np.clip(y[idx], lo_b[idx], hi_b[idx])
    return y


# ═══════════════════════════ scale and offset ══════════════════════════════

def apply_scale(rng, y, flags, cfg):
    """Give every row a raw physical scale and offset.

    The trainer's causal scaler removes absolute level and scale, so this is not
    about magnitude for its own sake.  It is about the three things that *do*
    survive that transform: the integer lattice relative to the running standard
    deviation, the outlier-to-scale ratio under arcsinh, and how ``loc``/``scale``
    themselves evolve.  The large-offset regime is deliberately over-represented
    because a tiny relative variance on a large level is the small-MASE-
    denominator regime, and the sign-crossing regime is over-represented because
    a small sum|y| is what makes WQL a relative-error amplifier.
    """
    n, L = y.shape
    s = cfg["scale"]
    comps = s["log10_scale_mixture"]
    ws = np.array([c["weight"] for c in comps], dtype=np.float64)
    mus = np.array([c["mean"] for c in comps], dtype=np.float64)
    sds = np.array([c["sigma"] for c in comps], dtype=np.float64)
    k = np.searchsorted(np.cumsum(ws / ws.sum()), rng.random(n), side="right")
    k = np.clip(k, 0, len(ws) - 1)
    log10s = mus[k] + sds[k] * rng.standard_normal(n)
    scale = np.power(10.0, np.clip(log10s, -11.0, 10.0)).reshape(n, 1)

    mode = np.where(flags["count"], 2, flags["scale_mode"])
    pref = flags["offset_pref"]

    r = rng.random(n)
    c0 = s["zero_anchor_share"]
    c1 = c0 + s["large_offset_share"]
    c2 = c1 + s["sign_cross_share"]
    reg = np.where(r < c0, 0, np.where(r < c1, 1, np.where(r < c2, 2, 3)))
    reg = np.where(pref == 1, 1, np.where(pref == 2, 0, np.where(pref == 3, 2, reg)))

    big = rng.uniform(s["large_offset_ratio"][0], s["large_offset_ratio"][1], n)
    off = np.where(reg == 0, 0.0,
                   np.where(reg == 1, big,
                            np.where(reg == 2, rng.standard_normal(n),
                                     -np.abs(rng.standard_normal(n)) * big)))
    # hard invariant: |offset| / scale <= 1e7, so the fluctuation always keeps
    # at least nine significant digits of float64 headroom
    off = np.clip(off, -1.0e7, 1.0e7).reshape(n, 1)

    idx = _rows(mode == 0)
    if idx.size:
        sub = y[idx]
        sub = sub - sub.mean(axis=1, keepdims=True)
        sub = sub / safe_std(sub)
        y[idx] = off[idx] * scale[idx] + scale[idx] * sub
    idx = _rows(mode == 1)
    if idx.size:
        sub = y[idx]
        mag = np.maximum(np.abs(sub).mean(axis=1, keepdims=True), 1e-200)
        peak = np.maximum(np.abs(sub).max(axis=1, keepdims=True), 1e-200)
        mult = np.minimum(scale[idx] / mag, MAX_PEAK / peak)
        y[idx] = sub * mult
    # keep the batch inside the magnitude envelope before anything downstream
    # divides by a row statistic
    peak = np.abs(y).max(axis=1, keepdims=True)
    hot = _rows(peak[:, 0] > MAX_PEAK)
    if hot.size:
        y[hot] = y[hot] * (MAX_PEAK / peak[hot])
    return y


# ═════════════════════ structured hierarchical aggregation ═════════════════

def apply_aggregation(rng, y, flags, cad, cfg):
    """Build a fraction of rows as genuine aggregates of same-cadence siblings.

    A sum of K components driven by a common latent factor has a variance that
    grows like K^2, while independent components give K.  That signature is the
    correct model for national demand, grid totals, portfolio series and total
    pageviews, and it is what a blind cross-family Dirichlet mixup destroys:
    blending a Poisson count series with a chaotic attractor at a 1e4 scale ratio
    produces a noised copy of the larger, not a composite.  We only ever combine
    rows that share a cadence, and we restore the host row's own level and scale.
    """
    n, L = y.shape
    sel = ((rng.random(n) < cfg["aggregation"]["rate"])
           & flags["allow_agg"] & (~flags["count"]) & (~flags["bounded"]))
    rows = _rows(sel)
    if rows.size == 0:
        return y
    order = np.argsort(cad.seconds, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    partners = order[(rank[:, None] + np.arange(1, 7)[None, :]) % n]
    K = rng.integers(2, 7, n)
    beta = rng.uniform(0.3, 1.0, size=(n, 6))
    fshare = rng.uniform(0.2, 0.9, size=(n, 1))

    need = np.unique(np.concatenate([rows, partners[rows].ravel()]))
    z = np.zeros((n, L))
    zs = y[need]
    zs = zs - zs.mean(axis=1, keepdims=True)
    z[need] = zs / safe_std(zs)

    acc = np.zeros((rows.size, L))
    for j in range(6):
        live = (j < K[rows]) & (cad.seconds[partners[rows, j]] == cad.seconds[rows])
        if live.any():
            acc[live] += beta[rows[live], j:j + 1] * z[partners[rows[live], j]]
    factor = z[partners[rows, 0]]
    agg = (fshare[rows] * factor * np.sqrt(np.maximum(K[rows], 1)[:, None])
           + (1.0 - fshare[rows]) * acc)
    span = np.abs(agg).max(axis=1, keepdims=True)
    ok = agg.std(axis=1, keepdims=True) > 1e-9 * np.maximum(span, 1e-300)
    agg = np.where(ok, agg / safe_std(agg), 0.0)
    host = y[rows]
    blended = host.mean(axis=1, keepdims=True) + safe_std(host) * agg
    y[rows] = np.where(ok, blended, host)
    return y


# ═══════════════════════════════ sanitiser ═════════════════════════════════

def sanitise(rng, y, flags, starts, cfg):
    """Finiteness, degeneracy, cold-start and magnitude guards, in that order.

    ``starts`` gives each row's emit offset, so the constant-prefix guard is
    applied to the window the trainer will actually see: the causal scaler begins
    at the first emitted sample, and a constant lead-in drives its standard
    deviation to the 1e-5 floor and clamps the standardised input at +-64.
    """
    n, L = y.shape
    bad = _rows(~np.isfinite(y.sum(axis=1)))
    if bad.size:
        y[bad] = np.nan_to_num(y[bad], nan=0.0, posinf=MAX_PEAK, neginf=-MAX_PEAK)

    # degeneracy: regenerate from a cheap fallback rather than emit a flat row
    span = y.max(axis=1) - y.min(axis=1)
    ref = np.maximum(np.abs(y).mean(axis=1), 1e-300)
    dead = _rows((span <= 1e-12 * ref) | (span == 0.0))
    if dead.size:
        c = dead.size
        t = np.arange(L, dtype=np.float64)[None, :]
        phi = rng.uniform(0.5, 0.99, size=(c, 1))
        w = np.cumsum(rng.standard_normal((c, L)) * np.sqrt(1.0 - phi ** 2),
                      axis=1) * 0.05
        seas = np.sin(2.0 * np.pi * t / rng.uniform(12.0, 400.0, size=(c, 1)))
        lvl = y[dead].mean(axis=1, keepdims=True)
        base = np.where(np.abs(lvl) > 1e-300, lvl, 1.0)
        y[dead] = base * (1.0 + 0.05 * (w + seas))

    _break_constant_prefix(rng, y, flags, starts)

    peak = np.abs(y).max(axis=1, keepdims=True)
    hot = _rows(peak[:, 0] > MAX_PEAK)
    if hot.size:
        # rescale rather than clip: clipping flat-tops exactly the extreme
        # events we paid to generate
        y[hot] = y[hot] * (MAX_PEAK / peak[hot])
    return y


def _break_constant_prefix(rng, y, flags, starts, window=128):
    """No emitted window may open with ``window`` exactly constant samples.

    Constant series are structurally forbidden here.  They waste token budget and
    they are a degenerate input to the causal arcsinh scaler, so a cold start is
    always a real onset carrying at least tick-level jitter.
    """
    n, L = y.shape
    idx = np.minimum(starts[:, None] + np.arange(window)[None, :], L - 1)
    head = np.take_along_axis(y, idx, axis=1)
    rows = _rows(np.all(np.diff(head, axis=1) == 0.0, axis=1))
    if rows.size == 0:
        return
    m = rows.size
    std = y[rows].std(axis=1)
    lvl = np.maximum(np.abs(y[rows]).mean(axis=1), 1e-12)
    unit = np.where(flags["integer"][rows], 1.0,
                    np.maximum(std, lvl * 1e-4) * rng.uniform(0.05, 0.4, m))
    unit = np.where(unit > 0.0, unit, 1e-9)
    sgn = np.where(flags["positive"][rows], 1.0,
                   np.where(rng.random(m) < 0.5, 1.0, -1.0))
    for _ in range(3):
        pos = np.minimum(starts[rows] + rng.integers(4, window - 4, m), L - 1)
        y[rows, pos] = y[rows, pos] + unit * sgn


# ═════════════════════════════ count prior ═════════════════════════════════

def apply_count_prior(rng, y, flags, cfg):
    """Impose the eval pool's dominant signature on a share of rows.

    Measured on the revealed pool (block 8838000, 2,256 series): the median
    series is 100% integer-valued, p90 negativity is 0.00, and the median series
    has only ~9% distinct values.  Transport (the largest domain) is 68% exact
    repeats with ~16 levels; sales spans ~476.  A corpus of signed, continuous,
    high-entropy series teaches the wrong prior for most of the eval mass.

    For ``count_prior_rate`` of the non-bounded rows: shift so the minimum sits
    at a small positive floor (``[0, count_floor_frac] * std``), then for
    ``count_integer_share`` of those, quantise onto ``n_levels`` equally spaced
    integer levels with ``log10(n_levels)`` uniform on ``count_levels_log10``.
    Quantising to a *level count* rather than to 1.0 keeps the stage scale-free
    (the row has already been through the log10 scale mixture) and lets the
    corpus span the pool's whole resolution range instead of one point in it.
    Runs after scale and aggregation, on its own stream, so rate 0 leaves every
    byte untouched.
    """
    o = cfg["observation"]
    rate = float(o.get("count_prior_rate", 0.0))
    if rate <= 0.0:
        return y
    n, L = y.shape
    bounded = flags["bounded"]
    sel = _rows((rng.random(n) < rate) & (~bounded))
    if sel.size == 0:
        return y
    rows = y[sel]
    std = np.maximum(rows.std(axis=1, keepdims=True), 1e-12)
    lo = rows.min(axis=1, keepdims=True)
    floor = std * rng.uniform(0.0, float(o.get("count_floor_frac", 0.15)), size=(sel.size, 1))
    rows = rows + np.where(lo < floor, floor - lo, 0.0)
    flags["positive"][sel] = True
    ishare = float(o.get("count_integer_share", 0.86))
    imask = rng.random(sel.size) < ishare
    if imask.any():
        lg = o.get("count_levels_log10", [0.9, 2.7])
        nlev = 10.0 ** rng.uniform(float(lg[0]), float(lg[1]), size=(sel.size, 1))
        r = rows[imask]
        rlo = r.min(axis=1, keepdims=True)
        span = np.maximum(r.max(axis=1, keepdims=True) - rlo, 1e-12)
        tick = span / np.maximum(nlev[imask], 2.0)
        r = np.rint((r - rlo) / tick) + np.rint(rlo / tick)   # integers, >= 0
        rows[imask] = np.maximum(r, 0.0)
        ii = sel[imask]
        flags["integer"][ii] = True
    y[sel] = rows
    return y


# ═══════════════ chronoforge: cf_fam_met ═══════════════
# Block A — meteorological / geophysical families.
#
# Why this block is the largest: the evaluation pool's weather source emits
# roughly 252 global grid points x 12 ERA5 variables and attaches no ``source``
# metadata, so every weather row is its own bootstrap cluster.  That makes weather
# the dominant share of both eval windows and — far more importantly — of the
# *clusters* the confidence bound is resampled over.  A consistent weather win
# converts almost one-for-one into LCB.
#
# Composition of those twelve variables: three smooth thermal, two ultra-smooth
# pressure, four **bounded with genuine probability mass on a boundary** (three
# cloud-cover channels and relative humidity), and three non-negative
# right-skewed (wind at two heights plus gusts).  The bounded-with-atoms group is
# the single largest addressable block in the pool and is exactly what an
# unbounded symmetric predictive distribution wastes quantile mass on.


# ══════════════════════════ shared bounded latent ══════════════════════════

def bounded_latent(rng, n, L, ell, beta, diurnal, phi_ar, ar_frac):
    """Standardised latent for a hard-censored bounded process.

    Combines a synoptic Matern field, an optional diurnal term and AR(1)
    micro-structure, then standardises.  The *censoring* (not a logistic
    squash) is applied by the caller, which is what makes the boundary a
    genuine point mass rather than an asymptote.
    """
    z = matern_gp(rng, n, L, ell, nu=1.5)
    if diurnal is not None:
        z = z + beta * diurnal
    z = z + ar_frac * ar1(rng, n, L, phi_ar)
    return unit_std(z)


def censor(z, eta, c, upper):
    """y = U * clip(eta z + c, 0, 1) — hard censoring, so both bounds are atoms."""
    return upper * np.clip(eta * z + c, 0.0, 1.0)


# ══════════════════════════════ A1 met_thermal ═════════════════════════════

def a1_params(rng, B, cfg):
    p = cfg["families"]["met_thermal"]
    regime = categorical(rng, p["diurnal_amp_mix"], B)
    a_small = logu(rng, 1e-3, 0.05, B)
    a_mid = logu(rng, 0.1, 0.6, B)
    a_big = logu(rng, 0.6, 2.0, B)
    amp_d = np.where(regime == 0, a_small, np.where(regime == 1, a_mid, a_big))
    return {
        "ell_syn": logu(rng, p["synoptic_ell"][0], p["synoptic_ell"][1], B),
        "amp_syn": logu(rng, p["synoptic_amp"][0], p["synoptic_amp"][1], B),
        "amp_d": amp_d,
        "kappa": rng.uniform(0.15, 0.55, B),
        "rho": rng.uniform(p["cloud_coupling"][0], p["cloud_coupling"][1], B),
        "cloud_ell": logu(rng, 6.0, 200.0, B),
        "cloud_eta": logu(rng, 0.8, 3.0, B),
        "cloud_c": rng.normal(0.45, 0.35, B),
        "front_rate": rng.uniform(1.0, 4.0, B) / 1000.0,
        "front_ramp": logu(rng, 3.0, 24.0, B),
        "front_mag": rng.normal(0.0, 1.2, B),
        "phi_e": rng.uniform(0.2, 0.75, B),
        "sig_e": logu(rng, p["noise_ratio"][0], p["noise_ratio"][1], B),
        "annual": rng.random(B) < p["annual_rate"],
        "annual_amp": rng.normal(0.0, 1.0, B),
    }


def a1_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    p_day = cad.p_day.reshape(n, 1)

    syn = matern_gp(rng, n, L, P["ell_syn"], nu=1.5) * P["amp_syn"].reshape(n, 1)

    # cloudiness latent — the same machinery as A3, which is why the diurnal
    # amplitude envelope is physically correct rather than an arbitrary AM.
    cz = bounded_latent(rng, n, L, P["cloud_ell"], 0.0, None,
                        rng.uniform(0.3, 0.7, n), 0.25)
    cloud = np.clip(P["cloud_eta"].reshape(n, 1) * cz + P["cloud_c"].reshape(n, 1),
                    0.0, 1.0)

    has_diurnal = (p_day >= 3.0)
    per = np.where(has_diurnal, p_day, 1e9)
    ph = t / per + rng.random((n, 1))
    k = P["kappa"].reshape(n, 1)
    phw = ph + (k / _cf_prims_TWO_PI) * np.sin(_cf_prims_TWO_PI * ph)
    # asymmetric daily shape: fast morning rise, slow evening fall
    daily = np.sin(_cf_prims_TWO_PI * phw) + 0.25 * np.sin(2.0 * _cf_prims_TWO_PI * phw + 0.9)
    amp_t = (P["amp_d"].reshape(n, 1) * P["amp_syn"].reshape(n, 1)
             * (1.0 - P["rho"].reshape(n, 1) * cloud))
    diurnal = np.where(has_diurnal, daily * amp_t, 0.0)

    # frontal passages: smooth monotone ramps, not steps
    E = 6
    cnt = rng.poisson(np.maximum(P["front_rate"].reshape(n, 1) * L, 0.0), size=(n, E))
    pos = rng.integers(0, L, size=(n, E)).astype(np.float64)
    mag = rng.standard_normal((n, E)) * (P["front_mag"].reshape(n, 1)
                                         * P["amp_syn"].reshape(n, 1))
    width = np.maximum(P["front_ramp"].reshape(n, 1)
                       * np.exp(rng.normal(0.0, 0.4, size=(n, E))), 1.0)
    front = np.zeros((n, L))
    for j in range(E):
        idx = np.nonzero(cnt[:, j] > 0)[0]
        if idx.size == 0:
            continue
        u = (t - pos[idx, j:j + 1]) / width[idx, j:j + 1]
        front[idx] += mag[idx, j:j + 1] * 0.5 * (1.0 + u / (1.0 + np.abs(u)))

    eps = ar1(rng, n, L, P["phi_e"]) * (P["sig_e"].reshape(n, 1)
                                        * P["amp_syn"].reshape(n, 1))

    ann = np.where(P["annual"].reshape(n, 1),
                   P["annual_amp"].reshape(n, 1) * P["amp_syn"].reshape(n, 1)
                   * ((t / L) - 0.5) ** 2 * 4.0, 0.0)

    y = syn + diurnal + front + eps + ann
    fl = new_flags(n, scale_mode=0, quant_boost=0.7, quant_rel_hi=0.35)
    return y, fl


# ═══════════════════════ A2 met_pressure_smooth ════════════════════════════

def a2_params(rng, B, cfg):
    p = cfg["families"]["met_pressure_smooth"]
    return {
        "ell_syn": logu(rng, p["synoptic_ell"][0], p["synoptic_ell"][1], B),
        "tide_amp": rng.uniform(0.15, 0.45, B) * 0.06,
        "dip_on": rng.random(B) < p["dip_rate"],
        "dip_depth": rng.uniform(1.5, 4.0, B),
        "dip_width": logu(rng, 24.0, 180.0, B),
        "sig_e": logu(rng, p["noise_ratio"][0], p["noise_ratio"][1], B),
        "quant": rng.random(B) < p["quantise_rate"],
    }


def a2_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    p_day = cad.p_day.reshape(n, 1)

    # Matern-5/2 is twice mean-square differentiable: it has genuinely
    # extrapolable local curvature, which is the only thing a 64-step forecast
    # of a pressure field can exploit.
    syn = matern_gp(rng, n, L, P["ell_syn"], nu=2.5)

    has_day = p_day >= 4.0
    ph = t / np.where(has_day, p_day, 1e9) + rng.random((n, 1))
    tide = np.where(has_day,
                    P["tide_amp"].reshape(n, 1)
                    * (np.sin(2.0 * _cf_prims_TWO_PI * ph) + 0.5 * np.sin(_cf_prims_TWO_PI * ph + 1.1)),
                    0.0)

    E = 2
    dpos = rng.integers(0, L, size=(n, E)).astype(np.float64)
    dw = P["dip_width"].reshape(n, 1) * np.exp(rng.normal(0.0, 0.3, size=(n, E)))
    dip = np.zeros((n, L))
    live = P["dip_on"].reshape(n, 1) & (rng.random((n, E)) < 0.6)
    for j in range(E):
        idx = np.nonzero(live[:, j])[0]
        if idx.size == 0:
            continue
        dip[idx] -= P["dip_depth"][idx].reshape(-1, 1) * np.exp(
            -0.5 * ((t - dpos[idx, j:j + 1])
                    / np.maximum(dw[idx, j:j + 1], 1.0)) ** 2)

    eps = rng.standard_normal((n, L)) * P["sig_e"].reshape(n, 1)

    y = syn + tide + dip + eps
    # The defining property is a tiny relative variance riding on a large level:
    # that is the small-MASE-denominator regime, where any trend error explodes
    # log-MASE.  offset_pref = 1 forces the large-offset scale regime.
    fl = new_flags(n, scale_mode=0, offset_pref=1, quant_rel_hi=0.55)
    fl["quant_boost"] = np.where(P["quant"], 3.0, 0.4)
    return y, fl


# ═══════════════════════ A3 met_bounded_atom ═══════════════════════════════

UPPER_GRID = np.array([1.0, 8.0, 10.0, 100.0, 1000.0])


def a3_params(rng, B, cfg):
    p = cfg["families"]["met_bounded_atom"]
    variant = categorical(rng, p["variant_mix"], B)          # cloud / humidity / utilisation
    return {
        "variant": variant,
        "ell": logu(rng, p["latent_ell"][0], p["latent_ell"][1], B),
        "eta": logu(rng, p["eta"][0], p["eta"][1], B),
        "c": rng.normal(p["centre_mean"], p["centre_sigma"], B),
        "beta_cloud": rng.uniform(0.0, 0.25, B),
        "beta_hum": rng.uniform(0.4, 1.1, B),
        "phi_ar": rng.uniform(0.3, 0.85, B),
        "ar_frac": logu(rng, 0.05, 0.45, B),
        "upper": UPPER_GRID[categorical(rng, p["upper_mix"], B)],
        "quantise": rng.random(B) < p["quantise_rate"],
        "event_on": rng.random(B) < p["saturation_event_rate"],
        "event_mu": logu(rng, 2e-4, 4e-3, B),
        "event_branch": rng.uniform(0.2, 0.8, B),
        "event_tau": logu(rng, 20.0, 400.0, B),
        "event_len": logu(rng, 2.0, 30.0, B),
    }


def a3_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    p_day = cad.p_day.reshape(n, 1)
    variant = P["variant"]

    has_day = p_day >= 3.0
    ph = t / np.where(has_day, p_day, 1e9) + rng.random((n, 1))
    daily = np.sin(_cf_prims_TWO_PI * ph)
    biz = profile_trapezoid(rng, n, L, ph)
    biz = unit_std(biz)

    beta = np.where((variant == 0).reshape(n, 1), P["beta_cloud"].reshape(n, 1),
                    np.where((variant == 1).reshape(n, 1),
                             -P["beta_hum"].reshape(n, 1),  # humidity is anti-phase
                             0.0))
    diurnal = np.where((variant == 2).reshape(n, 1), biz, daily)
    diurnal = np.where(has_day, diurnal, 0.0)
    beta = np.where((variant == 2).reshape(n, 1), P["beta_hum"].reshape(n, 1) * 0.6, beta)

    z = bounded_latent(rng, n, L, P["ell"], beta, diurnal, P["phi_ar"],
                       P["ar_frac"].reshape(n, 1))
    upper = P["upper"].reshape(n, 1)
    y = censor(z, P["eta"].reshape(n, 1), P["c"].reshape(n, 1), upper)

    # clustered saturation events (rain / overcast spells) pin the ceiling
    on = P["event_on"]
    if on.any():
        mu = np.where(on.reshape(n, 1), P["event_mu"].reshape(n, 1), 0.0)
        _, cnt = hawkes(rng, n, L, np.broadcast_to(mu, (n, L)).copy(),
                        P["event_branch"], P["event_tau"])
        starts = np.argsort(-cnt, axis=1)[:, :6]
        lens = (P["event_len"].reshape(n, 1)
                * np.exp(rng.normal(0.0, 0.5, size=(n, 6))))
        live = np.take_along_axis(cnt, starts, axis=1) > 0
        mask = run_mask(n, L, starts, np.where(live, lens, 0.0))
        y = np.where(mask & on.reshape(n, 1), upper, y)

    q = P["quantise"].reshape(n, 1)
    step = np.where(upper <= 1.0, upper / 100.0, 1.0)
    y = np.where(q, np.round(y / step) * step, y)

    fl = new_flags(n, scale_mode=2, bounded=True, positive=True,
                   quant_boost=0.0, obs_boost=0.5, allow_agg=False)
    fl["integer"] = (P["quantise"] & (P["upper"] > 1.0))
    return y, fl


# ═══════════════════════════ A4 met_wind_speed ═════════════════════════════

def a4_params(rng, B, cfg):
    p = cfg["families"]["met_wind_speed"]
    return {
        "ell": logu(rng, p["component_ell"][0], p["component_ell"][1], B),
        "mean_wind": logu(rng, 0.2, 6.0, B),
        "gamma_mix": rng.uniform(-0.4, 0.8, B),
        "gust": rng.random(B) < p["gust_rate"],
        "gust_phi": rng.uniform(0.1, 0.5, B),
        "gust_sig": rng.uniform(0.35, 0.7, B),
        "drift_amp": rng.uniform(0.0, 1.2, B),
        "quantise": rng.random(B) < p["quantise_rate"],
        "quant_step": np.where(rng.random(B) < 0.5, 0.1, 1.0),
    }


def a4_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    p_day = cad.p_day.reshape(n, 1)

    u = matern_gp(rng, n, L, P["ell"], nu=1.5)
    v = matern_gp(rng, n, L, P["ell"], nu=1.5)

    has_day = p_day >= 3.0
    ph = t / np.where(has_day, p_day, 1e9) + rng.random((n, 1))
    prof = np.where(has_day, np.sin(_cf_prims_TWO_PI * ph - 1.2), 0.0)
    sig = 1.0 + P["gamma_mix"].reshape(n, 1) * 0.5 * prof
    sig = np.maximum(sig, 0.15)

    drift = matern_gp(rng, n, L, np.full(n, L / 2.0), nu=1.5) \
        * P["drift_amp"].reshape(n, 1)
    mw = P["mean_wind"].reshape(n, 1)
    # Rice/Weibull marginal with a genuine hard floor at zero
    s = np.sqrt((u * sig + drift) ** 2 + (v * sig) ** 2) * mw

    g = P["gust"].reshape(n, 1)
    lg = ar1(rng, n, L, P["gust_phi"]) * P["gust_sig"].reshape(n, 1) + np.log(0.35)
    s = np.where(g, s * (1.0 + np.exp(np.clip(lg, -20.0, 6.0))), s)

    q = P["quantise"].reshape(n, 1)
    step = P["quant_step"].reshape(n, 1)
    s = np.where(q, np.round(s / step) * step, s)

    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=0.5,
                   quant_rel_hi=0.4)
    return s, fl


# ═══════════════════════ A5 wet_dry_intermittent ═══════════════════════════

def a5_params(rng, B, cfg):
    p = cfg["families"]["wet_dry_intermittent"]
    return {
        "d_dry": logu(rng, p["dry_dwell"][0], p["dry_dwell"][1], B),
        "d_wet": logu(rng, p["wet_dwell"][0], p["wet_dwell"][1], B),
        "k_int": rng.uniform(p["intensity_shape"][0], p["intensity_shape"][1], B),
        "theta": logu(rng, 0.2, 20.0, B),
        "start_wet": rng.random(B) < 0.15,
        "bell": rng.random(B) < 0.7,
    }


def a5_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    M = 96
    dry = np.exp(rng.normal(np.log(P["d_dry"]).reshape(n, 1), 1.1, size=(n, M)))
    wet = np.exp(rng.normal(np.log(P["d_wet"]).reshape(n, 1), 0.8, size=(n, M)))
    dwell = np.empty((n, M))
    sw = P["start_wet"].reshape(n, 1)
    even = (np.arange(M)[None, :] % 2) == 0
    dwell = np.where(even ^ sw, dry, wet)
    dwell = np.maximum(dwell, 1.0)

    seg_id, seg_start = segment_map(dwell, L)
    seg_len = gather_levels(dwell, seg_id)
    is_wet = ((seg_id % 2) == 1) ^ sw

    t = time_grid(L)[None, :]
    pos = (t - seg_start) / np.maximum(seg_len, 1.0)
    env = np.where(P["bell"].reshape(n, 1), np.sin(np.pi * np.clip(pos, 0.0, 1.0)), 1.0)

    k = np.maximum(P["k_int"].reshape(n, 1), 1e-2)
    inten = rng.gamma(np.broadcast_to(k, (n, L))) * P["theta"].reshape(n, 1)
    y = np.where(is_wet, inten * env, 0.0)

    extended = np.zeros(n, dtype=bool)
    if cfg.get('_complete_native_renewals', False):
        med = np.column_stack((np.where(P['start_wet'], P['d_wet'], P['d_dry']),
                               np.where(P['start_wet'], P['d_dry'], P['d_wet'])))
        sig = np.column_stack((np.where(P['start_wet'], 0.8, 1.1),
                               np.where(P['start_wet'], 1.1, 0.8)))
        rr, dd, first = complete_dwell_table(rng, dwell, L, med, sig, 1.0)
        if rr.size:
            extended[rr] = True
            sid, start = segment_map(dd, L)
            slen = gather_levels(dd, sid)
            wet_now = ((sid % 2) == 1) ^ sw[rr]
            progress = (t - start) / np.maximum(slen, 1.0)
            env_now = np.where(P['bell'][rr, None], np.sin(np.pi * np.clip(progress, 0.0, 1.0)), 1.0)
            fixed = np.where(wet_now, inten[rr] * env_now, 0.0)
            y[rr] = np.where(t >= first[:, None], fixed, y[rr])

    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=1.2,
                   quant_rel_hi=0.3, allow_agg=True)
    if cfg.get('_complete_native_renewals', False):
        fl['renewal_extended'] = extended
    return y, fl


# ═══════════════ chronoforge: cf_fam_ops ═══════════════
# Block B — operational telemetry / web-cloudops families.
#
# The web-cloudops domain is one of the two the field loses.  The diagnosis is
# structural: these shapes are produced by *control systems*, and every family in
# the competitive field is open-loop.  An autoscaling sawtooth, a rate-limit
# plateau, a queue backlog, a deploy overshoot-and-settle and a counter reset are
# all closed-loop artefacts.  A model trained only on open-loop processes treats a
# sawtooth as a staircase plus noise and cannot anticipate the next scale-out.


def _daily_phase(rng, n, L, p_day, min_res=3.0):
    t = time_grid(L)[None, :]
    pd = np.asarray(p_day).reshape(n, 1)
    has = pd >= min_res
    return np.where(has, t / np.where(has, pd, 1e9), 0.0) + rng.random((n, 1)), has


# ═══════════════════════ B1 ops_diurnal_traffic ════════════════════════════

def b1_params(rng, B, cfg):
    p = cfg["families"]["ops_diurnal_traffic"]
    return {
        "base": logu(rng, 1.0, 5.0e4, B),
        "theta_taylor": rng.uniform(p["taylor_exponent"][0], p["taylor_exponent"][1], B),
        "cv": logu(rng, 0.03, 0.6, B),
        "burst_on": rng.random(B) < p["burst_rate"],
        "branch": rng.uniform(0.2, 0.85, B),
        "burst_tau": logu(rng, 3.0, 120.0, B),
        "mark_sig": rng.uniform(0.6, 1.6, B),
        "burst_mu": logu(rng, 1e-4, 5e-3, B),
        "count_emit": rng.random(B) < p["count_emit_rate"],
        "nb_k": logu(rng, 0.5, 200.0, B),
        "weekend": np.exp(rng.normal(np.log(0.42), 0.5, B)),
        "sat_sun": np.exp(rng.normal(0.0, 0.25, B)),
        "trend": rng.normal(0.0, 0.25, B),
        "growth_on": rng.random(B) < 0.45,
        "dst": rng.random(B) < p["dst_rate"],
    }


def b1_demand(P, rng, L, cad, cal, cfg):
    """Shared demand process — also the driver for B2's controller."""
    n = cad.n
    t = time_grid(L)[None, :]
    ph, has_day = _daily_phase(rng, n, L, cad.p_day)
    # Civil-time operational feeds shift by an hour twice a year; the weather
    # families deliberately do NOT, because that source requests UTC.
    ph = ph + np.where(P["dst"].reshape(n, 1), dst_shift(rng, cal, 1.0), 0.0)
    prof = profile_trapezoid(rng, n, L, ph)
    prof = np.where(has_day, prof, 0.0)
    prof = prof - prof.min(axis=1, keepdims=True)
    m = np.maximum(prof.mean(axis=1, keepdims=True), 1e-9)
    shape = 0.25 + 0.75 * prof / m

    wk = np.ones((n, 7))
    wk[:, 5] = P["weekend"]
    wk[:, 6] = P["weekend"] * P["sat_sun"]
    weekly = apply_dow(wk, cal)

    trend = np.where(P["growth_on"].reshape(n, 1),
                     np.exp(P["trend"].reshape(n, 1) * t / L), 1.0)

    lam = P["base"].reshape(n, 1) * shape * weekly * trend

    # bursts arrive when traffic is high: immigrant rate proportional to lambda
    mu = (P["burst_mu"].reshape(n, 1) * shape
          * np.where(P["burst_on"].reshape(n, 1), 1.0, 0.0))
    _, cnt = hawkes(rng, n, L, np.ascontiguousarray(mu), P["branch"], P["burst_tau"])
    marks = cnt * np.exp(rng.normal(0.0, P["mark_sig"].reshape(n, 1), size=(n, L)))
    burst = decay_convolve(marks, P["burst_tau"])
    lam = lam * (1.0 + 2.5 * burst)
    return np.maximum(lam, 0.0)


def b1_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    lam = b1_demand(P, rng, L, cad, cal, cfg)
    # Taylor's law heteroscedasticity: sigma proportional to lambda^theta
    z = rng.standard_normal((n, L))
    sig = P["cv"].reshape(n, 1) * np.power(np.maximum(lam, 1e-12),
                                           P["theta_taylor"].reshape(n, 1))
    y = np.maximum(lam + sig * z, 0.0)
    ci = P["count_emit"].reshape(n, 1)
    counts = nb_counts(rng, lam, P["nb_k"])
    y = np.where(ci, counts, y)
    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=0.8)
    fl["integer"] = P["count_emit"]
    return y, fl


# ════════════════════ B2 ops_saturating_feedback ═══════════════════════════

def b2_params(rng, B, cfg):
    p = cfg["families"]["ops_saturating_feedback"]
    q = b1_params(rng, B, cfg)
    q.update({
        "th_up": rng.uniform(0.6, 0.9, B),
        "th_dn": rng.uniform(0.15, 0.45, B),
        "gam": rng.uniform(p["step_gain"][0], p["step_gain"][1], B),
        "k_up": rng.integers(3, 61, B),
        "k_dn": rng.integers(3, 61, B),
        "lag": np.round(logu(rng, 2.0, 20.0, B)).astype(np.int64),
        "mode": categorical(rng, p["emit_mix"], B),
        "u_scale": np.array([1.0, 100.0])[categorical(rng, [0.35, 0.65], B)],
    })
    return q


def b2_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    d = b1_demand(P, rng, L, cad, cal, cfg)
    c0 = np.maximum(d[:, :64].mean(axis=1) / 0.7, 1e-6)
    out = np.zeros((n, L))
    cap = np.zeros((n, L))
    k_feedback(np.ascontiguousarray(d),
                 np.ascontiguousarray(P["th_up"]),
                 np.ascontiguousarray(P["th_dn"]),
                 np.ascontiguousarray(P["k_up"]).astype(np.int64),
                 np.ascontiguousarray(P["k_dn"]).astype(np.int64),
                 np.ascontiguousarray(P["gam"]),
                 np.ascontiguousarray(P["lag"]).astype(np.int64),
                 c0, np.ascontiguousarray(P["mode"]).astype(np.int64), out, cap)
    mode = P["mode"].reshape(n, 1)
    y = np.where(mode == 0, out * P["u_scale"].reshape(n, 1), out)
    fl = new_flags(n, scale_mode=2, positive=True, obs_boost=0.8)
    fl["bounded"] = (P["mode"] == 0)
    fl["scale_mode"] = np.where(P["mode"] == 0, 2, 1).astype(np.int8)
    return y, fl


# ═══════════════════════ B3 ops_counter_reset ══════════════════════════════

def b3_params(rng, B, cfg):
    p = cfg["families"]["ops_counter_reset"]
    return {
        "rate": logu(rng, 1.0, 4.0e3, B),
        "nb_k": logu(rng, 0.4, 60.0, B),
        "reset_mode": categorical(rng, p["reset_mix"], B),
        "reset_tau": logu(rng, 400.0, 4000.0, B),
        "cal_week": rng.random(B) < 0.35,
        "diurnal_amp": rng.uniform(0.0, 0.9, B),
        "revise": rng.random(B) < p["revision_rate"],
        "revise_k": rng.integers(2, 9, B),
        "revise_depth": rng.uniform(0.02, 0.3, B),
        "integer": rng.random(B) < 0.55,
    }


def b3_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    ph, has_day = _daily_phase(rng, n, L, cad.p_day)
    prof = 1.0 + P["diurnal_amp"].reshape(n, 1) * np.where(has_day, np.sin(_cf_prims_TWO_PI * ph), 0.0)
    mean_inc = P["rate"].reshape(n, 1) * np.maximum(prof, 0.05)
    inc = nb_counts(rng, mean_inc, P["nb_k"])
    cont = np.maximum(mean_inc * rng.gamma(np.broadcast_to(
        np.maximum(P["nb_k"].reshape(n, 1), 1e-2), (n, L))) /
        np.maximum(P["nb_k"].reshape(n, 1), 1e-2), 0.0)
    inc = np.where(P["integer"].reshape(n, 1), inc, cont)

    mode = P["reset_mode"].reshape(n, 1)
    poisson_reset = rng.random((n, L)) < (1.0 / P["reset_tau"].reshape(n, 1))
    day_idx = cal.day_index
    period = np.where(P["cal_week"].reshape(n, 1), 7, 1)
    blk = day_idx // np.maximum(period, 1)
    cal_reset = np.zeros((n, L), dtype=bool)
    cal_reset[:, 1:] = blk[:, 1:] != blk[:, :-1]
    reset = np.where(mode == 0, poisson_reset,
                     np.where(mode == 1, cal_reset, False))
    out = np.zeros((n, L))
    k_counter_reset_cal(np.ascontiguousarray(inc),
                          np.ascontiguousarray(reset).astype(np.int8), out)

    # revisions: the last few published points are corrected downward
    if P["revise"].any():
        age = (L - 1) - np.arange(L)[None, :]
        w = np.clip(1.0 - age / np.maximum(P["revise_k"].reshape(n, 1), 1), 0.0, 1.0)
        out = np.where(P["revise"].reshape(n, 1),
                       out * (1.0 - w * P["revise_depth"].reshape(n, 1)), out)

    fl = new_flags(n, scale_mode=2, positive=True, obs_boost=0.5, quant_boost=0.3)
    fl["integer"] = P["integer"] & (~P["revise"])
    return out, fl


# ═══════════════════════ B4 ops_latency_queue ══════════════════════════════

def b4_params(rng, B, cfg):
    p = cfg["families"]["ops_latency_queue"]
    return {
        "rho_mu": rng.normal(-0.8, 1.0, B),
        "rho_phi": rng.uniform(0.85, 0.999, B),
        "rho_sig": rng.uniform(0.2, 1.2, B),
        "service": logu(rng, 1e-3, 100.0, B),
        "lognorm": rng.random(B) < p["lognormal_rate"],
        "ln_sig": rng.uniform(0.4, 1.3, B),
        "pareto_a": rng.uniform(1.6, 3.5, B),
        "agg": rng.random(B) < p["percentile_rate"],
        "agg_m": rng.integers(8, 201, B),
    }


def b4_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    lr = ar1(rng, n, L, P["rho_phi"]) * P["rho_sig"].reshape(n, 1) \
        + P["rho_mu"].reshape(n, 1)
    rho = 1.0 / (1.0 + np.exp(-np.clip(lr, -30.0, 30.0)))
    rho = np.clip(rho, 0.0, 0.9995)
    mean = P["service"].reshape(n, 1) * (1.0 + rho / (1.0 - rho))
    mean = np.minimum(mean, P["service"].reshape(n, 1) * 1e4)

    ln = np.exp(rng.normal(0.0, P["ln_sig"].reshape(n, 1), size=(n, L))
                - 0.5 * P["ln_sig"].reshape(n, 1) ** 2)
    u = np.maximum(rng.random((n, L)), 1e-12)
    a = P["pareto_a"].reshape(n, 1)
    par = np.power(u, -1.0 / a) * (a - 1.0) / a
    obs = np.where(P["lognorm"].reshape(n, 1), ln, par)
    y = mean * obs

    # a percentile aggregate has a Gumbel-shaped marginal, quite different from
    # the mean series it is computed from
    if P["agg"].any():
        m = np.maximum(P["agg_m"].reshape(n, 1).astype(np.float64), 2.0)
        um = np.maximum(rng.random((n, L)), 1e-12)
        gmax = mean * np.where(P["lognorm"].reshape(n, 1),
                               np.exp(P["ln_sig"].reshape(n, 1)
                                      * np.sqrt(2.0 * np.log(m))),
                               np.power(np.power(um, 1.0 / m), -1.0 / a))
        y = np.where(P["agg"].reshape(n, 1), gmax, y)

    if _CAS1_WORKLOAD_ENABLED:
        y = _cas1_workload_response(P, rho, obs, y,
            np.random.Generator(rng.bit_generator.jumped()))

    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=0.9)
    return y, fl


# ═══════════════════════ B5 ops_deploy_transient ═══════════════════════════

def b5_params(rng, B, cfg):
    p = cfg["families"]["ops_deploy_transient"]
    return {
        "phi": rng.uniform(0.6, 0.995, B),
        "n_ev": rng.integers(2, 8, B),
        "shift_sig": rng.uniform(0.5, 3.0, B),
        "trans_amp": rng.uniform(1.5, 4.0, B),
        "trans_tau": logu(rng, 8.0, 300.0, B),
        "osc": rng.random(B) < 0.45,
        "zeta": rng.uniform(0.1, 0.6, B),
        "osc_period": logu(rng, 10.0, 200.0, B),
        "var_switch": rng.random(B) < 0.35,
        "n_out": rng.integers(1, 5, B),
        "out_len": logu(rng, 4.0, 300.0, B),
        "out_mode": categorical(rng, p["outage_mix"], B),
    }


def b5_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    base = ar1(rng, n, L, P["phi"])

    E = 8
    pos = rng.integers(int(0.02 * L), L, size=(n, E)).astype(np.float64)
    live = np.arange(E)[None, :] < P["n_ev"].reshape(n, 1)
    delta = rng.standard_normal((n, E)) * P["shift_sig"].reshape(n, 1)
    lvl = np.zeros((n, L))
    trans = np.zeros((n, L))
    for j in range(E):
        idx = np.nonzero(live[:, j])[0]
        if idx.size == 0:
            continue
        p0 = pos[idx, j:j + 1]
        after = (t >= p0)
        d = delta[idx, j:j + 1]
        lvl[idx] += np.where(after, d, 0.0)
        dt = np.maximum(t - p0, 0.0)
        amp = P["trans_amp"][idx].reshape(-1, 1) * np.abs(d)
        expo = amp * np.exp(-dt / P["trans_tau"][idx].reshape(-1, 1))
        w = _cf_prims_TWO_PI / np.maximum(P["osc_period"][idx].reshape(-1, 1), 2.0)
        zt = P["zeta"][idx].reshape(-1, 1)
        dosc = amp * np.exp(-np.minimum(zt * w * dt, 60.0)) * np.cos(w * dt)
        pick = np.where(P["osc"][idx].reshape(-1, 1), dosc, expo)
        trans[idx] += np.where(after, pick * np.sign(d), 0.0)

    var_mult = np.ones((n, L))
    if P["var_switch"].any():
        mid = pos[:, :1]
        var_mult = np.where(P["var_switch"].reshape(n, 1) & (t >= mid),
                            np.exp(rng.normal(0.0, 0.8, size=(n, 1))), 1.0)

    y = base * var_mult + lvl + trans

    # outages: hold-last / exact zero / linear backfill
    starts = rng.integers(0, L, size=(n, 4))
    lens = P["out_len"].reshape(n, 1) * np.exp(rng.normal(0.0, 0.5, size=(n, 4)))
    live_o = np.arange(4)[None, :] < P["n_out"].reshape(n, 1)
    mask = run_mask(n, L, starts, np.where(live_o, lens, 0.0))
    held = locf(y, mask)
    idx = np.where(mask, -1, np.arange(L)[None, :])
    prev = np.maximum(np.maximum.accumulate(idx, axis=1), 0)
    nxt = np.where(mask, L, np.arange(L)[None, :])
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    nxt = np.minimum(nxt, L - 1)
    va = np.take_along_axis(y, prev, axis=1)
    vb = np.take_along_axis(y, nxt, axis=1)
    span = np.maximum(nxt - prev, 1)
    w = (np.arange(L)[None, :] - prev) / span
    lin = va + (vb - va) * w
    om = P["out_mode"].reshape(n, 1)
    y = np.where(mask, np.where(om == 0, held, np.where(om == 1, 0.0, lin)), y)

    fl = new_flags(n, scale_mode=0, quant_boost=1.0)
    return y, fl


# ═══════════════════════ B6 ops_rate_plateau ═══════════════════════════════

_LADDER_MANT = np.array([1.0, 2.0, 5.0])


def b6_params(rng, B, cfg):
    p = cfg["families"]["ops_rate_plateau"]
    n_lvl = rng.integers(3, 6, B)
    exp0 = rng.integers(-2, 6, B)
    return {
        "n_lvl": n_lvl,
        "exp0": exp0,
        "mant": _LADDER_MANT[rng.integers(0, 3, (B, 5))],
        "expo": rng.integers(0, 2, (B, 5)),
        "switch_tau": logu(rng, p["plateau_dwell"][0], p["plateau_dwell"][1], B),
        "phi": rng.uniform(0.8, 0.995, B),
        "demand_cv": rng.uniform(0.15, 0.8, B),
        "diurnal_amp": rng.uniform(0.1, 1.0, B),
    }


def b6_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    ph, has_day = _daily_phase(rng, n, L, cad.p_day)
    prof = 1.0 + P["diurnal_amp"].reshape(n, 1) * np.where(has_day, np.sin(_cf_prims_TWO_PI * ph), 0.0)
    demand = np.maximum(prof, 0.05) * np.exp(
        ar1(rng, n, L, P["phi"]) * P["demand_cv"].reshape(n, 1))

    M = 64
    dwell = np.exp(rng.normal(np.log(P["switch_tau"]).reshape(n, 1), 0.8, size=(n, M)))
    seg_id, _ = segment_map(np.maximum(dwell, 2.0), L)
    ladder = P["mant"] * np.power(10.0, P["expo"] + P["exp0"].reshape(n, 1))
    ladder = ladder / np.maximum(ladder.mean(axis=1, keepdims=True), 1e-12)
    pick = rng.integers(0, np.maximum(P["n_lvl"].reshape(n, 1), 1), size=(n, M))
    caps = np.take_along_axis(ladder, np.clip(pick, 0, 4), axis=1)
    cap_t = gather_levels(caps, seg_id)

    y = np.minimum(demand, cap_t)
    extended = np.zeros(n, dtype=bool)
    if cfg.get('_complete_native_renewals', False):
        rr, dd, first = complete_dwell_table(
            rng, np.maximum(dwell, 2.0), L, P['switch_tau'].reshape(n, 1),
            np.full((n, 1), 0.8), 2.0)
        if rr.size:
            extended[rr] = True
            extra_pick = rng.integers(0, np.maximum(P['n_lvl'][rr, None], 1),
                                      size=(rr.size, dd.shape[1] - M))
            extra_caps = np.take_along_axis(ladder[rr], np.clip(extra_pick, 0, 4), axis=1)
            full_caps = np.concatenate((caps[rr], extra_caps), axis=1)
            sid, _ = segment_map(dd, L)
            fixed = np.minimum(demand[rr], gather_levels(full_caps, sid))
            t = time_grid(L)[None, :]
            y[rr] = np.where(t >= first[:, None], fixed, y[rr])

    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=0.4, obs_boost=0.7)
    if cfg.get('_complete_native_renewals', False):
        fl['renewal_extended'] = extended
    return y, fl


from numba import njit as _cas1_njit

@_cas1_njit(cache=False, fastmath=False, nogil=True)
def _cas1_lindley_kernel(service, gap, initial):
    n, length = service.shape
    waits = np.empty_like(service)
    for i in range(n):
        if length == 0:
            continue
        waits[i, 0] = initial[i]
        for t in range(1, length):
            work = waits[i, t-1] + service[i, t-1] - gap[i, t]
            waits[i, t] = max(work, 0.)
    return waits


def _cas1_workload_waits(service, gap, initial):
    """gap[:,t] precedes job t; gap[:,0] is unused. +inf clears the queue."""
    b, a = (np.asarray(x, dtype=np.float64) for x in (service, gap))
    w = np.asarray(initial, dtype=np.float64)
    if b.ndim != 2 or a.shape != b.shape or w.shape != (len(b),):
        raise ValueError('Matched service/gap matrices and an initial-work vector required')
    if (not np.isfinite(b).all() or not np.isfinite(w).all() or np.isnan(a).any()
            or (b < 0).any() or (a < 0).any() or (w < 0).any()):
        raise ValueError('Finite nonnegative service/work and nonnegative gaps required')
    out = _cas1_lindley_kernel(b, a, w)
    if not np.isfinite(out).all():
        raise ValueError('Workload overflow; do not silently clip')
    return out


def _cas1_initial_workload(rng, rho, lognormal, sigma, shape, service):
    """Ideal constant-rho M/G/1 initial workload, before subsequent time variation."""
    rho, sigma, shape, service = (np.asarray(x, dtype=np.float64)
                                 for x in (rho, sigma, shape, service))
    ln = np.asarray(lognormal)
    if rho.ndim != 1 or any(x.shape != rho.shape for x in (sigma, shape, service, ln)):
        raise ValueError('Matching parameter vectors required')
    if (ln.dtype.kind != 'b' or any(not np.isfinite(x).all() for x in (rho, sigma, shape, service))
            or ((rho < 0) | (rho >= 1)).any() or (sigma < 0).any()
            or (shape <= 1).any() or (service <= 0).any()):
        raise ValueError('Invalid stable queue/service parameters')
    count = rng.geometric(1.-rho) - 1
    out = np.zeros(len(rho), dtype=np.float64)
    for i in range(len(rho)):
        number = int(count[i])
        if number == 0:
            continue
        if ln[i]:
            biased = service[i] * np.exp(.5*sigma[i]**2 + sigma[i]*rng.standard_normal(number))
        else:
            lower = service[i]*(shape[i]-1.)/shape[i]
            biased = lower*(1.+rng.pareto(shape[i]-1., size=number))
        out[i] = np.sum(rng.random(number)*biased)
    if not np.isfinite(out).all():
        raise ValueError('Initial workload overflow; do not truncate its tail')
    return out


def _cas1_workload_response(P, rho, obs, parent, rng):
    """Only non-aggregate rows change; all old draws precede this child stream."""
    rows = np.flatnonzero(~P['agg'])
    out = parent.copy()
    if rows.size == 0 or parent.shape[1] == 0:
        return out
    service = P['service'][rows]
    initial = _cas1_initial_workload(rng, rho[rows, 0], P['lognorm'][rows],
                                    P['ln_sig'][rows], P['pareto_a'][rows], service)
    durations = service[:, None]*obs[rows]
    utilization = rho[rows]
    gap = np.full_like(durations, np.inf)
    exponential = rng.exponential(size=durations.shape)
    np.divide(service[:, None]*exponential, utilization, out=gap, where=utilization > 0.)
    out[rows] = _cas1_workload_waits(durations, gap, initial) + durations
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite response time')
    return out


_CAS1_WORKLOAD_ENABLED = True


# ═══════════════ chronoforge: cf_fam_health ═══════════════
# Block C — healthcare / epidemiological / administrative families.
#
# Healthcare is the worst domain for the incumbent field.  The failure is almost
# certainly *reporting structure*, not dynamics: public-health and hospital-admin
# feeds carry weekend factors of 0.2-0.9x, Monday catch-up spikes of 1.3-2.4x,
# batched multi-day releases with exact zeros in between, systematic revisions of
# the most recent points, holiday collapses with compensation, and genuinely
# under-dispersed counts.  A +-0.12 additive log day-of-week offset — which is
# what the field ships — is an order of magnitude too weak.
#
# C1 adds the dynamics half: a real renewal equation, so the 64-step forecast
# depends on whether R_t has crossed 1.  That is the actual forecasting question
# for those feeds and it is not representable by a piecewise-linear log trend.


MAX_TAPS = 64


# ═════════════════════════════ C1 epi_renewal ══════════════════════════════

def c1_params(rng, B, cfg):
    p = cfg["families"]["epi_renewal"]
    q = {
        "r_tau": logu(rng, p["logr_corr_time"][0], p["logr_corr_time"][1], B),
        "r_sig": np.abs(rng.normal(0.0, p["logr_sigma"], B)) + 0.05,
        "r_mu": rng.normal(0.0, 0.12, B),
        "mu_g": logu(rng, p["serial_interval_mean"][0], p["serial_interval_mean"][1], B),
        "cv_g": rng.uniform(0.35, 0.8, B),
        "disp": logu(rng, p["dispersion"][0], p["dispersion"][1], B),
        "import_tau": logu(rng, 500.0, 4000.0, B),
        "import_size": logu(rng, 1.0, 200.0, B),
        "endemic": rng.random(B) < p["endemic_rate"],
        "endemic_rate": logu(rng, 0.05, 6.0, B),
        "i0": logu(rng, 1.0, 500.0, B),
        "report": rng.random(B) < p["reporting_layer_rate"],
    }
    q.update(c2_params(rng, B, cfg))
    return q


def _serial_interval(mu_g, cv_g, n):
    """Discretised Gamma serial-interval kernel, truncated at 4 mu_g."""
    a = 1.0 / np.maximum(cv_g.reshape(n, 1) ** 2, 1e-3)
    scale = np.maximum(mu_g.reshape(n, 1), 1.0) / a
    s = np.arange(MAX_TAPS, dtype=np.float64)[None, :] + 0.5
    logw = (a - 1.0) * np.log(s) - s / scale
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    trunc = np.minimum(np.ceil(4.0 * mu_g.reshape(n, 1)), MAX_TAPS)
    w = np.where(np.arange(MAX_TAPS)[None, :] < trunc, w, 0.0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    taps = np.clip(trunc[:, 0].astype(np.int64), 1, MAX_TAPS)
    return np.ascontiguousarray(w), taps


def c1_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    logr = ou(rng, n, L, P["r_tau"]) * P["r_sig"].reshape(n, 1) \
        + P["r_mu"].reshape(n, 1)
    rt = np.exp(np.clip(logr, -3.0, 2.0))

    w, taps = _serial_interval(P["mu_g"], P["cv_g"], n)
    imp = np.where(rng.random((n, L)) < (1.0 / P["import_tau"].reshape(n, 1)),
                   P["import_size"].reshape(n, 1), 0.0)
    # A pure renewal process with heavy over-dispersion has zero as an absorbing
    # state, so most windows would be extinct.  Real notifiable-disease feeds
    # carry a background importation rate; endemic rows get one, and the rest
    # keep the sporadic-introduction behaviour.
    imp = imp + np.where(P["endemic"].reshape(n, 1),
                         P["endemic_rate"].reshape(n, 1), 0.0)

    k = np.maximum(P["disp"].reshape(n, 1), 1e-2)
    gam = rng.gamma(np.broadcast_to(k, (n, L))) / k
    u = rng.random((n, L))
    z = rng.standard_normal((n, L))
    out = np.zeros((n, L))
    k_renewal(w, taps, np.ascontiguousarray(rt), imp, gam, u, z,
                np.ascontiguousarray(P["i0"]), out)

    rep = P["report"].reshape(n, 1)
    if P["report"].any():
        obs = _reporting_layer(P, rng, out, cal, cfg)
        out = np.where(rep, obs, out)

    fl = new_flags(n, scale_mode=2, positive=True, obs_boost=0.6, quant_boost=0.2)
    fl["integer"] = (
        ~(P["report"] & P["revise_on"])
        if cfg.get("_native_integer_flags", False) else np.ones(n, dtype=bool)
    )
    return out, fl


# ═══════════════════ C2 admin_reporting_counts ═════════════════════════════

def c2_params(rng, B, cfg):
    p = cfg["families"]["admin_reporting_counts"]
    return {
        "base": logu(rng, 3.0, 2.0e5, B),
        "smooth_ell": logu(rng, 30.0, 1500.0, B),
        "smooth_amp": rng.uniform(0.1, 0.9, B),
        "batch_on": rng.random(B) < p["batch_rate"],
        "batch_days": rng.integers(2, 8, B),
        "revise_on": rng.random(B) < p["revision_rate"],
        "revise_k": rng.integers(3, 13, B),
        "revise_depth": rng.uniform(0.05, 0.5, B),
        "emit": categorical(rng, p["emission_mix"], B),
        "nb_k": logu(rng, 0.3, 100.0, B),
        "under_phi": rng.uniform(0.05, 0.9, B),
        "hol_on": rng.random(B) < p["holiday_rate"],
        "trend": rng.normal(0.0, 0.35, B),
        "month_end": rng.random(B) < p["month_end_rate"],
        "month_end_lift": rng.uniform(1.2, 3.0, B),
    }


def _reporting_layer(P, rng, latent, cal, cfg):
    """Multiplicative calendar + emission, applied to a non-negative latent."""
    n, L = latent.shape
    fac = dow_factors(rng, n, cfg)
    dow = apply_dow(fac, cal)
    # holidays only for the rows that use them: the moving-feast masks and the
    # compensation convolve are full-(rows, L) work, so computing them for
    # every row and then masking was most of this family's cost.
    hol_rows = np.nonzero(P["hol_on"])[0]
    hol = np.ones((n, L))
    comp = np.zeros((n, L))
    if hol_rows.size:
        h_sub, c_sub = holiday_factor(rng, cal.take(hol_rows), cfg)
        hol[hol_rows] = h_sub
        comp[hol_rows] = c_sub
    mu = np.maximum(latent * dow * hol * (1.0 + comp), 0.0)
    if P["month_end"].any():
        lift = np.where(month_boundary(cal),
                        P["month_end_lift"].reshape(n, 1), 1.0)
        mu = np.where(P["month_end"].reshape(n, 1), mu * lift, mu)

    # one count model per row, sampled only on the rows that select it (the
    # three full-grid draws + a where() were three times the sampling cost)
    emit = np.asarray(P["emit"]).reshape(n)
    y = np.zeros((n, L))
    r_nb = np.nonzero(emit == 0)[0]
    r_po = np.nonzero(emit == 1)[0]
    r_bi = np.nonzero(emit >= 2)[0]
    if r_nb.size:
        y[r_nb] = nb_counts(rng, mu[r_nb], P["nb_k"][r_nb])
    if r_po.size:
        y[r_po] = rng.poisson(np.minimum(mu[r_po], 1e8)).astype(np.float64)
    if r_bi.size:
        phi = P["under_phi"][r_bi].reshape(-1, 1)
        ntr = np.maximum(np.round(mu[r_bi] / np.maximum(1.0 - phi, 1e-3)), 0.0)
        ntr = np.minimum(ntr, 1e7)
        # under-dispersed counts: var = mu * phi < mu.  Exists nowhere in the field.
        y[r_bi] = rng.binomial(ntr.astype(np.int64),
                               np.broadcast_to(1.0 - phi, (r_bi.size, L))).astype(np.float64)

    if P["batch_on"].any():
        rel = batch_release(rng, y, cal, P["batch_days"])
        y = np.where(P["batch_on"].reshape(n, 1), rel, y)
    if P["revise_on"].any():
        rv = revision_ramp(rng, y, P["revise_k"], P["revise_depth"])
        y = np.where(P["revise_on"].reshape(n, 1), rv, y)
    return y


def c2_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    smooth = matern_gp(rng, n, L, P["smooth_ell"], nu=1.5)
    latent = P["base"].reshape(n, 1) * np.exp(
        P["smooth_amp"].reshape(n, 1) * smooth + P["trend"].reshape(n, 1) * t / L)
    y = _reporting_layer(P, rng, latent, cal, cfg)
    fl = new_flags(n, scale_mode=2, positive=True, obs_boost=0.5, quant_boost=0.15)
    fl["integer"] = (~P["revise_on"])
    return y, fl


# ═══════════════════ C3 physio_quasiperiodic ═══════════════════════════════

TEMPLATE_S = 256


def c3_params(rng, B, cfg):
    p = cfg["families"]["physio_quasiperiodic"]
    return {
        "p0": logu(rng, p["base_period"][0], p["base_period"][1], B),
        "resp_ratio": rng.uniform(3.0, 6.0, B),
        "rsa": rng.uniform(p["rsa_depth"][0], p["rsa_depth"][1], B),
        "hrv": rng.uniform(0.005, 0.08, B),
        "biphasic": rng.random(B) < 0.45,
        "n_harm": rng.integers(3, 7, B),
        "peak_sharp": rng.uniform(1.0, 4.0, B),
        "amp_resp": rng.uniform(0.05, 0.45, B),
        "wander": rng.uniform(0.1, 1.5, B),
        "artefact_p": rng.uniform(0.005, 0.05, B),
        "artefact_amp": rng.uniform(5.0, 40.0, B),
        "flat_tau": logu(rng, 600.0, 5000.0, B),
        "flat_len": logu(rng, 10.0, 200.0, B),
        "noise": logu(rng, 0.005, 0.15, B),
    }


def c3_build(P, rng, L, cad, cal, cfg):
    n = cad.n

    # waveform template: sharp systolic peak, slow recovery (or biphasic)
    s = np.arange(TEMPLATE_S, dtype=np.float64)[None, :] / TEMPLATE_S
    tmpl = np.zeros((n, TEMPLATE_S))
    sharp = P["peak_sharp"].reshape(n, 1)
    nh = P["n_harm"].reshape(n, 1)
    for h in range(1, 7):
        amp = (h ** (-sharp)) * np.where(h <= nh, 1.0, 0.0)
        psi = (h - 1) * 0.35
        tmpl += amp * np.cos(_cf_prims_TWO_PI * h * s - psi)
    bip = P["biphasic"].reshape(n, 1)
    tmpl = np.where(bip, tmpl - 0.6 * np.roll(tmpl, TEMPLATE_S // 6, axis=1), tmpl)
    tmpl = unit_std(tmpl)

    # instantaneous period: respiratory sinus arrhythmia + 1/f variability
    resp_p = P["p0"] * P["resp_ratio"]
    t = time_grid(L)[None, :]
    resp = np.sin(_cf_prims_TWO_PI * t / resp_p.reshape(n, 1) + rng.random((n, 1)) * _cf_prims_TWO_PI)
    hrv = pink_gp(rng, n, L, np.full(n, 1.2))
    per = P["p0"].reshape(n, 1) * (1.0 + P["rsa"].reshape(n, 1) * resp
                                   + P["hrv"].reshape(n, 1) * hrv)
    per = np.maximum(per, 2.0)
    phase = np.cumsum(1.0 / per, axis=1) + rng.random((n, 1))

    y = np.zeros((n, L))
    k_template(np.ascontiguousarray(phase), np.ascontiguousarray(tmpl),
                 np.arange(n, dtype=np.int64), y)
    y = y * (1.0 + P["amp_resp"].reshape(n, 1) * resp)

    y = y + P["wander"].reshape(n, 1) * pink_gp(rng, n, L, np.full(n, 2.0))
    y = y + rng.standard_normal((n, L)) * P["noise"].reshape(n, 1)

    art = rng.random((n, L)) < P["artefact_p"].reshape(n, 1)
    y = np.where(art, y + rng.standard_normal((n, L)) * P["artefact_amp"].reshape(n, 1), y)

    # electrode-off flatlines
    starts = rng.integers(0, L, size=(n, 4))
    hit = rng.random((n, 4)) < (L / P["flat_tau"].reshape(n, 1) / 4.0)
    lens = P["flat_len"].reshape(n, 1) * np.exp(rng.normal(0.0, 0.6, size=(n, 4)))
    mask = run_mask(n, L, starts, np.where(hit, lens, 0.0))
    y = np.where(mask, 0.0, y)

    fl = new_flags(n, scale_mode=0, quant_boost=0.8)
    return y, fl


# ═══════════════════ C4 clinical_bounded_vitals ════════════════════════════

VITAL_RANGES = np.array([
    [70.0, 100.0],    # SpO2
    [35.0, 190.0],    # heart rate
    [34.5, 41.5],     # core temperature
    [50.0, 200.0],    # blood pressure
])


def c4_params(rng, B, cfg):
    p = cfg["families"]["clinical_bounded_vitals"]
    idx = categorical(rng, p["range_mix"], B)
    return {
        "range_idx": idx,
        "lo": VITAL_RANGES[idx, 0],
        "hi": VITAL_RANGES[idx, 1],
        "tau": logu(rng, 60.0, 2000.0, B),
        "bias": rng.uniform(p["ceiling_bias"][0], p["ceiling_bias"][1], B),
        "spread": logu(rng, 0.3, 3.0, B),
        "ev_tau": logu(rng, p["excursion_tau"][0], p["excursion_tau"][1], B),
        "fall": logu(rng, 1.0, 8.0, B),
        "recover": logu(rng, 6.0, 80.0, B),
        "depth": logu(rng, 0.02, 0.5, B),
        "tick": np.where(rng.random(B) < 0.6, 1.0, 0.1),
        "noise": logu(rng, 1e-3, 0.03, B),
    }


def c4_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    lo = P["lo"].reshape(n, 1)
    hi = P["hi"].reshape(n, 1)
    span = hi - lo

    slow = ou(rng, n, L, P["tau"]) * P["spread"].reshape(n, 1) \
        + P["bias"].reshape(n, 1)
    base = lo + span * np.clip(1.0 / (1.0 + np.exp(-np.clip(slow, -30.0, 30.0))), 0.0, 1.0)

    # asymmetric excursions: fast fall, slow recovery.  A symmetric predictive
    # distribution cannot represent this, and the ceiling pinning makes the
    # lag-m MASE denominator tiny, so a missed desaturation is catastrophic.
    E = 8
    cnt = rng.random((n, E)) < (L / (P["ev_tau"].reshape(n, 1) * E))
    pos = rng.integers(0, L, size=(n, E)).astype(np.float64)
    exc = np.zeros((n, L))
    for j in range(E):
        idx = np.nonzero(cnt[:, j])[0]
        if idx.size == 0:
            continue
        p0 = pos[idx, j:j + 1]
        dt = np.maximum(t - p0, 0.0)
        tr = P["recover"][idx].reshape(-1, 1)
        tf = np.minimum(P["fall"][idx].reshape(-1, 1), tr * 0.9)
        shape = np.exp(-dt / tr) - np.exp(-dt / tf)
        shape = shape / np.maximum(shape.max(axis=1, keepdims=True), 1e-9)
        exc[idx] -= np.where(t >= p0,
                             P["depth"][idx].reshape(-1, 1) * span[idx] * shape, 0.0)

    y = base + exc + rng.standard_normal((n, L)) * P["noise"].reshape(n, 1) * span
    y = np.clip(y, lo, hi)
    tick = P["tick"].reshape(n, 1)
    y = np.round(y / tick) * tick

    fl = new_flags(n, scale_mode=2, bounded=True, positive=True,
                   quant_boost=0.0, obs_boost=0.4, allow_agg=False)
    fl["integer"] = (P["tick"] == 1.0)
    return y, fl


# ═══════════════ chronoforge: cf_fam_regime ═══════════════
# Block D — persistence / regime core.
#
# This is the incumbent prior, and it is here to protect against regression on a
# warm-started checkpoint rather than to differentiate.  We match it in *effect*
# and improve its statistics in three specific ways:
#
# * dwell times are heavy-tailed (LogN mixed with a Pareto tail) instead of
#   uniform — uniform dwell teaches a wrong hazard function;
# * there is a third regime type, ``transitional``: a smooth monotone ramp
#   between levels, because real regime changes are frequently gradual;
# * the *variance* switches with the regime, not only the level.
#
# D2's staircase snaps 40% of its levels onto a recurring ``{1,2,5}x10^k`` tick
# ladder, so successive levels sit on the same grid.  That is a real, learnable
# regularity (policy rates, price ladders, config values, thermostat setpoints)
# which an arbitrary-real staircase destroys.


DORMANT_FLAT = 0
DORMANT_MICRO = 1
DORMANT_COUNTER = 2
DORMANT_ZERO = 3
DORMANT_TRANS = 4

N_ACTIVE_KINDS = 6


# ───────────────────────── active-regime base menu ─────────────────────────

def active_base(rng, n, L, kind, cad, cfg):
    """Unit-std active-regime carrier; each kind is built for its rows only."""
    out = np.zeros((n, L))
    for k in range(N_ACTIVE_KINDS):
        m = kind == k
        cnt = int(m.sum())
        if cnt == 0:
            continue
        if k == 0:
            sub = multi_seasonal(rng, cnt, L, cad.take(np.nonzero(m)[0]), cfg)
        elif k == 1:
            phi = np.stack([rng.uniform(0.1, 1.35, cnt),
                            rng.uniform(-0.75, 0.15, cnt)], axis=1)
            # keep the AR(2) inside the stationarity triangle
            phi[:, 1] = np.minimum(phi[:, 1], 0.98 - np.abs(phi[:, 0]))
            e = rng.standard_normal((cnt, L))
            sub = np.zeros((cnt, L))
            k_arma(np.ascontiguousarray(phi), np.full(cnt, 2, dtype=np.int64),
                     np.zeros((cnt, 1)), np.zeros(cnt, dtype=np.int64), e, sub)
            sub = unit_std(sub)
        elif k == 2:
            sub = matern_gp(rng, cnt, L, logu(rng, 8.0, 800.0, cnt),
                            nu=float(rng.choice(np.array([0.5, 1.5, 2.5]))))
        elif k == 3:
            sub = unit_std(np.cumsum(rng.standard_normal((cnt, L)), axis=1))
        elif k == 4:
            mu = np.broadcast_to(logu(rng, 1e-3, 5e-2, (cnt, 1)), (cnt, L))
            _, c = hawkes(rng, cnt, L, np.ascontiguousarray(mu),
                          rng.uniform(0.2, 0.9, cnt), logu(rng, 3.0, 120.0, cnt))
            marks = c * np.exp(rng.normal(0.0, 0.8, size=(cnt, L)))
            sub = unit_std(decay_convolve(marks, logu(rng, 2.0, 60.0, cnt)))
        else:
            sub = ou(rng, cnt, L, logu(rng, 4.0, 600.0, cnt))
        out[m] = sub
    return out


# ═════════════════════════════ D1 regime_dwell ═════════════════════════════

def d1_params(rng, B, cfg):
    p = cfg["families"]["regime_dwell"]
    return {
        "d_dorm": logu(rng, p["dormant_dwell"][0], p["dormant_dwell"][1], B),
        "d_act": logu(rng, p["active_dwell"][0], p["active_dwell"][1], B),
        "dwell_sigma": rng.uniform(0.6, 1.3, B),
        "pareto_a": rng.uniform(1.2, 2.0, B),
        "heavy_tail": rng.random(B) < p["pareto_mix"],
        "kind": rng.integers(0, N_ACTIVE_KINDS, B),
        "amp": logu(rng, 0.3, 12.0, B),
        "level_step": logu(rng, 0.05, 3.0, B),
        "var_switch": logu(rng, 1.0, 8.0, B),
        "start_active": rng.random(B) < 0.35,
        "micro_slope": rng.normal(0.0, 0.02, B),
        "counter_step": logu(rng, 1.0, 50.0, B),
        "sparse_p": logu(rng, 1e-4, 2e-2, B),
        "sparse_amp": logu(rng, 0.5, 20.0, B),
        "ramp_frac": rng.uniform(0.05, 0.15, B),
        "noise": logu(rng, 1e-3, 0.2, B),
    }


def _heavy_dwell(rng, n, M, mean, sigma, pareto_a, heavy):
    ln = np.exp(rng.normal(np.log(np.maximum(mean, 1.0)).reshape(n, 1),
                           sigma.reshape(n, 1), size=(n, M)))
    u = np.maximum(rng.random((n, M)), 1e-9)
    par = np.maximum(mean, 1.0).reshape(n, 1) * np.power(u, -1.0 / pareto_a.reshape(n, 1))
    take_par = (rng.random((n, M)) < 0.15) & heavy.reshape(n, 1)
    return np.maximum(np.where(take_par, par, ln), 2.0)


def d1_build_factory(mode):
    def build(P, rng, L, cad, cal, cfg):
        n = cad.n
        M = 64
        dorm = _heavy_dwell(rng, n, M, P["d_dorm"], P["dwell_sigma"],
                            P["pareto_a"], P["heavy_tail"])
        act = _heavy_dwell(rng, n, M, P["d_act"], P["dwell_sigma"],
                           P["pareto_a"], P["heavy_tail"])
        sa = P["start_active"].reshape(n, 1)
        even = (np.arange(M)[None, :] % 2) == 0
        dwell = np.where(even ^ sa, dorm, act)

        seg_id, seg_start = segment_map(dwell, L)
        seg_len = gather_levels(dwell, seg_id)
        is_active = ((seg_id % 2) == 1) ^ sa

        # segment levels follow a random walk; the variance switches too
        steps = rng.standard_normal((n, M)) * P["level_step"].reshape(n, 1)
        levels = np.cumsum(steps, axis=1)
        lvl_t = gather_levels(levels, seg_id)
        prev_lvl = gather_levels(np.concatenate(
            [levels[:, :1], levels[:, :-1]], axis=1), seg_id)

        t = time_grid(L)[None, :]
        pos = (t - seg_start) / np.maximum(seg_len, 1.0)

        base = active_base(rng, n, L, P["kind"], cad, cfg) * P["amp"].reshape(n, 1)
        var_hi = P["var_switch"].reshape(n, 1)
        carrier = np.where(is_active, base * var_hi, base / var_hi)

        if mode == DORMANT_FLAT:
            dormant = lvl_t
        elif mode == DORMANT_MICRO:
            dormant = lvl_t + P["micro_slope"].reshape(n, 1) \
                * P["level_step"].reshape(n, 1) * (t - seg_start)
        elif mode == DORMANT_COUNTER:
            step = P["counter_step"].reshape(n, 1)
            dormant = np.round(lvl_t / step) * step
        elif mode == DORMANT_ZERO:
            spike = np.where(rng.random((n, L)) < P["sparse_p"].reshape(n, 1),
                             rng.standard_normal((n, L)) * P["sparse_amp"].reshape(n, 1),
                             0.0)
            dormant = np.abs(spike)
        else:  # DORMANT_TRANS — a smooth monotone ramp between levels
            w = np.clip(pos / np.maximum(P["ramp_frac"].reshape(n, 1), 1e-3), 0.0, 1.0)
            w = 0.5 * (1.0 + np.tanh(6.0 * (w - 0.5)))
            dormant = prev_lvl + (lvl_t - prev_lvl) * w

        noise = rng.standard_normal((n, L)) * P["noise"].reshape(n, 1) \
            * P["amp"].reshape(n, 1)
        y = np.where(is_active, lvl_t + carrier + noise, dormant)
        if mode == DORMANT_ZERO:
            y = np.where(is_active, np.abs(carrier) + noise * 0.0, dormant)

        extended = np.zeros(n, dtype=bool)
        if cfg.get('_complete_native_renewals', False):
            med = np.column_stack((np.where(P['start_active'], P['d_act'], P['d_dorm']),
                                   np.where(P['start_active'], P['d_dorm'], P['d_act'])))
            rr, dd, first = complete_dwell_table(
                rng, dwell, L, med, P['dwell_sigma'].reshape(n, 1), 2.0,
                P['heavy_tail'], P['pareto_a'])
            if rr.size:
                extended[rr] = True
                extra = dd.shape[1] - M
                add = np.cumsum(rng.standard_normal((rr.size, extra))
                                * P['level_step'][rr, None], axis=1)
                lev = np.concatenate((levels[rr], levels[rr, -1:] + add), axis=1)
                sid, start = segment_map(dd, L)
                slen = gather_levels(dd, sid)
                active = ((sid % 2) == 1) ^ sa[rr]
                now = gather_levels(lev, sid)
                previous = gather_levels(np.concatenate((lev[:, :1], lev[:, :-1]), axis=1), sid)
                if mode == DORMANT_FLAT:
                    calm = now
                elif mode == DORMANT_MICRO:
                    calm = now + P['micro_slope'][rr, None] * P['level_step'][rr, None] * (t - start)
                elif mode == DORMANT_COUNTER:
                    tick = P['counter_step'][rr, None]
                    calm = np.round(now / tick) * tick
                elif mode == DORMANT_ZERO:
                    calm = dormant[rr]  # already-drawn sparse observations
                else:
                    progress = (t - start) / np.maximum(slen, 1.0)
                    w = np.clip(progress / np.maximum(P['ramp_frac'][rr, None], 1e-3), 0.0, 1.0)
                    w = 0.5 * (1.0 + np.tanh(6.0 * (w - 0.5)))
                    calm = previous + (now - previous) * w
                carrier_new = np.where(active, base[rr] * var_hi[rr], base[rr] / var_hi[rr])
                fixed = np.where(active, now + carrier_new + noise[rr], calm)
                if mode == DORMANT_ZERO:
                    fixed = np.where(active, np.abs(carrier_new), calm)
                y[rr] = np.where(t >= first[:, None], fixed, y[rr])

        fl = new_flags(n, scale_mode=1 if mode == DORMANT_ZERO else 0,
                       positive=(mode == DORMANT_ZERO),
                       quant_boost=1.4 if mode == DORMANT_COUNTER else 0.9)
        if mode == DORMANT_ZERO:
            fl["offset_pref"] = np.full(n, 2, dtype=np.int8)
        if cfg.get('_complete_native_renewals', False):
            fl['renewal_extended'] = extended
        return y, fl

    return build


# ═══════════════════ D2 level_ladder_staircase ═════════════════════════════

_MANT = np.array([1.0, 2.0, 5.0])


def d2_params(rng, B, cfg):
    p = cfg["families"]["level_ladder_staircase"]
    return {
        "run_mean": logu(rng, p["run_length"][0], p["run_length"][1], B),
        "run_sigma": rng.uniform(0.5, 1.4, B),
        "jump_kind": categorical(rng, p["jump_mix"], B),
        "small": logu(rng, 0.05, 0.6, B),
        "medium": logu(rng, 0.6, 4.0, B),
        "huge": logu(rng, 4.0, 60.0, B),
        "snap": rng.random(B) < p["tick_snap_rate"],
        "tick_mant": _MANT[rng.integers(0, 3, B)],
        "tick_exp": rng.integers(-3, 3, B),
        "noiseless": rng.random(B) < p["noiseless_rate"],
        "monotone": rng.random(B) < p["monotone_rate"],
        "noise": logu(rng, 1e-3, 0.3, B),
    }


def d2_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    M = 96
    dwell = np.maximum(np.exp(rng.normal(np.log(P["run_mean"]).reshape(n, 1),
                                         P["run_sigma"].reshape(n, 1), size=(n, M))), 2.0)
    seg_id, _ = segment_map(dwell, L)

    kind = P["jump_kind"].reshape(n, 1)
    size = np.where(kind == 0, P["small"].reshape(n, 1),
                    np.where(kind == 1, P["medium"].reshape(n, 1),
                             P["huge"].reshape(n, 1)))
    jumps = rng.standard_normal((n, M)) * size
    mono = P["monotone"].reshape(n, 1)
    jumps = np.where(mono, np.abs(jumps), jumps)
    levels = np.cumsum(jumps, axis=1)

    tick = P["tick_mant"].reshape(n, 1) * np.power(10.0, P["tick_exp"].reshape(n, 1))
    raw_levels = levels
    snapped = np.round(levels / tick) * tick
    levels = np.where(P["snap"].reshape(n, 1), snapped, levels)

    y = gather_levels(levels, seg_id)
    noise = rng.standard_normal((n, L)) * P["noise"].reshape(n, 1) * size
    y = np.where(P["noiseless"].reshape(n, 1), y, y + noise)

    extended = np.zeros(n, dtype=bool)
    if cfg.get('_complete_native_renewals', False):
        rr, dd, first = complete_dwell_table(
            rng, dwell, L, P['run_mean'].reshape(n, 1), P['run_sigma'].reshape(n, 1), 2.0)
        if rr.size:
            extended[rr] = True
            add = rng.standard_normal((rr.size, dd.shape[1] - M)) * size[rr]
            add = np.where(mono[rr], np.abs(add), add)
            extra_levels = raw_levels[rr, -1:] + np.cumsum(add, axis=1)
            extra_levels = np.where(P['snap'][rr, None],
                                    np.round(extra_levels / tick[rr]) * tick[rr], extra_levels)
            lev = np.concatenate((levels[rr], extra_levels), axis=1)
            sid, _ = segment_map(dd, L)
            fixed = gather_levels(lev, sid)
            fixed = np.where(P['noiseless'][rr, None], fixed, fixed + noise[rr])
            t = time_grid(L)[None, :]
            y[rr] = np.where(t >= first[:, None], fixed, y[rr])

    fl = new_flags(n, scale_mode=0, quant_boost=0.6)
    fl["obs_boost"] = np.where(P["noiseless"], 0.4, 1.0)
    if cfg.get('_complete_native_renewals', False):
        fl['renewal_extended'] = extended
    return y, fl


# ═══════════════ D3 smooth_drift_extrapolable ══════════════════════════════

def d3_params(rng, B, cfg):
    p = cfg["families"]["smooth_drift_extrapolable"]
    return {
        "nu_idx": categorical(rng, p["matern_nu_mix"], B),
        "ell": logu(rng, p["lengthscale"][0], p["lengthscale"][1], B),
        "gp_w": rng.uniform(0.2, 1.0, B),
        "llt_w": rng.uniform(0.2, 1.0, B),
        "sig_level": logu(rng, 1e-3, 1.0, B),
        "slope_ratio": logu(rng, 1e-4, 1e-1, B),
        "damped": rng.random(B) < p["damped_rate"],
        "damp_phi": rng.uniform(0.80, 0.995, B),
        "saturating": rng.random(B) < p["saturating_rate"],
        "sat_kind": rng.integers(0, 2, B),
        "sat_infl": rng.uniform(-0.4, 1.4, B),
        "sat_rate": logu(rng, 2.0, 30.0, B),
        "sat_amp": logu(rng, 0.5, 12.0, B),
        "obs_ratio": logu(rng, p["obs_noise_ratio"][0], p["obs_noise_ratio"][1], B),
    }


NU_VALUES = (0.5, 1.5, 2.5)


def d3_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    gp = np.zeros((n, L))
    for k, nu in enumerate(NU_VALUES):
        m = P["nu_idx"] == k
        c = int(m.sum())
        if c:
            gp[m] = matern_gp(rng, c, L, P["ell"][m], nu=nu)

    damp = np.where(P["damped"], P["damp_phi"], 1.0)
    llt = local_linear_trend(rng, n, L, P["sig_level"],
                             P["sig_level"] * P["slope_ratio"],
                             damp=damp)
    llt = unit_std(llt)

    y = P["gp_w"].reshape(n, 1) * gp + P["llt_w"].reshape(n, 1) * llt

    if P["saturating"].any():
        x0 = P["sat_infl"].reshape(n, 1) * L
        k = np.maximum(L / P["sat_rate"].reshape(n, 1), 1.0)
        logis = 1.0 / (1.0 + np.exp(-np.clip((t - x0) / k, -40.0, 40.0)))
        gomp = np.exp(-np.exp(-np.clip((t - x0) / k, -40.0, 40.0)))
        curve = np.where(P["sat_kind"].reshape(n, 1) == 0, logis, gomp)
        y = np.where(P["saturating"].reshape(n, 1),
                     y + P["sat_amp"].reshape(n, 1) * curve, y)

    sig = unit_std(y)
    obs = rng.standard_normal((n, L)) * P["obs_ratio"].reshape(n, 1)
    y = sig + obs

    fl = new_flags(n, scale_mode=0, quant_boost=0.7)
    fl["offset_pref"] = np.where(P["obs_ratio"] < 0.02, 1, 0).astype(np.int8)
    return y, fl


# ═══════════════ chronoforge: cf_fam_stoch ═══════════════
# Block E — the stochastic backbone.
#
# Four things here have no analogue anywhere in the competitive field:
#
# * **moving-average and seasonal-differenced structure** (E1).  The field's whole
#   linear vocabulary is AR(1)/AR(2)/threshold-AR.  MA terms and ``(1 - B^m)``
#   differencing change the 64-step conditional-mean path and the error-variance
#   profile in ways no pure-AR family can express, and via the batched ARMA kernel
#   they cost the same as AR(2).
# * **true conditional heteroscedasticity with leverage** (E2).  CRPS is a
#   distributional score, so the largest relative gains come from conditioning
#   interval *width* on the recent context.
# * **self-exciting clustered arrivals with a power-law kernel** (E3), including
#   genuine long-memory clustering built from four exponentials.
# * **spectral-mixture kernels** (E5), which give quasi-periodic structure at
#   non-integer, mutually incommensurate periods — a direct attack on the
#   field-wide fixed integer period grid, at O(L log L).


MAX_SEASONAL_M = 48
MAX_ORDER = 3 + 2 * MAX_SEASONAL_M


# ═════════════════════════════ E1 arma_sarima ══════════════════════════════

def _poly_from_inverse_roots(rng, n, order, lo=0.05, hi=0.97):
    """Degree-3 polynomial coefficients from drawn inverse roots.

    Building the polynomial from roots (rather than drawing coefficients and
    rejecting) makes stationarity and invertibility hold *by construction* — no
    rejection loop, so the cost is a fixed handful of vector ops.
    """
    r = rng.uniform(lo, hi, size=(n, 3)) * np.sign(rng.standard_normal((n, 3)))
    live = np.arange(3)[None, :] < np.asarray(order).reshape(n, 1)
    r = np.where(live, r, 0.0)
    complex_pair = (rng.random(n) < 0.45) & (np.asarray(order) >= 2)
    rho = rng.uniform(lo, hi, n)
    th = rng.uniform(0.15, np.pi - 0.15, n)
    a, b, c = r[:, 0], r[:, 1], r[:, 2]
    # all-real product
    c1_r = a + b + c
    c2_r = a * b + a * c + b * c
    c3_r = a * b * c
    # one real root x one complex-conjugate pair.  At order 2 the pair IS the
    # polynomial: drop the real root so the degree stays 2 (c3 == 0).  Leaving
    # it in made a degree-3 seasonal factor whose B^3m term _expand_seasonal
    # then truncated, and a truncated polynomial is not stationary.
    a_c = np.where(np.asarray(order) >= 3, a, 0.0)
    two_rc = 2.0 * rho * np.cos(th)
    c1_c = a_c + two_rc
    c2_c = rho ** 2 + a_c * two_rc
    c3_c = a_c * rho ** 2
    use = complex_pair
    c1 = np.where(use, c1_c, c1_r)
    c2 = np.where(use, c2_c, c2_r)
    c3 = np.where(use, c3_c, c3_r)
    # (1 - aB)(1 - bB)(1 - cB) = 1 - e1*B + e2*B^2 - e3*B^3: the elementary
    # symmetric coefficients ALTERNATE in sign.  _expand_seasonal builds
    # ``1 - k1*B - k2*B^2 - k3*B^3`` from what we return, so hand it
    # (e1, -e2, e3) — returning (e1, e2, e3) flips the B^2 term and about a
    # quarter of the draws land outside the unit circle and explode.
    return np.stack([c1, -c2, c3], axis=1)


def _expand_seasonal(base, seas, m, n):
    """Convolve a degree-3 polynomial with a seasonal polynomial at lag m."""
    out = np.zeros((n, MAX_ORDER + 1))
    poly = np.concatenate([np.ones((n, 1)), -base], axis=1)          # 1 - c1 B - ...
    spol = np.concatenate([np.ones((n, 1)), -seas], axis=1)          # 1 - S1 B^m - ...
    rows = np.arange(n)
    for j in range(4):
        for k in range(3):
            idx = j + k * np.asarray(m).astype(np.int64)
            idx = np.clip(idx, 0, MAX_ORDER)
            np.add.at(out, (rows, idx), poly[:, j] * spol[:, k])
    return -out[:, 1:]


def e1_params(rng, B, cfg):
    p = cfg["families"]["arma_sarima"]
    mgrid = np.array([2, 3, 4, 6, 7, 12, 24, 48])
    return {
        "p": rng.integers(0, 4, B),
        "q": rng.integers(0, 4, B),
        "P": rng.integers(0, 3, B),
        "Q": rng.integers(0, 3, B),
        "d": (rng.random(B) < p["d_rate"]).astype(np.int64),
        "D": (rng.random(B) < p["big_d_rate"]).astype(np.int64),
        "m": mgrid[rng.integers(0, len(mgrid), B)],
        "seasonal_on": rng.random(B) < p["seasonal_rate"],
        "student": rng.random(B) < p["student_rate"],
        "nu": rng.uniform(3.0, 10.0, B),
        "sig": logu(rng, 0.05, 20.0, B),
        "drift": rng.normal(0.0, 0.02, B),
    }


def e1_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    son = P["seasonal_on"]
    m = np.where(son, P["m"], 0).astype(np.int64)
    Pord = np.where(son, P["P"], 0)
    Qord = np.where(son, P["Q"], 0)
    D = np.where(son, P["D"], 0)

    ar = _poly_from_inverse_roots(rng, n, P["p"])
    ma = _poly_from_inverse_roots(rng, n, P["q"])
    sar = _poly_from_inverse_roots(rng, n, np.minimum(Pord, 3))
    sma = _poly_from_inverse_roots(rng, n, np.minimum(Qord, 3))

    phi = np.ascontiguousarray(_expand_seasonal(ar, sar, m, n))
    theta = np.ascontiguousarray(-_expand_seasonal(ma, sma, m, n))
    ordv = np.where(m > 0, 3 + 2 * m, 3).astype(np.int64)
    ordv = np.minimum(ordv, MAX_ORDER)

    z = rng.standard_normal((n, L))
    if P["student"].any():
        nu = np.maximum(P["nu"].reshape(n, 1), 2.5)
        g = rng.chisquare(np.broadcast_to(nu, (n, L))) / nu
        tt = z / np.sqrt(np.maximum(g, 1e-9))
        tt *= np.sqrt(np.maximum((nu - 2.0) / nu, 1e-3))
        z = np.where(P["student"].reshape(n, 1), tt, z)
    e = np.ascontiguousarray(z * P["sig"].reshape(n, 1))

    out = np.zeros((n, L))
    k_arma(phi, ordv, theta, ordv, e, out)
    out = np.clip(out, -1e120, 1e120)

    if (D > 0).any():
        acc = np.zeros((n, L))
        k_seasonal_int(np.ascontiguousarray(out),
                         np.where(D > 0, np.maximum(m, 1), L + 1).astype(np.int64), acc)
        out = np.where((D > 0).reshape(n, 1), acc, out)
    dmask = (P["d"] > 0).reshape(n, 1)
    if dmask.any():
        t = time_grid(L)[None, :]
        integ = np.cumsum(out, axis=1) + P["drift"].reshape(n, 1) \
            * P["sig"].reshape(n, 1) * t
        out = np.where(dmask, integ, out)

    fl = new_flags(n, scale_mode=0, quant_boost=0.9)
    return out, fl


# ═══════════════════════════ E2 garch_leverage ═════════════════════════════

def e2_params(rng, B, cfg):
    p = cfg["families"]["garch_leverage"]
    persist = rng.uniform(p["persistence"][0], p["persistence"][1], B)
    lev = rng.uniform(0.0, 1.5, B)
    alpha = rng.uniform(0.02, 0.14, B)
    gamma = alpha * lev
    beta = np.maximum(persist - alpha - 0.5 * gamma, 0.05)
    return {
        "omega": logu(rng, 1e-8, 1e-3, B),
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "student": rng.random(B) < p["student_rate"],
        "nu": rng.uniform(3.5, 12.0, B),
        "emit": categorical(rng, p["emit_mix"], B),
        "arma_mean": rng.random(B) < p["arma_mean_rate"],
        "mean_phi": rng.uniform(-0.3, 0.4, B),
        "s0": logu(rng, 1.0, 5.0e4, B),
        "mu_drift": rng.normal(0.0, 3e-4, B),
    }


def e2_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    z = rng.standard_normal((n, L))
    if P["student"].any():
        nu = np.maximum(P["nu"].reshape(n, 1), 2.5)
        g = rng.chisquare(np.broadcast_to(nu, (n, L))) / nu
        tt = z / np.sqrt(np.maximum(g, 1e-9)) * np.sqrt((nu - 2.0) / nu)
        z = np.where(P["student"].reshape(n, 1), tt, z)

    r = np.zeros((n, L))
    sd = np.zeros((n, L))
    k_garch(np.ascontiguousarray(P["omega"]), np.ascontiguousarray(P["alpha"]),
              np.ascontiguousarray(P["gamma"]), np.ascontiguousarray(P["beta"]),
              np.ascontiguousarray(z), r, sd)

    if P["arma_mean"].any():
        mean = np.zeros((n, L))
        k_arma(np.ascontiguousarray(P["mean_phi"].reshape(n, 1)),
                 np.ones(n, dtype=np.int64), np.zeros((n, 1)),
                 np.zeros(n, dtype=np.int64), np.ascontiguousarray(r), mean)
        r = np.where(P["arma_mean"].reshape(n, 1), mean, r)
    r = r + P["mu_drift"].reshape(n, 1)

    price = P["s0"].reshape(n, 1) * np.exp(np.clip(np.cumsum(r, axis=1), -50.0, 50.0))
    rv = decay_convolve(sd ** 2, np.full(n, 24.0)) * 24.0
    emit = P["emit"].reshape(n, 1)
    y = np.where(emit == 0, r, np.where(emit == 1, price, np.sqrt(np.maximum(rv, 0.0))))

    fl = new_flags(n, scale_mode=0, quant_boost=0.9)
    # mean-zero return series have a small sum|y| and are therefore the
    # relative-WQL amplifiers; anchor them at zero rather than on a big offset
    fl["offset_pref"] = np.where(P["emit"] == 0, 3,
                                 np.where(P["emit"] == 1, 1, 2)).astype(np.int8)
    fl["scale_mode"] = np.where(P["emit"] == 0, 0, 1).astype(np.int8)
    fl["positive"] = (P["emit"] > 0)
    return y, fl


# ════════════════════════════ E3 hawkes_marked ═════════════════════════════

def e3_params(rng, B, cfg):
    p = cfg["families"]["hawkes_marked"]
    return {
        "mu": logu(rng, 1e-4, 2.0, B),
        "branch": rng.uniform(p["branching"][0], p["branching"][1], B),
        "tau": logu(rng, 2.0, 200.0, B),
        "power_law": rng.random(B) < p["power_law_rate"],
        "diurnal": rng.random(B) < 0.45,
        "diurnal_amp": rng.uniform(0.2, 0.9, B),
        "emit": categorical(rng, p["emit_mix"], B),
        "mark_sig": rng.uniform(0.4, 1.8, B),
        "decay_tau": logu(rng, 2.0, 120.0, B),
    }


def e3_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    pd = cad.p_day.reshape(n, 1)
    has_day = pd >= 3.0
    ph = t / np.where(has_day, pd, 1e9) + rng.random((n, 1))
    prof = 1.0 + np.where(P["diurnal"].reshape(n, 1) & has_day,
                          P["diurnal_amp"].reshape(n, 1) * np.sin(_cf_prims_TWO_PI * ph), 0.0)
    mu = np.ascontiguousarray(P["mu"].reshape(n, 1) * np.maximum(prof, 0.05))
    mu = np.ascontiguousarray(np.broadcast_to(mu, (n, L)).copy())

    lam, cnt = hawkes(rng, n, L, mu, P["branch"], P["tau"], P["power_law"])
    marks = cnt * np.exp(rng.normal(0.0, P["mark_sig"].reshape(n, 1), size=(n, L)))
    marked = decay_convolve(marks, P["decay_tau"])

    emit = P["emit"].reshape(n, 1)
    y = np.where(emit == 0, cnt, np.where(emit == 1, lam, marked))
    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=0.5)
    fl["integer"] = (P["emit"] == 0)
    fl["offset_pref"] = np.full(n, 2, dtype=np.int8)
    return y, fl


# ═══════════════════════════ E4 chaotic_delay ══════════════════════════════

_SYS_PAR = {
    0: (10.0, 28.0, 8.0 / 3.0, 0.0),      # Lorenz
    1: (0.2, 0.2, 5.7, 0.0),              # Rossler
    2: (1.0, 0.35, 10.0, 15.0),           # Chua (cubic)
    3: (1.0, 3.0, 5.0, 3.2),              # Hindmarsh-Rose
}


def e4_params(rng, B, cfg):
    p = cfg["families"]["chaotic_delay"]
    sysid = categorical(rng, p["system_mix"], B)
    par = np.zeros((B, 4))
    for k, v in _SYS_PAR.items():
        m = sysid == k
        par[m] = np.array(v)
    par *= np.exp(rng.normal(0.0, 0.04, size=(B, 4)))
    return {
        "system": sysid,
        "par": par,
        "dt": logu(rng, 0.004, 0.05, B),
        "sub": rng.integers(1, 4, B),
        "state0": rng.normal(0.0, 1.0, size=(B, 3)) + np.array([0.6, 0.4, 1.2]),
        "project": rng.random(B) < p["projection_rate"],
        "proj": rng.normal(0.0, 1.0, size=(B, 3)),
        "mg": rng.random(B) < p["mackey_glass_rate"],
        "mg_beta": rng.uniform(0.15, 0.3, B),
        "mg_gamma": rng.uniform(0.08, 0.12, B),
        "mg_n": rng.uniform(8.0, 12.0, B),
        "mg_tau": rng.uniform(15.0, 40.0, B),
        "obs_noise": logu(rng, 1e-4, 5e-2, B),
    }


def e4_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    traj = np.zeros((n, L, 3))
    k_rk4_3d(np.ascontiguousarray(P["system"]).astype(np.int64),
               np.ascontiguousarray(P["par"]),
               np.ascontiguousarray(P["dt"]),
               np.ascontiguousarray(P["state0"]),
               np.ascontiguousarray(P["sub"]).astype(np.int64), traj)
    pick = rng.integers(0, 3, n)
    coord = traj[np.arange(n), :, pick]
    proj = np.einsum("ntk,nk->nt", traj, P["proj"])
    y = np.where(P["project"].reshape(n, 1), proj, coord)

    if P["mg"].any():
        Bw = 256
        dsteps = np.clip(np.round(P["mg_tau"] * 4.0).astype(np.int64), 4, Bw - 2)
        hist = 1.0 + 0.15 * rng.standard_normal((n, Bw))
        mg = np.zeros((n, L))
        k_mackey_glass(np.ascontiguousarray(P["mg_beta"]),
                         np.ascontiguousarray(P["mg_gamma"]),
                         np.ascontiguousarray(P["mg_n"]),
                         np.ascontiguousarray(dsteps),
                         np.ascontiguousarray(hist), mg)
        y = np.where(P["mg"].reshape(n, 1), mg, y)

    y = unit_std(y) + rng.standard_normal((n, L)) * P["obs_noise"].reshape(n, 1)
    fl = new_flags(n, scale_mode=0, quant_boost=0.6)
    return y, fl


# ═══════════════════════ E5 spectral_kernel_zoo ════════════════════════════

N_PSD_TYPES = 7


def e5_params(rng, B, cfg):
    p = cfg["families"]["spectral_kernel_zoo"]
    return {
        "n_comp": rng.integers(1, 4, B),
        "types": rng.integers(0, N_PSD_TYPES, (B, 3)),
        "weights": rng.uniform(0.2, 1.0, (B, 3)),
        "ell": logu(rng, 4.0, 2000.0, (B, 3)),
        "alpha": logu(rng, 0.1, 10.0, (B, 3)),
        "f0": logu(rng, 1.0 / 900.0, 1.0 / 5.0, (B, 3)),
        "pwidth": logu(rng, 1e-4, 3e-3, (B, 3)),
        "pdecay": rng.uniform(0.2, 1.5, (B, 3)),
        "sm_q": rng.integers(1, 5, B),
        "sm_c": logu(rng, 1.0 / 2000.0, 0.35, (B, 4)),
        "sm_w": logu(rng, 2e-5, 8e-3, (B, 4)),
        "sm_a": rng.uniform(0.2, 1.0, (B, 4)),
        "beta": rng.uniform(0.4, 2.6, (B, 3)),
        "fbreak": logu(rng, 1e-5, 5e-3, (B, 3)),
        "product": rng.random(B) < p["product_rate"],
        "envelope": rng.random(B) < p["envelope_rate"],
        "warp": rng.random(B) < p["warp_rate"],
        "noise": logu(rng, 1e-3, 0.25, B),
    }


def _psd_slot(rng, P, f, n, slot):
    types = P["types"][:, slot]
    out = np.zeros((n, f.shape[0]))
    for ty in range(N_PSD_TYPES):
        m = types == ty
        c = int(m.sum())
        if c == 0:
            continue
        ell = P["ell"][m, slot:slot + 1]
        if ty == 0:
            s = psd_rbf(f, ell)
        elif ty == 1:
            s = psd_matern(f, ell, 0.5)
        elif ty == 2:
            s = psd_matern(f, ell, 1.5)
        elif ty == 3:
            s = psd_matern(f, ell, 2.5)
        elif ty == 4:
            s = psd_rq(f, ell, P["alpha"][m, slot:slot + 1])
        elif ty == 5:
            s = psd_periodic(f, P["f0"][m, slot:slot + 1], 6,
                             P["pdecay"][m, slot:slot + 1],
                             P["pwidth"][m, slot:slot + 1])
        else:
            s = psd_pink(f, P["beta"][m, slot:slot + 1],
                         P["fbreak"][m, slot:slot + 1])
        out[m] = s / np.maximum(s.sum(axis=1, keepdims=True), 1e-300)
    return out


def e5_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    f = freq_grid(L)
    psd = np.zeros((n, f.shape[0]))
    slots = []
    for slot in range(3):
        s = _psd_slot(rng, P, f, n, slot)
        live = (slot < P["n_comp"]).reshape(n, 1)
        slots.append(s * live)
        psd += s * P["weights"][:, slot:slot + 1] * live

    # spectral-mixture component: Q Gaussian peaks at arbitrary centres, giving
    # quasi-periodic structure at mutually incommensurate, non-integer periods
    sm = psd_spectral_mixture(f, P["sm_c"], P["sm_w"],
                              P["sm_a"] * (np.arange(4)[None, :] < P["sm_q"].reshape(n, 1)))
    psd += sm / np.maximum(sm.sum(axis=1, keepdims=True), 1e-300)

    # A kernel *product* is a PSD convolution; it needs two live components.
    prod_on = P["product"] & (P["n_comp"] >= 2)
    if prod_on.any():
        prod = psd_convolve(np.maximum(slots[0], 1e-300), np.maximum(slots[1], 1e-300))
        prod /= np.maximum(prod.sum(axis=1, keepdims=True), 1e-300)
        psd = np.where(prod_on.reshape(n, 1), prod + 1e-6 * psd, psd)

    y = gp_from_psd(rng, np.maximum(psd, 1e-300), L)

    t = time_grid(L)[None, :]
    if P["envelope"].any():
        env = 1.0 + 0.9 * matern_gp(rng, n, L, np.full(n, L / 3.0), nu=1.5)
        y = np.where(P["envelope"].reshape(n, 1), y * np.maximum(env, 0.05), y)
    if P["warp"].any():
        bb = np.cumsum(rng.standard_normal((n, L)), axis=1)
        bb = bb - (t / (L - 1.0)) * bb[:, -1:]
        bb = bb / np.maximum(np.abs(bb).max(axis=1, keepdims=True), 1e-9)
        src = np.clip(t + bb * (L * 0.06), 0.0, L - 1.0001)
        i0 = src.astype(np.int64)
        fr = src - i0
        w = np.take_along_axis(y, i0, axis=1) * (1.0 - fr) + \
            np.take_along_axis(y, np.minimum(i0 + 1, L - 1), axis=1) * fr
        y = np.where(P["warp"].reshape(n, 1), w, y)

    y = unit_std(y) + rng.standard_normal((n, L)) * P["noise"].reshape(n, 1)
    fl = new_flags(n, scale_mode=0, quant_boost=0.9)
    return y, fl


# ═══════════════════════ E6 changepoint_composite ══════════════════════════

MAX_SEG = 5


def e6_params(rng, B, cfg):
    p = cfg["families"]["changepoint_composite"]
    return {
        "n_seg": rng.integers(2, MAX_SEG + 1, B),
        "kinds": rng.integers(0, N_ACTIVE_KINDS + 2, (B, MAX_SEG)),
        "amps": logu(rng, 0.2, 8.0, (B, MAX_SEG)),
        "levels": rng.normal(0.0, 1.0, (B, MAX_SEG)),
        "join": categorical(rng, p["join_mix"], B),
        "fade_frac": rng.uniform(0.05, 0.10, B),
        "hazard_state": rng.uniform(p["state_hazard"][0], p["state_hazard"][1], B),
        "trend_slope": rng.normal(0.0, 2.0, (B, MAX_SEG)),
        "ladder_step": logu(rng, 0.2, 4.0, (B, MAX_SEG)),
        "ladder_run": logu(rng, 20.0, 600.0, (B, MAX_SEG)),
    }


def e6_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]

    # Build MAX_SEG candidate processes per row, each with its own family.
    kinds = P["kinds"].reshape(-1)
    cad_rep = cad.take(np.repeat(np.arange(n), MAX_SEG))
    base_kind = np.minimum(kinds, N_ACTIVE_KINDS - 1)
    cands = active_base(rng, n * MAX_SEG, L, base_kind, cad_rep, cfg)

    # two extra menu entries beyond the shared active-base menu
    tr = kinds == N_ACTIVE_KINDS
    if tr.any():
        slope = P["trend_slope"].reshape(-1)[tr].reshape(-1, 1)
        cands[tr] = unit_std(slope * (t / L) + 0.15 * np.cumsum(
            rng.standard_normal((int(tr.sum()), L)), axis=1) / np.sqrt(L))
    ld = kinds == N_ACTIVE_KINDS + 1
    if ld.any():
        c = int(ld.sum())
        run = P["ladder_run"].reshape(-1)[ld].reshape(c, 1)
        dwell = np.maximum(np.exp(rng.normal(np.log(run), 0.7, size=(c, 48))), 2.0)
        seg, _ = segment_map(dwell, L)
        lv = np.cumsum(rng.standard_normal((c, 48)), axis=1)
        cands[ld] = unit_std(gather_levels(lv, seg))

    cands = cands.reshape(n, MAX_SEG, L) * P["amps"].reshape(n, MAX_SEG, 1)
    cands = cands + P["levels"].reshape(n, MAX_SEG, 1)

    # Break hazard: uniform in time, elevated after a high-volatility stretch.
    # The model therefore learns break hazard conditioned on observable
    # precursors rather than on a fixed relative position.
    v = np.abs(np.diff(cands[:, 0, :], axis=1, prepend=cands[:, 0, :1]))
    v = decay_convolve(v, np.full(n, 64.0))
    v = (v - v.mean(axis=1, keepdims=True)) / np.maximum(v.std(axis=1, keepdims=True), 1e-9)
    logw = P["hazard_state"].reshape(n, 1) * np.clip(v, -3.0, 3.0)
    guard = np.zeros((n, L))
    guard[:, :L // 16] = -30.0
    guard[:, -L // 16:] = -30.0
    gum = -np.log(-np.log(np.maximum(rng.random((n, L)), 1e-12)))
    key = logw + gum + guard
    cut = np.sort(np.argsort(-key, axis=1)[:, :MAX_SEG - 1], axis=1)

    live = np.arange(MAX_SEG - 1)[None, :] < (P["n_seg"] - 1).reshape(n, 1)
    cut = np.where(live, cut, L + 1)
    seg_id = np.zeros((n, L), dtype=np.int64)
    for j in range(MAX_SEG - 1):
        seg_id += (t >= cut[:, j:j + 1]).astype(np.int64)

    y = np.take_along_axis(cands, seg_id[:, None, :], axis=1)[:, 0, :]

    join = P["join"].reshape(n, 1)
    if (P["join"] > 0).any():
        fade = np.maximum(P["fade_frac"].reshape(n, 1) * L, 2.0)
        blend = y.copy()
        for j in range(MAX_SEG - 1):
            cj = cut[:, j:j + 1].astype(np.float64)
            w = np.clip((t - cj) / fade + 0.5, 0.0, 1.0)
            inside = (np.abs(t - cj) < fade) & live[:, j:j + 1]
            lhs = cands[:, j, :]
            rhs = cands[:, min(j + 1, MAX_SEG - 1), :]
            blend = np.where(inside, lhs * (1.0 - w) + rhs * w, blend)
        y = np.where(join == 1, blend, y)

    if (P["join"] == 2).any():
        # level-matched continuous joins: remove the jump at each break
        step = np.zeros((n, L))
        for j in range(MAX_SEG - 1):
            cj = np.clip(cut[:, j:j + 1], 0, L - 1)
            before = np.take_along_axis(y, np.maximum(cj - 1, 0), axis=1)
            after = np.take_along_axis(y, cj, axis=1)
            step += np.where((t >= cj) & live[:, j:j + 1], before - after, 0.0)
        y = np.where(join == 2, y + step, y)

    fl = new_flags(n, scale_mode=0, quant_boost=0.9)
    return y, fl


# ═══════════════ chronoforge: cf_fam_domain ═══════════════
# Block F — energy / transport / retail / macro families.
#
# Three of these encode a *nonlinear map from a smooth latent to the observable*,
# which is what actually generates the shapes and what a purely additive prior
# cannot represent:
#
# * electricity load is a hockey-stick function of temperature (heating below one
#   breakpoint, cooling above another), and the price is a convex supply stack, so
#   a modest change in residual load produces an extreme price spike;
# * traffic flow is non-monotone in demand — past capacity, density rises and flow
#   *falls* along the backward-bending branch of the fundamental diagram;
# * solar output is a clipped diurnal bell times a cloud latent, i.e. an exact zero
#   every night and a lag-24 difference of approximately zero through it.


# ══════════════════ F1 energy_load_price_solar ═════════════════════════════

def f1_params(rng, B, cfg):
    p = cfg["families"]["energy_load_price_solar"]
    return {
        "mode": categorical(rng, p["mode_mix"], B),
        "base": logu(rng, 10.0, 5.0e4, B),
        "temp_ell": logu(rng, 40.0, 900.0, B),
        "temp_amp": rng.uniform(3.0, 14.0, B),
        "t_cool": rng.uniform(16.0, 24.0, B),
        "t_heat": rng.uniform(8.0, 16.0, B),
        "b_cool": logu(rng, 0.005, 0.09, B),
        "c_heat": logu(rng, 0.005, 0.09, B),
        "weekend": np.exp(rng.normal(np.log(0.88), 0.12, B)),
        "noise": logu(rng, 5e-3, 0.12, B),
        "p0": logu(rng, 5.0, 200.0, B),
        "kappa": logu(rng, 0.5, 40.0, B),
        "theta": rng.uniform(1.5, 7.0, B),
        "cap_q": rng.uniform(0.75, 0.97, B),
        "negative": rng.random(B) < p["negative_price_rate"],
        "solar_peak": logu(rng, 1.0, 5.0e3, B),
        "cloud_ell": logu(rng, 4.0, 120.0, B),
        "cloud_eta": logu(rng, 0.8, 3.0, B),
        "cloud_c": rng.normal(0.55, 0.3, B),
        "day_frac": rng.uniform(0.32, 0.55, B),
    }


def f1_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    pd = cad.p_day.reshape(n, 1)
    has_day = pd >= 4.0
    ph = t / np.where(has_day, pd, 1e9) + rng.random((n, 1))
    frac = ph - np.floor(ph)

    # latent temperature (the same construction as the thermal weather family)
    temp = matern_gp(rng, n, L, P["temp_ell"], nu=1.5) * P["temp_amp"].reshape(n, 1) \
        + 15.0 + np.where(has_day, 4.0 * np.sin(_cf_prims_TWO_PI * ph - 1.9), 0.0)

    shape = profile_double_peak(rng, n, L, ph)
    shape = shape / np.maximum(shape.mean(axis=1, keepdims=True), 1e-9)
    shape = np.where(has_day, 0.55 + 0.45 * shape, 1.0)

    wk = np.ones((n, 7))
    wk[:, 5] = P["weekend"]
    wk[:, 6] = P["weekend"] * 0.97
    weekly = apply_dow(wk, cal)

    # the hockey stick: heating below t_heat, cooling above t_cool
    resp = (1.0 + P["b_cool"].reshape(n, 1) * np.maximum(temp - P["t_cool"].reshape(n, 1), 0.0)
            + P["c_heat"].reshape(n, 1) * np.maximum(P["t_heat"].reshape(n, 1) - temp, 0.0))
    load = P["base"].reshape(n, 1) * shape * weekly * resp
    load = load * np.exp(rng.standard_normal((n, L)) * P["noise"].reshape(n, 1))

    # convex supply stack -> occasional extreme spikes near capacity
    cap = load.mean(axis=1, keepdims=True) * (1.0 + 1.4 * P["cap_q"].reshape(n, 1))
    excess = (load - cap) / np.maximum(cap, 1e-9)
    price = P["p0"].reshape(n, 1) + P["kappa"].reshape(n, 1) * (
        np.exp(np.clip(P["theta"].reshape(n, 1) * excess, -30.0, 12.0)) - np.exp(-1.0))
    neg = P["negative"].reshape(n, 1)
    price = np.where(neg, price - P["p0"].reshape(n, 1) * 1.35, price)

    # solar: clipped bell x cloud, exactly zero every night
    bell = np.cos(np.pi * (frac - 0.5) / np.maximum(P["day_frac"].reshape(n, 1), 1e-3))
    bell = np.maximum(bell, 0.0)
    cz = unit_std(matern_gp(rng, n, L, P["cloud_ell"], nu=1.5))
    clear = 1.0 - np.clip(P["cloud_eta"].reshape(n, 1) * cz
                          + P["cloud_c"].reshape(n, 1), 0.0, 1.0)
    solar = P["solar_peak"].reshape(n, 1) * bell * np.maximum(clear, 0.0)
    solar = np.where(has_day, solar, np.maximum(P["solar_peak"].reshape(n, 1) * clear, 0.0))

    mode = P["mode"].reshape(n, 1)
    y = np.where(mode == 0, load, np.where(mode == 1, price, solar))
    fl = new_flags(n, scale_mode=1, quant_boost=0.8)
    fl["positive"] = (P["mode"] != 1)
    fl["scale_mode"] = np.where(P["mode"] == 1, 0, 1).astype(np.int8)
    fl["offset_pref"] = np.where(P["mode"] == 1, 3, 2).astype(np.int8)
    return y, fl


# ═══════════════════════════ F2 transport_flow ═════════════════════════════

def f2_params(rng, B, cfg):
    p = cfg["families"]["transport_flow"]
    return {
        "base": logu(rng, 5.0, 5.0e3, B),
        "weekend": np.exp(rng.normal(np.log(0.55), 0.35, B)),
        "capacity": rng.uniform(p["capacity_ratio"][0], p["capacity_ratio"][1], B),
        "jam_slope": rng.uniform(0.3, 1.4, B),
        "nb_k": logu(rng, 1.0, 300.0, B),
        "count_emit": rng.random(B) < 0.6,
        "inc_tau": logu(rng, 600.0, 6000.0, B),
        "inc_drop": rng.uniform(0.25, 0.8, B),
        "inc_recover": logu(rng, 20.0, 200.0, B),
        "noise": logu(rng, 0.02, 0.35, B),
    }


def f2_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    pd = cad.p_day.reshape(n, 1)
    has_day = pd >= 4.0
    ph = t / np.where(has_day, pd, 1e9) + rng.random((n, 1))

    weekday = profile_double_peak(rng, n, L, ph)
    weekend_shape = np.exp(20.0 * (np.cos(_cf_prims_TWO_PI * (ph - 0.55)) - 1.0))
    is_we = cal.is_weekend
    shape = np.where(is_we, weekend_shape * P["weekend"].reshape(n, 1), weekday)
    shape = np.where(has_day, shape, 1.0)
    shape = shape / np.maximum(shape.mean(axis=1, keepdims=True), 1e-9)

    demand = P["base"].reshape(n, 1) * shape * np.exp(
        ar1(rng, n, L, np.full(n, 0.97)) * P["noise"].reshape(n, 1))

    # fundamental diagram: flow rises to capacity, then FALLS as density grows
    cap = P["capacity"].reshape(n, 1) * P["base"].reshape(n, 1)
    over = np.maximum(demand - cap, 0.0)
    flow = np.minimum(demand, cap) - P["jam_slope"].reshape(n, 1) * over \
        / (1.0 + over / np.maximum(cap, 1e-9))
    flow = np.maximum(flow, 0.0)

    # incidents: sharp drop, queue build, slow recovery ramp
    E = 6
    hit = rng.random((n, E)) < (L / (P["inc_tau"].reshape(n, 1) * E))
    pos = rng.integers(0, L, size=(n, E)).astype(np.float64)
    drop = np.zeros((n, L))
    for j in range(E):
        idx = np.nonzero(hit[:, j])[0]
        if idx.size == 0:
            continue
        p0 = pos[idx, j:j + 1]
        dt = np.maximum(t - p0, 0.0)
        shape_j = np.exp(-dt / P["inc_recover"][idx].reshape(-1, 1))
        drop[idx] += np.where(t >= p0,
                              P["inc_drop"][idx].reshape(-1, 1) * shape_j, 0.0)
    flow = flow * np.maximum(1.0 - np.minimum(drop, 0.95), 0.02)

    counts = nb_counts(rng, flow, P["nb_k"])
    y = np.where(P["count_emit"].reshape(n, 1), counts, flow)
    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=0.7)
    fl["integer"] = P["count_emit"]
    return y, fl


# ═══════════════════ F3 retail_promo_intermittent ══════════════════════════

def f3_params(rng, B, cfg):
    p = cfg["families"]["retail_promo_intermittent"]
    return {
        "base": logu(rng, 0.4, 3.0e4, B),
        "trend": rng.normal(0.0, 0.4, B),
        "season_amp": rng.uniform(0.0, 0.8, B),
        "promo_period": np.array([7.0, 14.0, 30.436875, 91.310625])[
            categorical(rng, [0.30, 0.20, 0.35, 0.15], B)],
        "promo_lift": logu(rng, 1.3, 6.0, B),
        "promo_len": rng.integers(1, 5, B),
        "trough_depth": rng.uniform(0.5, 0.9, B),
        "trough_len": rng.integers(3, 22, B),
        "elasticity": rng.uniform(-3.0, -0.3, B),
        "price_sig": rng.uniform(0.02, 0.25, B),
        "slow": rng.random(B) < p["slow_mover_rate"],
        "interval": logu(rng, 2.0, 60.0, B),
        "size_corr": rng.uniform(0.0, 0.9, B),
        "stockout": rng.random(B) < p["stockout_rate"],
        "so_len": logu(rng, 5.0, 120.0, B),
        "launch": rng.random(B) < p["launch_rate"],
        "launch_at": rng.uniform(0.05, 0.5, B),
        "launch_ramp": logu(rng, 20.0, 600.0, B),
        "nb_k": logu(rng, 0.3, 60.0, B),
    }


def f3_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    day = cal.day_index.astype(np.float64)

    season = seasonal_profile(rng, n, L, np.maximum(
        P["promo_period"].reshape(n, 1) * cad.p_day.reshape(n, 1) * 4.0, 8.0), cfg)
    lam = P["base"].reshape(n, 1) * np.exp(
        P["trend"].reshape(n, 1) * t / L + P["season_amp"].reshape(n, 1) * season)

    # promotions are calendar-locked: same weekday, monthly/quarterly recurrence
    per = np.maximum(P["promo_period"].reshape(n, 1), 1.0)
    phase = (day % per)
    on = phase < P["promo_len"].reshape(n, 1)
    lift = np.where(on, P["promo_lift"].reshape(n, 1), 1.0)
    # pull-forward cannibalisation: a trough follows every promotion
    after = (phase >= P["promo_len"].reshape(n, 1)) & \
        (phase < (P["promo_len"] + P["trough_len"]).reshape(n, 1))
    lift = np.where(after, P["trough_depth"].reshape(n, 1), lift)

    logp = ar1(rng, n, L, np.full(n, 0.99)) * P["price_sig"].reshape(n, 1)
    lam = lam * lift * np.exp(P["elasticity"].reshape(n, 1) * logp)

    if P["launch"].any():
        at = P["launch_at"].reshape(n, 1) * L
        ramp = np.clip((t - at) / P["launch_ramp"].reshape(n, 1), 0.0, 1.0)
        # a genuine ramp from zero, never a constant prefix
        lam = np.where(P["launch"].reshape(n, 1), lam * ramp, lam)

    y = nb_counts(rng, lam, P["nb_k"])

    # slow movers: Croston-style intermittency with correlated interval and size
    if P["slow"].any():
        M = 128
        gaps = np.maximum(np.exp(rng.normal(
            np.log(P["interval"]).reshape(n, 1), 0.7, size=(n, M))), 1.0)
        seg, seg_start = segment_map(gaps, L)
        hit = (t == seg_start)
        glen = gather_levels(gaps, seg)
        corr = P["size_corr"].reshape(n, 1)
        size = lam * (1.0 - corr + corr * glen / np.maximum(
            P["interval"].reshape(n, 1), 1.0))
        sparse = np.where(hit, nb_counts(rng, size, P["nb_k"]), 0.0)
        y = np.where(P["slow"].reshape(n, 1), sparse, y)

    if P["stockout"].any():
        starts = rng.integers(0, L, size=(n, 3))
        lens = P["so_len"].reshape(n, 1) * np.exp(rng.normal(0.0, 0.5, size=(n, 3)))
        live = rng.random((n, 3)) < 0.5
        mask = run_mask(n, L, starts, np.where(live, lens, 0.0))
        y = np.where(mask & P["stockout"].reshape(n, 1), 0.0, y)

    fl = new_flags(n, scale_mode=1, positive=True, quant_boost=0.3)
    fl["integer"] = np.ones(n, dtype=bool)
    fl["offset_pref"] = np.full(n, 2, dtype=np.int8)
    return y, fl


# ═══════════════════ F4 econ_release_staircase ═════════════════════════════

def f4_params(rng, B, cfg):
    p = cfg["families"]["econ_release_staircase"]
    rel = np.array([7.0, 30.436875, 91.310625])
    return {
        "release_days": rel[categorical(rng, p["release_mix"], B)],
        "latent_ell": logu(rng, 100.0, 3000.0, B),
        "revise_n": rng.integers(1, 4, B),
        "revise_depth": rng.uniform(0.01, 0.2, B),
        "ns": rng.random(B) < p["nelson_siegel_rate"],
        "ns_phi": rng.uniform(0.985, 0.9999, (B, 3)),
        "ns_sig": logu(rng, 0.005, 0.12, (B, 3)),
        "ns_tau": logu(rng, 6.0, 60.0, B),
        "ns_mat": logu(rng, 0.25, 30.0, B),
        "level0": rng.normal(0.0, 1.0, B),
        "biz_grid": rng.random(B) < p["business_day_rate"],
        "month_end": rng.random(B) < p["month_end_rate"],
        "month_end_amp": rng.normal(0.0, 0.6, B),
    }


def f4_build(P, rng, L, cad, cal, cfg):
    n = cad.n
    t = time_grid(L)[None, :]
    latent = matern_gp(rng, n, L, P["latent_ell"], nu=2.5)

    # The release calendar advances in days; for business-day feeds it advances
    # only on weekdays, which makes the effective weekly period 5.
    day_eff = np.where(P["biz_grid"].reshape(n, 1),
                       business_day_index(cal), cal.day_index).astype(np.float64)
    blk = np.floor(day_eff / np.maximum(P["release_days"].reshape(n, 1), 1.0))
    boundary = np.zeros((n, L), dtype=bool)
    boundary[:, 1:] = blk[:, 1:] != blk[:, :-1]
    boundary[:, 0] = True
    idx = np.where(boundary, np.arange(L)[None, :], 0)
    idx = np.maximum.accumulate(idx, axis=1)
    stair = np.take_along_axis(latent, idx, axis=1)

    # revisions to the last 1-3 published values
    age_blk = blk.max(axis=1, keepdims=True) - blk
    rev = (age_blk < P["revise_n"].reshape(n, 1))
    stair = np.where(rev, stair * (1.0 - P["revise_depth"].reshape(n, 1)), stair)

    # Nelson-Siegel term structure: level / slope / curvature, each near unit root
    if P["ns"].any():
        f = np.zeros((n, L, 3))
        for j in range(3):
            f[:, :, j] = ar1(rng, n, L, P["ns_phi"][:, j]) * P["ns_sig"][:, j:j + 1]
            f[:, :, j] = np.cumsum(f[:, :, j], axis=1) * 0.05
        lam = 1.0 / np.maximum(P["ns_tau"].reshape(n, 1), 1e-3)
        mat = P["ns_mat"].reshape(n, 1)
        x = np.maximum(lam * mat, 1e-6)
        l1 = (1.0 - np.exp(-x)) / x
        l2 = l1 - np.exp(-x)
        ns = f[:, :, 0] + f[:, :, 1] * l1 + f[:, :, 2] * l2
        stair = np.where(P["ns"].reshape(n, 1), ns, stair)

    # month-end level jumps on econ/admin feeds
    if P["month_end"].any():
        me = np.cumsum(month_boundary(cal).astype(np.float64), axis=1)
        me = me - me.mean(axis=1, keepdims=True)
        stair = np.where(P["month_end"].reshape(n, 1),
                         stair + P["month_end_amp"].reshape(n, 1) * 0.05 * me, stair)

    y = stair + P["level0"].reshape(n, 1)
    fl = new_flags(n, scale_mode=0, offset_pref=1, quant_boost=1.5, quant_rel_hi=0.4)
    return y, fl
