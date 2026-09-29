#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hartigan's dip test of unimodality.

Implemented from the method described in:
    J. A. Hartigan and P. M. Hartigan, "The Dip Test of Unimodality",
    The Annals of Statistics 13(1), 70-84 (1985).

The dip of a sample is the largest distance between its empirical distribution function and
the closest unimodal distribution function. It is computed by shrinking a candidate modal
interval: on its left the unimodal fit is the greatest convex minorant (GCM) of the empirical
distribution, on its right the least concave majorant (LCM). The p-value is the fraction of
uniform samples of the same size (the least favourable unimodal distribution) with a larger dip.
"""
from functools import lru_cache
from typing import Tuple

import numpy as np


def _hull(xs: np.ndarray, ys: np.ndarray, lower: bool) -> np.ndarray:
    """Indices of the vertices of the lower (convex minorant) or upper (concave majorant) hull."""
    idx = []
    for i in range(len(xs)):
        while len(idx) >= 2:
            a, b = idx[-2], idx[-1]
            cross = (xs[b] - xs[a]) * (ys[i] - ys[a]) - (ys[b] - ys[a]) * (xs[i] - xs[a])
            if (cross <= 0) if lower else (cross >= 0):
                idx.pop()
            else:
                break
        idx.append(i)
    return np.asarray(idx)


def _interp(xs: np.ndarray, ys: np.ndarray, vertices: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.interp(x, xs[vertices], ys[vertices])


def dip_statistic(values) -> float:
    """
    Hartigan's dip statistic of a 1D sample (between 1/(2n) and 1/4; larger means less unimodal).

    Tied values are handled as a single point with the combined weight of its copies.
    """
    x = np.sort(np.asarray(values, dtype=float).ravel())
    n = len(x)
    if n < 2 or x[0] == x[-1]:
        return 0.0
    # Empirical distribution at the distinct values: lower (before the jump) and upper (after the jump)
    xs, counts = np.unique(x, return_counts=True)
    upper = np.cumsum(counts) / n
    lower = upper - counts / n

    lo, hi = 0, len(xs) - 1
    dip = 0.0
    while True:
        seg = np.arange(lo, hi + 1)
        sx, s_low, s_up = xs[seg], lower[seg], upper[seg]
        g = _hull(sx, s_low, lower=True)    # GCM through the lower corners
        l = _hull(sx, s_up, lower=False)    # LCM through the upper corners
        # Largest vertical distance between the two hulls, taken at the vertices of each
        d_g = _interp(sx, s_up, l, sx[g]) - s_low[g]
        d_l = s_up[l] - _interp(sx, s_low, g, sx[l])
        ig, il = int(np.argmax(d_g)), int(np.argmax(d_l))
        if d_g[ig] >= d_l[il]:
            d = d_g[ig]
            new_lo = g[ig]
            new_hi = l[np.searchsorted(l, g[ig])] if np.any(l >= g[ig]) else l[-1]
        else:
            d = d_l[il]
            new_hi = l[il]
            below = g[g <= l[il]]
            new_lo = below[-1] if len(below) else g[0]
        if d <= dip or (new_lo == 0 and new_hi == len(seg) - 1):
            dip = max(dip, d) if (new_lo == 0 and new_hi == len(seg) - 1) else dip
            break
        # Deviation of the empirical distribution from the GCM left of the new modal interval,
        # and from the LCM right of it
        left = np.arange(0, new_lo + 1)
        right = np.arange(new_hi, len(seg))
        dev_left = np.max(s_up[left] - _interp(sx, s_low, g, sx[left])) if len(left) else 0.0
        dev_right = np.max(_interp(sx, s_up, l, sx[right]) - s_low[right]) if len(right) else 0.0
        dip = max(dip, dev_left, dev_right)
        lo, hi = lo + new_lo, lo + new_hi
        if hi - lo < 1:
            break
    return float(max(dip, 1.0 / n) / 2)


@lru_cache(maxsize=256)
def _null_dips(n: int, n_sims: int, seed: int) -> np.ndarray:
    """Sorted dips of uniform samples of size n (cached per sample size)."""
    rng = np.random.default_rng(seed)
    return np.sort([dip_statistic(rng.random(n)) for _ in range(n_sims)])


def dip_test(values, n_sims: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """
    Hartigan's dip test of unimodality.

    Returns the dip statistic and the p-value: the fraction of uniform samples of the same size
    whose dip is at least as large. Small p-values (e.g. < 0.05) indicate more than one mode.
    """
    x = np.asarray(values, dtype=float).ravel()
    dip = dip_statistic(x)
    if len(x) < 4:
        return dip, 1.0
    null = _null_dips(len(x), n_sims, seed)
    p = (len(null) - np.searchsorted(null, dip, side='left')) / len(null)
    return dip, float(p)
