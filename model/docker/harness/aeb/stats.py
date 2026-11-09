"""Small, dependency-free statistics used by runners and the report."""

from __future__ import annotations

import math
import statistics


def describe(values) -> dict | None:
  vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
  if not vals:
    return None
  vals.sort()
  n = len(vals)
  mean = statistics.fmean(vals)
  sd = statistics.stdev(vals) if n > 1 else 0.0
  return {
      "n": n,
      "median": statistics.median(vals),
      "min": vals[0],
      "max": vals[-1],
      "mean": mean,
      "stdev": sd,
      "cv": (sd / mean) if mean else None,
      "p10": quantile(vals, 0.10),
      "p90": quantile(vals, 0.90),
  }


def quantile(sorted_vals: list[float], q: float) -> float:
  if len(sorted_vals) == 1:
    return sorted_vals[0]
  pos = (len(sorted_vals) - 1) * q
  lo = math.floor(pos)
  hi = min(lo + 1, len(sorted_vals) - 1)
  return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def interp(xs: list[float], ys: list[float], x: float) -> float:
  """Piecewise-linear interpolation on sorted xs (clamped at the ends)."""
  if x <= xs[0]:
    return ys[0]
  if x >= xs[-1]:
    return ys[-1]
  lo, hi = 0, len(xs) - 1
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if xs[mid] <= x:
      lo = mid
    else:
      hi = mid
  span = xs[hi] - xs[lo]
  return ys[lo] if span == 0 else ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / span


def mad_outliers(values: list[float], threshold: float = 10.0) -> list[int]:
  """Indices whose modified z-score exceeds `threshold` robust sigmas
  (hyperfine's rule: |x - median| / (1.4826 * MAD) > 10)."""
  if len(values) < 3:
    return []
  med = statistics.median(values)
  mad = statistics.median([abs(v - med) for v in values])
  if mad == 0:
    return []
  return [i for i, v in enumerate(values) if abs(v - med) / (1.4826 * mad) > threshold]


def bootstrap_ratio_ci(a: list[float], b: list[float], iters: int = 2000,
                       seed: int = 0) -> tuple[float, float, float]:
  """Median(a)/median(b) with a percentile-bootstrap 95% interval."""
  import random
  rng = random.Random(seed)
  point = statistics.median(a) / statistics.median(b)
  ratios = []
  for _ in range(iters):
    ra = statistics.median(rng.choices(a, k=len(a)))
    rb = statistics.median(rng.choices(b, k=len(b)))
    ratios.append(ra / rb)
  ratios.sort()
  return point, quantile(ratios, 0.025), quantile(ratios, 0.975)
