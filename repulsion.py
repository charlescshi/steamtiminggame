"""Repulsion R_i(t): prior, day-by-day computation, and a prior predictive check.

Agreed 24 Nov 2026 (see README, "Model being built"):

    R_i(t) = sum_{j != i} M_j^a * h(r_ij(t)) * w(t - t_j)
    r_ij(t)^2 = ((t - t_j) / ell)^2 + d(x_i, x_j)^2,   ell = ell_after if t > t_j else ell_before

  * t is a candidate release day of game i, t_j the rival's actual release day (both in days).
  * d is the sparse genre distance from repulsion_prior.py (tags rescaled to [0, 1]).
  * ell_after = 7 days, fixed. This sets the units: r = 1 is one week after a same-genre rival.
  * ell_before is learned.
  * h has a free shape and either sign (positive = push away, negative = cluster). Prior: a
    zero-mean Gaussian process over r with squared exponential kernel; its size and smoothness
    are learned. h is stored on a grid of r values and interpolated in between.
  * w fades rivals out: 1 up to 60 days away, smoothly to 0 at 90 days, 0 beyond.
  * beta = 1.
  * Mass (chosen 24 Nov 2026, "for now"): the rival's review count stands in for its size,
    M_j = (1 + reviews_j) / 1000, raised to a learned power a. A rival with 1,000 reviews has
    weight 1 whatever a is. Review count is measured after release, unlike the proposal's
    pre-release followers.

Hyperpriors here are first-shot defaults, to be checked against the prior predictive output:
  ell_before ~ LogNormal(log 7, 1)   centred on symmetry; 95% between about 1 and 50 days
  a          ~ Normal(1, 0.5)        centred on gravity's linear mass
  lambda_h   ~ HalfNormal(s)         size of h (note 9's lambda) per unit mass; s set by the check below
  ell_h      ~ LogNormal(0, 0.5)     length scale of h (note 9's ell), in r units (median: one week)

Usage: python scripts/repulsion.py   (runs the prior predictive check on Silksong's 2025)
"""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import Predictive  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AQUA, BLUE, INK, INK_2, ORANGE, OUT, load_games, style_axes, tag_matrix  # noqa: E402
from repulsion_prior import genre_distance_prior, tag_scale  # noqa: E402

ELL_AFTER = 7.0  # days; fixes the units of r
FLAT, HORIZON = 60.0, 90.0  # days: full weight up to FLAT, zero from HORIZON
R_GRID = jnp.linspace(0.0, 30.0, 121)  # h is stored at these r values; beyond 30 it stays at h(30)
SILKSONG = 1030300
M_REF = 1000.0  # reviews; a rival this size has mass 1


def mass(review_count):
    return (1.0 + np.asarray(review_count, dtype=float)) / M_REF


def repulsion_prior(n_tags, size_scale=1.0):
    """NumPyro sample sites for everything in R. Returns (rho, ell_before, a, h_grid)."""
    rho = genre_distance_prior(n_tags)
    ell_before = numpyro.sample("ell_before", dist.LogNormal(jnp.log(ELL_AFTER), 1.0))
    a = numpyro.sample("a", dist.Normal(1.0, 0.5))
    lambda_h = numpyro.sample("lambda_h", dist.HalfNormal(size_scale))
    ell_h = numpyro.sample("ell_h", dist.LogNormal(0.0, 0.5))
    z = numpyro.sample("z_h", dist.Normal(0.0, 1.0).expand([R_GRID.size]).to_event(1))
    gap = R_GRID[:, None] - R_GRID[None, :]
    K = jnp.exp(-0.5 * gap ** 2 / ell_h ** 2) + 1e-6 * jnp.eye(R_GRID.size)
    h_grid = numpyro.deterministic("h_grid", lambda_h * (jnp.linalg.cholesky(K) @ z))
    return rho, ell_before, a, h_grid


def taper(dt):
    """Weight of a rival dt days away: 1 within FLAT days, fading to 0 at HORIZON."""
    a = jnp.abs(dt)
    fade = jnp.cos(0.5 * jnp.pi * jnp.clip((a - FLAT) / (HORIZON - FLAT), 0.0, 1.0)) ** 2
    return jnp.where(a >= HORIZON, 0.0, fade)


@jax.jit
def repulsion_profile(days, x_i, X_r, t_r, m_r, rho, ell_before, a, h_grid):
    """R_i(t) for every candidate day in `days`.

    days (T,) candidate days for game i; x_i (K,) its rescaled tag shares;
    X_r (N, K), t_r (N,), m_r (N,) rivals' rescaled shares, release days and masses
    (game i itself excluded).
    """
    d2 = jnp.sum(rho * (X_r - x_i) ** 2, axis=1)  # (N,)
    dt = days[:, None] - t_r[None, :]  # (T, N); > 0 means i launches after rival j
    ell = jnp.where(dt > 0, ELL_AFTER, ell_before)
    r = jnp.sqrt((dt / ell) ** 2 + d2[None, :])
    h = jnp.interp(jnp.clip(r, 0.0, R_GRID[-1]), R_GRID, h_grid)
    return jnp.sum(m_r[None, :] ** a * h * taper(dt), axis=1)


def day_number(dates):
    return (pd.to_datetime(pd.Series(dates)) - pd.Timestamp("2015-01-01")).dt.days.values


def main():
    os.makedirs(OUT, exist_ok=True)
    g = load_games(first_year=2015, last_year=2026)
    X, tagids = tag_matrix(g)
    X = X.multiply(tag_scale(X)).tocsr()  # rescale each tag to [0, 1]
    t_all = day_number(g["date"])

    # Silksong and its rivals: everyone within HORIZON days of its 2025 candidate window
    i = int(np.flatnonzero(g["appid"].values == SILKSONG)[0])
    days = np.arange(day_number(["2025-01-01"])[0], day_number(["2025-12-31"])[0] + 1)
    in_win = (t_all >= days[0] - HORIZON) & (t_all <= days[-1] + HORIZON)
    in_win[i] = False
    X_r = jnp.asarray(X[in_win].toarray())
    t_r = jnp.asarray(t_all[in_win], dtype=float)
    m_r = jnp.asarray(mass(g["review_count"].values[in_win]))
    x_i = jnp.asarray(X[i].toarray().ravel())
    t_sk = t_all[i]
    jd = jnp.asarray(days, dtype=float)
    print(f"Silksong: {len(days)} candidate days, {in_win.sum():,} rivals within {HORIZON:.0f} days of 2025")
    print(f"rival masses (reviews / {M_REF:.0f}): median {np.median(m_r):.3f}, 99th pct {np.percentile(m_r, 99):.1f}, "
          f"max {np.max(m_r):.0f}")

    scales = [1.0, 0.1, 0.01, 0.001]
    rows, keep = [], {}
    for k, s in enumerate(scales):
        draws = Predictive(lambda n: repulsion_prior(n, s), num_samples=200)(jax.random.PRNGKey(k), X.shape[1])
        prof = np.stack([np.asarray(repulsion_profile(jd, x_i, X_r, t_r, m_r, draws["rho"][d], draws["ell_before"][d],
                                                      draws["a"][d], draws["h_grid"][d])) for d in range(200)])
        spread = prof.max(axis=1) - prof.min(axis=1)
        q = np.percentile(spread, [10, 50, 90])
        rows.append({"prior on size of h": f"HalfNormal({s:g})", "range p10": q[0], "range median": q[1], "range p90": q[2],
                     "median most/least likely day ratio": float(np.exp(min(q[1], 700)))})
        keep[s] = (prof, spread)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "repulsion_prior_check_mass.csv"), index=False)
    pd.set_option("display.width", 200)
    print("Range of R over Silksong's 2025, in utility units (prior draws):")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # Figure: the median-spread draw for each prior size, one panel each
    fig, axes = plt.subplots(len(scales), 1, figsize=(8, 7), dpi=200, sharex=True)
    dates = pd.Timestamp("2015-01-01") + pd.to_timedelta(days, unit="D")
    for ax, s in zip(axes, scales):
        prof, spread = keep[s]
        d = int(np.argsort(spread)[len(spread) // 2])
        style_axes(ax)
        ax.plot(dates, prof[d] - prof[d].mean(), color=BLUE, lw=1.2)
        ax.axvline(pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(t_sk)), color=INK_2, lw=1, ls=":")
        ax.set_title(f"Size of h ~ HalfNormal({s:g})", loc="left", fontsize=9, color=INK)
        ax.set_ylabel("R(t) − mean", color=INK_2, fontsize=8)
    axes[-1].set_xlabel("Day in 2025 (dotted: actual launch)", color=INK_2, fontsize=9)
    fig.suptitle("Simulated repulsion for Silksong, by prior on the size of h", x=0.02, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_repulsion_prior_mass.png"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
