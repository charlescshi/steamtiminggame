"""Sparse prior on the genre distance, and a check of what it implies before seeing any choices.

Genre distance between games i and j (agreed 24 Nov 2026):

    d(x_i, x_j)^2 = sum_k rho_k (x_ik - x_jk)^2

x_i is game i's vector of tag shares (446 tags, at most 20 nonzero, summing to 1), with
each tag **rescaled to [0, 1]**: divided by that tag's largest share across all games
(chosen 24 Nov 2026; the SAAS prior was set for inputs on [0, 1]). See `tag_scale`.
rho_k >= 0 says how much tag k counts when judging whether two games are alike.

Prior: the "sparse axis-aligned subspace" (SAAS) prior of Eriksson and Jankowiak (2021),
the paper cited in the proposal:

    tau   ~ HalfCauchy(0.1)      global scale, shared by all tags
    rho_k ~ HalfCauchy(tau)      one weight per tag

The half-Cauchy puts most of its mass near zero (most tags ignored) but has a heavy tail
(a few tags can matter a lot). The global tau lets the data decide how many tags matter.

Units: in the spacetime distance r^2 = (dt / ell)^2 + d^2, a genre distance d = 1 is as much
separation as being ell days apart in time.

The `__main__` block is a prior predictive check. It draws rho from the prior and reports
the genre distances it implies between real games. Nothing here looks at release dates.

Usage: python scripts/repulsion_prior.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import Predictive

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, load_games, tag_matrix, tag_names  # noqa: E402

TAU_SCALE = 0.1  # as in Eriksson and Jankowiak (2021)


def genre_distance_prior(n_tags):
    """NumPyro sample sites for the tag weights rho (shape n_tags)."""
    tau = numpyro.sample("tau", dist.HalfCauchy(TAU_SCALE))
    rho = numpyro.sample("rho", dist.HalfCauchy(tau).expand([n_tags]).to_event(1))
    return rho


def tag_scale(X):
    """Per-tag factor that rescales shares to [0, 1]: 1 / (largest share of that tag across games).

    Multiply each column of the share matrix by this before computing distances.
    """
    return 1.0 / np.asarray(X.max(axis=0).todense()).ravel()


def genre_distance_sq(xa, xb, rho):
    """Squared genre distance between rows of xa and rows of xb (same number of rows)."""
    return jnp.sum(rho * (xa - xb) ** 2, axis=-1)


def main():
    rng = np.random.default_rng(5105)
    g = load_games()  # all 2015-2025 games; no review filter
    X, tagids = tag_matrix(g)
    names = tag_names()
    tag_label = np.array([names.get(t, str(t)) for t in tagids])
    K = X.shape[1]

    # Reference games: Silksong, Hollow Knight, and the most-reviewed game whose top tag is Sports.
    tags = pd.read_parquet(os.path.join(os.path.dirname(OUT), "data", "game_tags.parquet"))
    top = tags[tags["rank"] == 0].merge(g[["appid", "name", "review_count"]], on="appid")
    sports = top[top["tag"] == "Sports"].sort_values("review_count", ascending=False).iloc[0]
    ref = {"Silksong": 1030300, "Hollow Knight": 367520, sports["name"]: int(sports["appid"])}
    row = pd.Series(np.arange(len(g)), index=g["appid"].values)
    xref = {k: np.asarray(X[row[a]].todense()).ravel() for k, a in ref.items()}

    # Random pairs of games
    n_pairs = 20000
    ia, ib = rng.integers(0, len(g), n_pairs), rng.integers(0, len(g), n_pairs)
    keep = ia != ib
    Xa, Xb = X[ia[keep]].toarray(), X[ib[keep]].toarray()

    # The chosen input scaling (each tag to [0, 1]), with raw shares kept for comparison.
    scalings = {"each tag scaled to [0, 1] (chosen)": tag_scale(X), "raw shares": np.ones(K)}

    draws = Predictive(genre_distance_prior, num_samples=4000)(jax.random.PRNGKey(0), K)
    rho = np.asarray(draws["rho"])  # (draws, K)
    tau = np.asarray(draws["tau"])

    rows = []
    for label, s in scalings.items():
        rs = rho * s ** 2  # scaling the inputs by s is the same as scaling rho by s^2
        d_sk_hk = np.sqrt(np.sum(rs * (xref["Silksong"] - xref["Hollow Knight"]) ** 2, axis=1))
        d_sk_sp = np.sqrt(np.sum(rs * (xref["Silksong"] - xref[sports["name"]]) ** 2, axis=1))
        sub = rng.choice(len(rho), 400, replace=False)  # random pairs on a subset of draws (cost)
        d_rand = np.sqrt(np.einsum("sk,pk->sp", rs[sub], (Xa - Xb) ** 2))
        top10 = np.sort(rs, axis=1)[:, -10:].sum(axis=1) / rs.sum(axis=1)
        for what, v in [("Silksong to Hollow Knight", d_sk_hk),
                        (f"Silksong to {sports['name']}", d_sk_sp),
                        ("random pair of games", d_rand.ravel()),
                        ("share of weight on the top 10 tags", top10)]:
            q = np.percentile(v, [10, 50, 90])
            rows.append({"inputs": label, "quantity": what, "p10": q[0], "median": q[1], "p90": q[2]})
        closer = np.mean(d_sk_hk < d_sk_sp)
        rows.append({"inputs": label, "quantity": f"P(Hollow Knight closer than {sports['name']})",
                     "p10": np.nan, "median": closer, "p90": np.nan})

    res = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "genre_distance_prior_check.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"{len(g)} games, {K} tags; tau prior HalfCauchy({TAU_SCALE}); 4000 prior draws")
    print(f"tau draws: median {np.median(tau):.3f}, 90th percentile {np.percentile(tau, 90):.3f}")
    print(f"far reference game: {sports['name']} (appid {sports['appid']})")
    print("Silksong top tags:", ", ".join(f"{t} {v:.3f}" for t, v in
                                         sorted(zip(tag_label, xref['Silksong']), key=lambda z: -z[1])[:5]))
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
