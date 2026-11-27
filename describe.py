"""Descriptive facts and figures for the status report.

outputs/sample_by_year.csv          games per year, all and with >= 10 reviews
outputs/fig_week_by_year.png        share of each year's releases in each week (does the calendar repeat?)
outputs/fig_week_profile_tags.png   average weekly profile: all games vs Horror (the Halloween check)
outputs/fig_silksong_window.png     Metroidvania releases around Silksong's launch vs the 2022-24 pattern

Usage: python scripts/describe.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AQUA, BLUE, BLUE_RAMP, INK, INK_2, N_WEEKS, ORANGE, OUT, load_games, month_ticks,  # noqa: E402
                    style_axes, tag_names)

SILKSONG = 1030300


def has_tag(games, name):
    t = pd.read_parquet(os.path.join(os.path.dirname(OUT), "data", "game_tags.parquet"))
    return games["appid"].isin(t.loc[t["tag"] == name, "appid"])


def weekly_share(games):
    c = games.groupby(["year", "week"]).size().unstack(fill_value=0).reindex(columns=range(1, N_WEEKS + 1), fill_value=0)
    return c.div(c.sum(axis=1), axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    g = load_games()
    g10 = load_games(min_reviews=10)

    by_year = pd.DataFrame({
        "games": g.groupby("year").size(),
        "games_10plus_reviews": g10.groupby("year").size(),
        "early_access_now": g.groupby("year")["is_early_access"].apply(lambda s: int((s == True).sum())),  # noqa: E712
        "median_reviews": g.groupby("year")["review_count"].median(),
    })
    by_year.to_csv(os.path.join(OUT, "sample_by_year.csv"))
    print(by_year.to_string())
    print(f"total: {len(g)} games, {len(g10)} with >= 10 reviews")

    # 1. Week-by-year heatmap of release shares (games with >= 10 reviews).
    share = weekly_share(g10)
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP[1:])
    im = ax.imshow(share.values * 100, aspect="auto", cmap=cmap, interpolation="nearest",
                   extent=(0.5, N_WEEKS + 0.5, share.index.max() + 0.5, share.index.min() - 0.5))
    ax.set_yticks(share.index)
    ax.tick_params(colors=INK_2, labelsize=8)
    month_ticks(ax)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% of the year's releases", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Share of each year's releases by ISO week (games with 10+ reviews)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_week_by_year.png"))
    plt.close(fig)

    # 2. Average weekly profile: all games vs Horror.
    horror = g10[has_tag(g10, "Horror")]
    prof_all = weekly_share(g10).mean(axis=0) * 100
    prof_h = weekly_share(horror).mean(axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
    style_axes(ax)
    ax.plot(prof_all.index, prof_all.values, color=BLUE, lw=2, label="All games")
    ax.plot(prof_h.index, prof_h.values, color=ORANGE, lw=2, label="Horror")
    month_ticks(ax)
    ax.set_xlim(0.5, N_WEEKS + 0.5)
    ax.set_ylabel("% of the year's releases", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"Average release week, 2015-2025 (games with 10+ reviews; {len(horror):,} horror)",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_week_profile_tags.png"))
    plt.close(fig)

    # 3. Silksong window: Metroidvania releases by week, 2025 vs the 2022-24 pattern.
    sk = g[g["appid"] == SILKSONG]
    if len(sk):
        sk_week = int(sk["week"].iloc[0])
        mv = g[has_tag(g, "Metroidvania") & (g["appid"] != SILKSONG)]  # leave Silksong itself out
        counts = mv.groupby(["year", "week"]).size().unstack(fill_value=0).reindex(columns=range(1, N_WEEKS + 1), fill_value=0)
        base_years = [y for y in (2022, 2023, 2024) if y in counts.index]
        base_share = counts.loc[base_years].div(counts.loc[base_years].sum(axis=1), axis=0).mean(axis=0)
        actual = counts.loc[2025]
        expected = base_share * actual.sum()
        weeks = np.arange(max(1, sk_week - 10), min(N_WEEKS, sk_week + 10) + 1)
        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=200)
        style_axes(ax)
        ax.plot(weeks - sk_week, expected.reindex(weeks).values, color=INK_2, lw=2, ls="--",
                label="Expected from 2022-24 calendar")
        ax.plot(weeks - sk_week, actual.reindex(weeks, fill_value=0).values, color=BLUE, lw=2, marker="o", ms=4,
                label="2025 actual (Silksong excluded)")
        top = ax.get_ylim()[1]
        ax.axvline(0, color=ORANGE, lw=1.5)
        ax.text(0.2, top, "Silksong launch", color=INK_2, fontsize=8, va="top")
        ax.axvline(-2, color=AQUA, lw=1, ls=":")
        ax.text(-2.2, top, "date announced", color=INK_2, fontsize=8, va="top", ha="right")
        ax.set_xticks(range(weeks[0] - sk_week, weeks[-1] - sk_week + 1, 2))
        ax.set_xlabel("Weeks from Silksong's launch week (4 Sep 2025)", color=INK_2, fontsize=9)
        ax.set_ylabel("Metroidvania releases", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
        ax.set_title("Metroidvania releases around Silksong's launch", loc="left", fontsize=10, color=INK, pad=22)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_silksong_window.png"))
        plt.close(fig)
        win = pd.DataFrame({"actual": actual.reindex(weeks, fill_value=0), "expected": expected.reindex(weeks).round(1)})
        win.index = win.index - sk_week
        win.to_csv(os.path.join(OUT, "silksong_window.csv"))
        print("Silksong ISO week", sk_week, "\n", win.T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
