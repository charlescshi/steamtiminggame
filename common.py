"""Shared sample definition: which games enter, and how dates become weeks.

Choices (all open to revision, see README):
  * release date = `steam_release_date` (the date on the store page), read in US Pacific
    time, since Steam launches are scheduled around 10am PT;
  * week = ISO week of that date, with week 53 (2015, 2020) folded into week 52;
  * year = ISO year, so every game chooses among 52 weeks of one year;
  * games released 2015-2025, available in the US store, at least one tag, not
    "coming soon".
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
N_WEEKS = 52

# Chart colours: reference palette slots 1-3 and the blue sequential ramp.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_games(first_year=2015, last_year=2025, min_reviews=0, date_col="steam_release_date"):
    g = pd.read_parquet(os.path.join(DATA, "games.parquet"))
    g = g[(g["success"] == 1) & (g["type"] == 0) & g[date_col].notna() & (g["n_tags"] > 0)]
    g = g[~g["is_coming_soon"].eq(True)]
    local = g[date_col].dt.tz_convert("America/Los_Angeles")
    iso = local.dt.isocalendar()
    g = g.assign(date=local.dt.date, year=iso["year"].astype(int), week=iso["week"].astype(int).clip(upper=N_WEEKS))
    g = g[(g["year"] >= first_year) & (g["year"] <= last_year) & (g["review_count"] >= min_reviews)]
    return g.reset_index(drop=True)


def tag_names():
    return {t["tagid"]: t["name"] for t in json.load(open(os.path.join(DATA, "tags.json"), encoding="utf-8"))["tags"]}


def tag_matrix(games):
    """Sparse (games x tags) matrix of tag shares, rows aligned with `games`, plus the tag ids."""
    t = pd.read_parquet(os.path.join(DATA, "game_tags.parquet"))
    t = t[t["appid"].isin(games["appid"])]
    tagids = np.sort(t["tagid"].unique())
    col = pd.Series(np.arange(len(tagids)), index=tagids)
    row = pd.Series(np.arange(len(games)), index=games["appid"].values)
    X = sp.csr_matrix((t["share"].values, (row[t["appid"]].values, col[t["tagid"]].values)),
                      shape=(len(games), len(tagids)))
    return X, tagids


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def month_ticks(ax):
    """Label ISO weeks with the month they start in (approximately)."""
    starts = [1, 5, 9, 14, 18, 22, 27, 31, 36, 40, 44, 49]
    ax.set_xticks(starts)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
