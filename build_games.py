"""Parse the raw GetItems responses into two tables.

data/games.parquet      one row per game: dates, review counts, developer/publisher
data/game_tags.parquet  one row per (game, tag): vote weight and share of the game's votes

Share = a tag's vote weight / the sum of the game's (up to 20) tag weights, so each
game's shares sum to 1. This is the genre vector x_i in the proposal.

Dates are kept raw (all the release fields Steam returns). Which one counts as "the"
release date (first Early Access launch or 1.0) is a modelling choice made downstream.

Usage: python scripts/build_games.py
"""
import gzip
import json
import os
import sqlite3
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "steam_items.sqlite")
TAGS_JSON = os.path.join(ROOT, "data", "tags.json")
RELEASE_FIELDS = [
    "steam_release_date",
    "original_release_date",
    "release_from_early_access_date",
    "is_early_access",
    "is_coming_soon",
    "is_preload",
    "custom_release_date_message",
    "coming_soon_display",
]


def to_ts(v):
    """Unix seconds -> UTC timestamp; 0 or missing -> NaT."""
    if v in (None, 0, "0", ""):
        return pd.NaT
    return pd.to_datetime(int(v), unit="s", utc=True)


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    games, tag_rows, seen = [], [], set()
    for (body,) in con.execute("SELECT body FROM batches WHERE status = 'ok' ORDER BY batch_id"):
        for it in json.loads(gzip.decompress(body)).get("response", {}).get("store_items", []):
            appid = it.get("appid") or it.get("id")
            if appid in seen:
                continue
            seen.add(appid)
            rel = it.get("release") or {}
            rev = (it.get("reviews") or {}).get("summary_filtered") or {}
            info = it.get("basic_info") or {}
            tags = it.get("tags") or []
            total = sum(t.get("weight", 0) for t in tags)
            row = {
                "appid": appid,
                "name": it.get("name"),
                "success": it.get("success"),
                "visible": it.get("visible"),
                "type": it.get("type"),
                "is_free": bool(it.get("is_free", False)),
                "review_count": rev.get("review_count", 0),
                "percent_positive": rev.get("percent_positive"),
                "developers": "; ".join(d.get("name", "") for d in info.get("developers", []) or []),
                "publishers": "; ".join(d.get("name", "") for d in info.get("publishers", []) or []),
                "n_tags": len(tags),
                "tag_weight_total": total,
            }
            for f in RELEASE_FIELDS:
                v = rel.get(f)
                row[f] = to_ts(v) if f.endswith("_date") else v
            games.append(row)
            for rank, t in enumerate(tags):
                tag_rows.append({
                    "appid": appid,
                    "tagid": t["tagid"],
                    "rank": rank,
                    "weight": t.get("weight", 0),
                    "share": t.get("weight", 0) / total if total else 0.0,
                })
    games = pd.DataFrame(games)
    tags = pd.DataFrame(tag_rows)
    names = {t["tagid"]: t["name"] for t in json.load(open(TAGS_JSON, encoding="utf-8"))["tags"]}
    tags["tag"] = tags["tagid"].map(names)
    games.to_parquet(os.path.join(ROOT, "data", "games.parquet"), index=False)
    tags.to_parquet(os.path.join(ROOT, "data", "game_tags.parquet"), index=False)
    ok = games[games["success"] == 1]
    print(f"{len(games)} items parsed, {len(ok)} available in the US store, {len(tags)} game-tag rows")
    print(f"tags without a name in the tag list: {tags['tag'].isna().sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
