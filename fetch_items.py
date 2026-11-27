"""Fetch release dates, weighted user tags and review counts for every Steam game.

Source: the keyless store endpoint IStoreBrowseService/GetItems, 200 appids per
request (500 is refused as too long). The appid list is the typed game catalogue
already saved by the Microtransactions project (IStoreService/GetAppList, 5 Sep 2026),
so this script sends no API key and never touches the panel collector's budget.

Every batch is committed with its raw gzip response, so later parsing choices never
need a re-scrape. The run is resumable: it only asks for batches not yet `ok`.

Safety:
  * at least SPACING seconds between requests (the first run at 1.1 s was refused with
    HTTP 429 after 68 requests in about two minutes, so the default is now 4 s);
  * on HTTP 429: waits COOLDOWN, then retries the same batch with 1.5x the spacing;
    stops for the day after MAX_429 refusals;
  * stops after 3 batches in a row fail;
  * sends nothing after --stop-at (default 02:00 local), the same quiet period the
    Microtransactions `game_daily` job keeps before the 03:00 panel wave;
  * before starting, waits while the Microtransactions `game_daily` job is running,
    because both use keyless requests to api.steampowered.com.

Usage (from the repo root):
  python scripts/fetch_items.py init       # load the catalogue into the batch table
  python scripts/fetch_items.py tags       # save the tag list (one request)
  python scripts/fetch_items.py run [--limit N] [--no-wait]
  python scripts/fetch_items.py status
"""
import argparse
import glob
import gzip
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "steam_items.sqlite")
TAGS_JSON = os.path.join(ROOT, "data", "tags.json")
CATALOG = r"C:\Users\ice-o\Documents\GitHub\Research\Microtransactions\analysis\measurement\runs\2026-11-12\catalog"
GAME_DAILY_DB = r"C:\Users\ice-o\Documents\GitHub\Research\Microtransactions\scraper\game_daily.sqlite"

ITEMS_URL = "https://api.steampowered.com/IStoreBrowseService/GetItems/v1?input_json="
TAGS_URL = "https://api.steampowered.com/IStoreService/GetTagList/v1/?language=english"
BATCH = 200
SPACING = 4.0
MAX_SPACING = 30.0
COOLDOWN = 15 * 60
MAX_429 = 4
RETRY_WAITS = (5, 20, 60)
MAX_CONSECUTIVE_FAILURES = 3
DATA_REQUEST = {
    "include_tag_count": 20,
    "include_release": True,
    "include_reviews": True,
    "include_basic_info": True,
}


class RateLimited(Exception):
    pass


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS batches (
            batch_id    INTEGER PRIMARY KEY,
            appids      TEXT NOT NULL,              -- JSON list of up to 200 appids
            status      TEXT NOT NULL DEFAULT 'pending',  -- pending | ok | error
            http_status INTEGER,
            n_items     INTEGER,
            fetched_at  TEXT,
            body        BLOB                        -- gzip of the raw JSON response
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """
    )
    return conn


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return r.status, body


def fetch_with_retries(url):
    """Return (status, body). Raise RateLimited on 429; return (code, None) after retries."""
    last = None
    for attempt, wait in enumerate((0,) + RETRY_WAITS):
        if wait:
            time.sleep(wait)
        try:
            return http_get(url)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimited()
            last = e.code
            if 400 <= e.code < 500:
                break  # a client error will not fix itself
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = -1
    return last, None


def cmd_init(_args):
    conn = connect()
    if conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]:
        log("batches already loaded; nothing to do")
        return 0
    appids = []
    files = sorted(glob.glob(os.path.join(CATALOG, "game_*.json.gz")))
    fetched = None
    for f in files:
        d = json.load(gzip.open(f))
        fetched = d.get("fetched_at", fetched)
        appids += [a["appid"] for a in d["response"]["apps"]]
    appids = sorted(set(appids))
    rows = [(i, json.dumps(appids[k:k + BATCH])) for i, k in enumerate(range(0, len(appids), BATCH))]
    with conn:
        conn.executemany("INSERT INTO batches(batch_id, appids) VALUES(?, ?)", rows)
        conn.execute("INSERT OR REPLACE INTO meta VALUES('catalog_dir', ?)", (CATALOG,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES('catalog_fetched_at', ?)", (str(fetched),))
        conn.execute("INSERT OR REPLACE INTO meta VALUES('n_appids', ?)", (str(len(appids)),))
    log(f"loaded {len(appids)} appids into {len(rows)} batches")
    return 0


def cmd_tags(_args):
    status, body = http_get(TAGS_URL)
    tags = json.loads(body)["response"]["tags"]
    os.makedirs(os.path.dirname(TAGS_JSON), exist_ok=True)
    with open(TAGS_JSON, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": now_utc(), "tags": tags}, f, indent=1)
    log(f"saved {len(tags)} tags to {TAGS_JSON}")
    return 0


def game_daily_running():
    """True if the Microtransactions game_daily job has a run marked running in the last 6 hours."""
    if not os.path.exists(GAME_DAILY_DB):
        return False
    try:
        con = sqlite3.connect(f"file:{GAME_DAILY_DB}?mode=ro", uri=True, timeout=30)
        rows = con.execute("SELECT started_at FROM runs WHERE status = 'running'").fetchall()
        con.close()
    except sqlite3.Error:
        return True  # unsure: be conservative
    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
    return any(datetime.fromisoformat(s) > cutoff for (s,) in rows)


def stop_time(hhmm):
    """The next local datetime at hh:mm."""
    h, m = map(int, hhmm.split(":"))
    now = datetime.now()
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def cmd_run(args):
    conn = connect()
    if not args.no_wait:
        while game_daily_running():
            log("game_daily is running; waiting 5 minutes")
            time.sleep(300)
    # Scrambled order (389 is coprime to the 924 batches): a run that stops early leaves a
    # random sample of games rather than only the oldest appids.
    pending = conn.execute(
        "SELECT batch_id, appids FROM batches WHERE status != 'ok' ORDER BY (batch_id * 389) % 924, batch_id"
    ).fetchall()
    if args.limit:
        pending = pending[: args.limit]
    deadline = stop_time(args.stop_at)
    spacing = args.spacing
    log(f"{len(pending)} batches to fetch; spacing {spacing:.1f}s; no requests after {deadline:%Y-%m-%d %H:%M}")
    failures, refusals, next_at = 0, 0, 0.0
    n = 0
    while n < len(pending):
        batch_id, appids_json = pending[n]
        wait = next_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if datetime.now() >= deadline:
            log(f"reached the {args.stop_at} stop time with {len(pending) - n} batches left; re-run to resume")
            return 0
        req = {
            "ids": [{"appid": a} for a in json.loads(appids_json)],
            "context": {"language": "english", "country_code": "US"},
            "data_request": DATA_REQUEST,
        }
        url = ITEMS_URL + urllib.parse.quote(json.dumps(req, separators=(",", ":")))
        try:
            status, body = fetch_with_retries(url)
        except RateLimited:
            refusals += 1
            if refusals > MAX_429:
                log(f"HTTP 429 at batch {batch_id}, refusal {refusals}; stopping for today. Re-run later to resume.")
                return 1
            spacing = min(spacing * 1.5, MAX_SPACING)
            log(f"HTTP 429 at batch {batch_id} (refusal {refusals} of {MAX_429}); "
                f"waiting {COOLDOWN // 60} min, then spacing {spacing:.1f}s")
            next_at = time.monotonic() + COOLDOWN
            continue
        next_at = time.monotonic() + spacing
        n += 1
        if body is None:
            failures += 1
            with conn:
                conn.execute(
                    "UPDATE batches SET status='error', http_status=?, fetched_at=? WHERE batch_id=?",
                    (status, now_utc(), batch_id),
                )
            log(f"batch {batch_id} failed (status {status})")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                log("3 failures in a row; stopping")
                return 1
            continue
        failures = 0
        items = json.loads(body).get("response", {}).get("store_items", [])
        with conn:
            conn.execute(
                "UPDATE batches SET status='ok', http_status=?, n_items=?, fetched_at=?, body=? WHERE batch_id=?",
                (status, len(items), now_utc(), gzip.compress(body), batch_id),
            )
        if n % 50 == 0 or n == len(pending):
            log(f"{n}/{len(pending)} batches done")
    return 0


def cmd_status(_args):
    conn = connect()
    for status, count in conn.execute("SELECT status, COUNT(*) FROM batches GROUP BY status"):
        print(f"{status:8s} {count}")
    print("items:", conn.execute("SELECT COALESCE(SUM(n_items), 0) FROM batches").fetchone()[0])
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("tags")
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=None, help="fetch at most this many batches")
    r.add_argument("--no-wait", action="store_true", help="do not wait for game_daily to finish")
    r.add_argument("--spacing", type=float, default=SPACING, help="seconds between requests")
    r.add_argument("--stop-at", default="02:00", help="local time after which no request is sent")
    sub.add_parser("status")
    args = p.parse_args()
    return {"init": cmd_init, "tags": cmd_tags, "run": cmd_run, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
