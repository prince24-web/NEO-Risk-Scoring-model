"""
Monarch Industries | NEO Risk Scorer
=====================================
Step 1: Scraper — pulls the full NEO catalog from NASA NeoWs API
         and stores raw data in SQLite.

Two-phase design:
  Phase 1 (browse)   — pulls /neo/browse, stores core orbital data for every object
  Phase 2 (hydrate)  — pulls /neo/{id} for each object to get full close-approach history

Usage:
  python scraper.py --phase browse              # pull full catalog
  python scraper.py --phase hydrate             # enrich with CA data
  python scraper.py --phase browse --max 100    # test run, first 100 objects
"""

import os
import time
import json
import logging
import argparse
import sqlite3
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).with_name(".env"))
API_KEY     = os.getenv("NASA_API_KEY", "DEMO_KEY")   # set env var or replace
BASE_URL    = "https://api.nasa.gov/neo/rest/v1"
DB_PATH     = Path(__file__).parent / "neo_data.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
PAGE_SIZE        = 20    # NeoWs max per page
RATE_LIMIT       = 1.2   # seconds between requests (safe for NASA free tier burst limits)
MAX_RETRIES      = 5      # more attempts before giving up
RETRY_DELAY      = 10    # base seconds before retry on 429/5xx (multiplied per attempt)
COOLDOWN_AFTER_LIMIT = 60  # seconds to wait after rate limit exhausts all retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("neo_scraper")


# ── Database ──────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables from schema.sql if they don't exist."""
    conn = get_db()
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    log.info(f"Database ready: {DB_PATH}")


# ── HTTP ───────────────────────────────────────────────────────────────────────
def api_get(endpoint: str, params: dict = {}) -> dict | None:
    """GET with retry logic and rate limiting."""
    url = f"{BASE_URL}{endpoint}"
    params = {"api_key": API_KEY, **params}

    rate_limited = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 200:
                return resp.json()

            elif resp.status_code == 429:
                rate_limited = True
                # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                log.warning(f"Rate limited (429). Backing off {wait}s... (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)

            elif resp.status_code in (500, 502, 503):
                wait = RETRY_DELAY * attempt
                log.warning(f"Server error {resp.status_code}. Retrying in {wait}s...")
                time.sleep(wait)

            else:
                log.error(f"Unexpected status {resp.status_code} for {url}")
                return None

        except requests.RequestException as e:
            log.error(f"Request failed: {e}")
            time.sleep(RETRY_DELAY)

    # If we got here after repeated 429s, cool down hard before caller continues
    if rate_limited:
        log.warning(f"Rate limit persisted after {MAX_RETRIES} attempts. Cooling down {COOLDOWN_AFTER_LIMIT}s before resuming...")
        time.sleep(COOLDOWN_AFTER_LIMIT)

    log.error(f"All {MAX_RETRIES} attempts failed for {endpoint}")
    return None


# ── Parsers ────────────────────────────────────────────────────────────────────
def parse_neo_object(neo: dict) -> dict:
    """Extract and flatten fields from a raw NeoWs NEO object."""
    orb = neo.get("orbital_data", {})
    dia = neo.get("estimated_diameter", {}).get("kilometers", {})
    dia_min = dia.get("estimated_diameter_min")
    dia_max = dia.get("estimated_diameter_max")

    return {
        "id":                       neo.get("id"),
        "name":                     neo.get("name"),
        "absolute_magnitude_h":     neo.get("absolute_magnitude_h"),
        "diameter_min_km":          dia_min,
        "diameter_max_km":          dia_max,
        "diameter_mean_km":         round((dia_min + dia_max) / 2, 6) if dia_min and dia_max else None,
        "is_potentially_hazardous": int(neo.get("is_potentially_hazardous_asteroid", False)),
        "eccentricity":             _float(orb.get("eccentricity")),
        "semi_major_axis":          _float(orb.get("semi_major_axis")),
        "inclination":              _float(orb.get("inclination")),
        "perihelion_distance":      _float(orb.get("perihelion_distance")),
        "aphelion_distance":        _float(orb.get("aphelion_distance")),
        "orbital_period":           _float(orb.get("orbital_period")),
        "ascending_node_longitude": _float(orb.get("ascending_node_longitude")),
        "argument_of_perihelion":   _float(orb.get("argument_of_perihelion")),
        "mean_anomaly":             _float(orb.get("mean_anomaly")),
        "orbit_class":              orb.get("orbit_class", {}).get("orbit_class_type"),
        "first_observation_date":   orb.get("first_observation_date"),
        "last_observation_date":    orb.get("last_observation_date"),
    }


def parse_close_approach(neo_id: str, ca: dict) -> dict:
    """Parse a single close-approach entry."""
    vel = ca.get("relative_velocity", {})
    miss = ca.get("miss_distance", {})
    epoch_unix = ca.get("epoch_date_close_approach")

    return {
        "neo_id":                   neo_id,
        "epoch_date":               ca.get("close_approach_date"),
        "epoch_date_unix":          int(epoch_unix) if epoch_unix else None,
        "relative_velocity_km_s":   _float(vel.get("kilometers_per_second")),
        "miss_distance_km":         _float(miss.get("kilometers")),
        "miss_distance_au":         _float(miss.get("astronomical")),
        "miss_distance_lunar":      _float(miss.get("lunar")),
        "orbiting_body":            ca.get("orbiting_body", "Earth"),
    }


def _float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Phase 1: Browse ────────────────────────────────────────────────────────────
def run_browse(max_objects: int | None = None):
    """Pull all NEOs from the browse endpoint, store core orbital data."""
    conn = get_db()
    run_id = _start_run(conn, "browse")

    # ── Resume support ────────────────────────────────────────────────────────
    resume_page = int(_get_state(conn, "browse_last_page", default=-1)) + 1
    total_stored = int(_get_state(conn, "browse_total_stored", default=0))
    if resume_page > 0:
        log.info(f"Resuming from page {resume_page} ({total_stored} objects already stored)")
    else:
        log.info("Starting browse phase...")

    page        = resume_page
    total_pages = None
    total_errors = 0

    while True:
        data = api_get("/neo/browse", {"page": page, "size": PAGE_SIZE})

        if not data:
            total_errors += 1
            log.warning(f"Failed to fetch page {page}, will retry same page after cooldown.")
            time.sleep(COOLDOWN_AFTER_LIMIT)
            continue  # retry same page, don't increment

        # Learn total pages on first response
        if total_pages is None:
            total_pages = data["page"]["total_pages"]
            total_elements = data["page"]["total_elements"]
            log.info(f"Catalog: {total_elements} NEOs across {total_pages} pages")

        neos = data.get("near_earth_objects", [])

        for neo in neos:
            try:
                record = parse_neo_object(neo)
                _upsert_neo(conn, record)

                # Also store any CA data included in browse response (usually sparse)
                for ca in neo.get("close_approach_data", []):
                    ca_record = parse_close_approach(record["id"], ca)
                    _insert_ca(conn, ca_record)

                total_stored += 1
            except Exception as e:
                log.error(f"Error parsing NEO {neo.get('id')}: {e}")
                total_errors += 1

        # ── Save progress after every page ───────────────────────────────────
        _set_state(conn, "browse_last_page", page)
        _set_state(conn, "browse_total_stored", total_stored)
        conn.commit()
        log.info(f"Page {page+1}/{total_pages or '?'} | Total stored: {total_stored}")

        # Stop conditions
        if max_objects and total_stored >= max_objects:
            log.info(f"Reached max_objects limit ({max_objects}), stopping.")
            break
        if total_pages and page + 1 >= total_pages:
            log.info("All pages fetched.")
            _set_state(conn, "browse_last_page", -1)   # reset so next run starts fresh
            _set_state(conn, "browse_total_stored", 0)
            break

        page += 1
        time.sleep(RATE_LIMIT)

    _finish_run(conn, run_id, total_stored, total_errors, "complete")
    conn.close()
    log.info(f"Browse complete. {total_stored} objects stored, {total_errors} errors.")

    _finish_run(conn, run_id, total_stored, total_errors, "complete")
    conn.close()
    log.info(f"Browse complete. {total_stored} objects stored, {total_errors} errors.")


# ── Phase 2: Hydrate ───────────────────────────────────────────────────────────
def run_hydrate(batch_size: int = 500):
    """For each un-hydrated NEO, pull full close-approach history."""
    conn = get_db()
    run_id = _start_run(conn, "hydrate")

    # Fetch IDs that need hydration
    rows = conn.execute(
        "SELECT id FROM neo_objects WHERE hydrated = 0 ORDER BY id"
    ).fetchall()
    neo_ids = [r["id"] for r in rows]

    log.info(f"Hydrating {len(neo_ids)} NEOs...")
    total_stored = 0
    total_errors = 0

    for i, neo_id in enumerate(neo_ids):
        data = api_get(f"/neo/{neo_id}")

        if not data:
            total_errors += 1
            log.warning(f"[{i+1}/{len(neo_ids)}] Failed to hydrate {neo_id}")
            continue

        ca_list = data.get("close_approach_data", [])
        ca_count = 0

        for ca in ca_list:
            try:
                ca_record = parse_close_approach(neo_id, ca)
                _insert_ca(conn, ca_record)
                ca_count += 1
            except Exception as e:
                log.error(f"CA parse error for {neo_id}: {e}")

        # Mark as hydrated, update CA count
        conn.execute(
            """UPDATE neo_objects
               SET hydrated = 1,
                   close_approach_count = ?,
                   updated_at = datetime('now')
               WHERE id = ?""",
            (ca_count, neo_id)
        )

        total_stored += ca_count

        if (i + 1) % 50 == 0:
            conn.commit()
            log.info(f"[{i+1}/{len(neo_ids)}] Hydrated | CA records this batch: {total_stored}")

        time.sleep(RATE_LIMIT)

    conn.commit()
    _finish_run(conn, run_id, total_stored, total_errors, "complete")
    conn.close()
    log.info(f"Hydration complete. {total_stored} CA records stored, {total_errors} errors.")


# ── DB Helpers ─────────────────────────────────────────────────────────────────
def _upsert_neo(conn: sqlite3.Connection, record: dict):
    conn.execute("""
        INSERT INTO neo_objects (
            id, name, absolute_magnitude_h,
            diameter_min_km, diameter_max_km, diameter_mean_km,
            is_potentially_hazardous,
            eccentricity, semi_major_axis, inclination,
            perihelion_distance, aphelion_distance, orbital_period,
            ascending_node_longitude, argument_of_perihelion, mean_anomaly,
            orbit_class, first_observation_date, last_observation_date
        ) VALUES (
            :id, :name, :absolute_magnitude_h,
            :diameter_min_km, :diameter_max_km, :diameter_mean_km,
            :is_potentially_hazardous,
            :eccentricity, :semi_major_axis, :inclination,
            :perihelion_distance, :aphelion_distance, :orbital_period,
            :ascending_node_longitude, :argument_of_perihelion, :mean_anomaly,
            :orbit_class, :first_observation_date, :last_observation_date
        )
        ON CONFLICT(id) DO UPDATE SET
            updated_at = datetime('now')
    """, record)


def _insert_ca(conn: sqlite3.Connection, record: dict):
    conn.execute("""
        INSERT OR IGNORE INTO close_approaches (
            neo_id, epoch_date, epoch_date_unix,
            relative_velocity_km_s, miss_distance_km,
            miss_distance_au, miss_distance_lunar, orbiting_body
        ) VALUES (
            :neo_id, :epoch_date, :epoch_date_unix,
            :relative_velocity_km_s, :miss_distance_km,
            :miss_distance_au, :miss_distance_lunar, :orbiting_body
        )
    """, record)


def _start_run(conn: sqlite3.Connection, run_type: str) -> int:
    cursor = conn.execute(
        "INSERT INTO scraper_runs (run_type, status) VALUES (?, 'running')",
        (run_type,)
    )
    conn.commit()
    return cursor.lastrowid


def _finish_run(conn, run_id, stored, errors, status):
    conn.execute("""
        UPDATE scraper_runs
        SET objects_stored = ?, errors = ?, status = ?, finished_at = datetime('now')
        WHERE id = ?
    """, (stored, errors, status, run_id))
    conn.commit()


def _get_state(conn: sqlite3.Connection, key: str, default=None):
    """Read a persistent scraper state value."""
    row = conn.execute("SELECT value FROM scraper_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def _set_state(conn: sqlite3.Connection, key: str, value):
    """Write a persistent scraper state value."""
    conn.execute("""
        INSERT INTO scraper_state (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (key, str(value)))
    conn.commit()


# ── Stats ──────────────────────────────────────────────────────────────────────
def print_stats():
    conn = get_db()
    neo_count  = conn.execute("SELECT COUNT(*) FROM neo_objects").fetchone()[0]
    hydrated   = conn.execute("SELECT COUNT(*) FROM neo_objects WHERE hydrated=1").fetchone()[0]
    ca_count   = conn.execute("SELECT COUNT(*) FROM close_approaches").fetchone()[0]
    haz_count  = conn.execute("SELECT COUNT(*) FROM neo_objects WHERE is_potentially_hazardous=1").fetchone()[0]
    conn.close()

    print(f"""
╔══════════════════════════════════════╗
║     Monarch NEO Database Stats       ║
╠══════════════════════════════════════╣
║  Total NEOs         : {neo_count:<6}          ║
║  Hydrated           : {hydrated:<6}          ║
║  Close Approaches   : {ca_count:<6}          ║
║  NASA Hazardous     : {haz_count:<6}          ║
╚══════════════════════════════════════╝
""")


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monarch NEO Scraper")
    parser.add_argument("--phase",  choices=["browse", "hydrate", "stats"], required=True)
    parser.add_argument("--max",    type=int, default=None, help="Max objects (test runs)")
    args = parser.parse_args()

    init_db()

    if args.phase == "browse":
        run_browse(max_objects=args.max)
    elif args.phase == "hydrate":
        run_hydrate()
    elif args.phase == "stats":
        print_stats()