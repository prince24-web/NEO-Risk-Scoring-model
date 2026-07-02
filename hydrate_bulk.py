"""
Monarch Industries | NEO Risk Scorer
=====================================
Step 1b: Bulk Hydration via JPL Close Approach Data (CAD) API

WHY THIS EXISTS:
  The NeoWs /neo/{id} approach requires ~62,000 individual HTTP calls (~2 days).
  JPL's CAD API returns ALL close approaches in bulk — same data, minutes not days.

ENDPOINT:
  https://ssd-api.jpl.nasa.gov/cad.api
  No API key required. Returns paginated JSON with all CA records.

PARAMETERS USED:
  dist-max  = 0.2 AU   (capture anything that comes within ~30M km)
  date-min  = 1900-01-01
  date-max  = 2200-01-01
  fullname  = true      (include asteroid name for matching to neo_objects)
  diameter  = true      (include diameter data where available)

Usage:
  python hydrate_bulk.py            # full bulk hydration
  python hydrate_bulk.py --stats    # show what's in DB after run
"""

import sqlite3
import logging
import time
import argparse
import datetime
import requests
from pathlib import Path

DB_PATH  = Path(__file__).parent / "neo_data.db"
CAD_URL  = "https://ssd-api.jpl.nasa.gov/cad.api"
RATE_LIMIT = 2.0   # seconds between paginated CAD requests (be polite to JPL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("hydrate_bulk")


# ── DB ─────────────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Fetch CAD data ─────────────────────────────────────────────────────────────
def fetch_cad_page(params: dict) -> dict | None:
    """Fetch one page of CAD results with retry logic and increased timeout."""
    for attempt in range(1, 4):
        try:
            # Increased timeout to 90 seconds to give NASA more headroom
            resp = requests.get(CAD_URL, params=params, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 30 * attempt
                log.warning(f"Rate limited. Waiting {wait}s... (attempt {attempt}/3)")
                time.sleep(wait)
            else:
                log.error(f"CAD API error {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.RequestException as e:
            log.error(f"Request failed: {e}")
            time.sleep(10)
    return None


# ── Parse CAD response ─────────────────────────────────────────────────────────
def parse_cad_records(data: dict) -> list[dict]:
    fields = data.get("fields", [])
    rows   = data.get("data",   [])
    records = []

    for row in rows:
        record = dict(zip(fields, row))
        records.append(record)

    return records


def map_to_ca_record(record: dict, neo_id_map: dict) -> dict | None:
    des      = (record.get("des") or "").strip()
    fullname = (record.get("fullname") or "").strip()

    neo_id = neo_id_map.get(des) or neo_id_map.get(fullname)

    if not neo_id:
        neo_id = neo_id_map.get(des.lstrip("0"))

    if not neo_id:
        return None   

    try:
        dist_au  = float(record["dist"])   if record.get("dist")  else None
        v_rel    = float(record["v_rel"])  if record.get("v_rel") else None
        cd       = record.get("cd", "")    
    except (ValueError, TypeError):
        return None

    if not dist_au or not v_rel:
        return None

    dist_km    = dist_au * 149_597_870.7
    dist_lunar = dist_au * 389.17   

    return {
        "neo_id":                  neo_id,
        "epoch_date":              cd,
        "epoch_date_unix":         None,         
        "relative_velocity_km_s":  v_rel,
        "miss_distance_km":        round(dist_km, 2),
        "miss_distance_au":        dist_au,
        "miss_distance_lunar":     round(dist_lunar, 4),
        "orbiting_body":           "Earth",
    }


# ── Build lookup map ───────────────────────────────────────────────────────────
def build_neo_id_map(conn) -> dict:
    rows = conn.execute("SELECT id, name FROM neo_objects").fetchall()
    lookup = {}
    for row in rows:
        neo_id = row["id"]
        name   = (row["name"] or "").strip()
        lookup[neo_id] = neo_id
        lookup[name] = neo_id
        parts = name.split()
        if parts and parts[0].isdigit():
            lookup[parts[0]] = neo_id

    log.info(f"Built NEO lookup map with {len(lookup)} entries")
    return lookup


# ── Insert CA records ──────────────────────────────────────────────────────────
def insert_ca_batch(conn, ca_records: list[dict]) -> int:
    inserted = 0
    for ca in ca_records:
        try:
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
            """, ca)
            inserted += 1
        except sqlite3.Error as e:
            log.error(f"Insert error: {e} | record: {ca}")
    return inserted


def update_ca_counts(conn):
    conn.execute("""
        UPDATE neo_objects
        SET close_approach_count = (
            SELECT COUNT(*) FROM close_approaches
            WHERE close_approaches.neo_id = neo_objects.id
        ),
        hydrated = 1
        WHERE id IN (SELECT DISTINCT neo_id FROM close_approaches)
    """)
    conn.commit()
    log.info("Updated close_approach_count on neo_objects")


# ── Helper for Date Chunking ───────────────────────────────────────────────────
def date_chunks(start_year: int, end_year: int, step_years: int):
    """Generates lists of (start_date, end_date) in year blocks."""
    current_year = start_year
    while current_year < end_year:
        next_year = min(current_year + step_years, end_year)
        yield f"{current_year}-01-01", f"{next_year}-01-01"
        current_year = next_year


# ── Main ───────────────────────────────────────────────────────────────────────
def run_bulk_hydration():
    conn     = get_db()
    neo_map  = build_neo_id_map(conn)

    # Base params without static dates
    base_params = {
        "dist-max":  "0.2",
        "fullname":  "true",
        "sort":      "date",
        "limit":     "10000",     
    }

    total_ca    = 0
    total_match = 0
    page_size   = 10000

    # Split the massive 1900-2200 window into lightweight 10-year blocks
    for start_date, end_date in date_chunks(1900, 2200, step_years=10):
        log.info(f"--- Processing Time Block: {start_date} to {end_date} ---")
        
        offset = 0
        block_records = None
        
        while block_records is None or offset < block_records:
            params = {
                **base_params, 
                "date-min": start_date, 
                "date-max": end_date,
                "limit": str(page_size), 
                "limit-from": str(offset)
            }
            
            log.info(f"Fetching offset {offset} for block {start_date}...")
            data = fetch_cad_page(params)
            
            if not data:
                log.warning(f"Failed at offset {offset}, retrying block segment after 30s...")
                time.sleep(30)
                continue
            
            # Learn the total count inside this specific block
            if block_records is None:
                block_records = int(data.get("count", 0))
                log.info(f"Block has {block_records:,} total entries to pull.")
                if block_records == 0:
                    break

            raw_records = parse_cad_records(data)
            ca_records  = []

            for rec in raw_records:
                ca = map_to_ca_record(rec, neo_map)
                if ca:
                    ca_records.append(ca)

            matched  = insert_ca_batch(conn, ca_records)
            conn.commit()

            total_ca    += len(raw_records)
            total_match += matched

            log.info(f"  Parsed: {len(raw_records)} | Matched to DB: {matched} | Running total matches: {total_match:,}")

            offset += page_size
            time.sleep(RATE_LIMIT)

    # Update CA counts on neo_objects
    update_ca_counts(conn)
    conn.close()

    log.info(f"""
Bulk hydration complete.
  Total CAD records fetched : {total_ca:,}
  Matched to neo_objects    : {total_match:,}
  Unmatched (new/unknown)   : {total_ca - total_match:,}
    """)


# ── Stats ──────────────────────────────────────────────────────────────────────
def print_stats():
    conn = get_db()
    ca_total    = conn.execute("SELECT COUNT(*) FROM close_approaches").fetchone()[0]
    neo_hydrated = conn.execute("SELECT COUNT(*) FROM neo_objects WHERE hydrated=1").fetchone()[0]
    neo_total    = conn.execute("SELECT COUNT(*) FROM neo_objects").fetchone()[0]
    top = conn.execute("""
        SELECT n.name, COUNT(c.id) as ca_count
        FROM neo_objects n
        JOIN close_approaches c ON c.neo_id = n.id
        GROUP BY n.id ORDER BY ca_count DESC LIMIT 5
    """).fetchall()
    conn.close()

    print(f"""
╔══════════════════════════════════════╗
║    Monarch NEO Hydration Stats       ║
╠══════════════════════════════════════╣
║  Total NEOs         : {neo_total:<6}          ║
║  Hydrated           : {neo_hydrated:<6}          ║
║  Close Approaches   : {ca_total:<6}          ║
╠══════════════════════════════════════╣
║  Most-approached NEOs:               ║""")
    for r in top:
        print(f"║    {r[0][:28]:<28} {r[1]:>4} CA  ║")
    print("╚══════════════════════════════════════╝")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monarch NEO Bulk Hydration")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        run_bulk_hydration()