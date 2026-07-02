"""
Monarch Industries | NEO Risk Scorer
=====================================
Step 2: Feature Engineering + Label Construction

Reads from neo_objects + close_approaches, computes:
  - Orbital feature matrix (X) — model inputs
  - Continuous risk score (y) — engineered from physics, NOT NASA's binary label

Key design decisions:
  - Epsilon softening (1e-6) prevents zero-division when miss_distance ≈ 0
  - Log transform before z-score handles hyper-skewed long-tail distribution
  - MOID, velocity, miss_distance withheld from X (they go into y only)

Usage:
  python features.py             # run full feature engineering pipeline
  python features.py --stats     # print distribution stats after running
"""

import sqlite3
import argparse
import logging
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent / "neo_data.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("neo_features")

# ── Constants ──────────────────────────────────────────────────────────────────
EPSILON         = 1e-6   # softening term: prevents miss_distance=0 blowup
CHRONIC_WEIGHT  = 0.3    # weight of mean risk vs peak risk in final score
EARTH_AU        = 1.0    # AU — perihelion below this crosses Earth's orbit

# Orbit class label encoding (Apollo/Aten most dangerous, Amor/other less so)
ORBIT_CLASS_MAP = {
    "Apollo": 3,   # Earth-crossing, semi-major > 1 AU
    "Aten":   3,   # Earth-crossing, semi-major < 1 AU
    "Amor":   2,   # Earth-approaching, doesn't cross
    "IEO":    2,   # Interior Earth objects
    "Atira":  1,
}


# ── DB ─────────────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Step 1: Load raw data ──────────────────────────────────────────────────────
def load_neo_objects(conn) -> list[dict]:
    """Load all NEOs with at least one close approach."""
    rows = conn.execute("""
        SELECT n.*
        FROM neo_objects n
        WHERE n.close_approach_count > 0
          AND n.diameter_mean_km IS NOT NULL
          AND n.eccentricity IS NOT NULL
    """).fetchall()
    log.info(f"Loaded {len(rows)} NEOs with close approach data")
    return [dict(r) for r in rows]


def load_close_approaches(conn, neo_id: str) -> list[dict]:
    """Load all Earth close approaches for a given NEO."""
    rows = conn.execute("""
        SELECT *
        FROM close_approaches
        WHERE neo_id = ?
          AND orbiting_body = 'Earth'
          AND miss_distance_au IS NOT NULL
          AND relative_velocity_km_s IS NOT NULL
          AND miss_distance_au > 0
    """, (neo_id,)).fetchall()
    return [dict(r) for r in rows]


# ── Step 2: Risk Score Construction ───────────────────────────────────────────
def compute_approach_risk(
    diameter_km: float,
    velocity_km_s: float,
    miss_distance_au: float
) -> float:
    """
    Per-approach danger value based on kinetic energy scaled by distance.

    kinetic_energy_proxy = diameter² × velocity²
        (proportional to actual KE: ½mv² where mass ∝ diameter³,
         but we drop the ½ and diameter exponent for simplicity —
         what matters is relative ranking, not absolute energy)

    approach_risk = KE_proxy / (miss_distance_au² + ε)
        ε = 1e-6 prevents denominator collapse when miss_distance ≈ 0
    """
    kinetic_energy_proxy = (diameter_km ** 2) * (velocity_km_s ** 2)
    approach_risk = kinetic_energy_proxy / (miss_distance_au ** 2 + EPSILON)
    return approach_risk, kinetic_energy_proxy


def compute_object_risk_score(ca_risks: list[float]) -> tuple[float, float, float]:
    """
    Aggregate per-approach risks into a single object-level score.

    risk_raw = max(approach_risk) + CHRONIC_WEIGHT × mean(approach_risk)
        Peak term  : captures the single most dangerous event
        Chronic term: rewards objects that repeatedly come close
    """
    if not ca_risks:
        return 0.0, 0.0, 0.0

    peak = max(ca_risks)
    mean = sum(ca_risks) / len(ca_risks)
    raw  = peak + CHRONIC_WEIGHT * mean
    return raw, peak, mean


# ── Step 3: Feature Engineering ───────────────────────────────────────────────
def engineer_features(neo: dict, ca_list: list[dict]) -> dict | None:
    """
    Build the full feature record for one NEO.
    Returns None if insufficient data to compute.
    """
    neo_id = neo["id"]

    # ── Compute per-approach risks ──────────────────────────────────────────
    ca_risks      = []
    ke_proxies    = []
    velocities    = []
    miss_dists_au = []

    for ca in ca_list:
        risk, ke = compute_approach_risk(
            diameter_km      = neo["diameter_mean_km"],
            velocity_km_s    = ca["relative_velocity_km_s"],
            miss_distance_au = ca["miss_distance_au"]
        )
        ca_risks.append(risk)
        ke_proxies.append(ke)
        velocities.append(ca["relative_velocity_km_s"])
        miss_dists_au.append(ca["miss_distance_au"])

    if not ca_risks:
        return None

    risk_raw, risk_peak, risk_mean = compute_object_risk_score(ca_risks)

    # ── Orbital features (X — model inputs) ────────────────────────────────
    ecc  = neo["eccentricity"]   or 0.0
    inc  = neo["inclination"]    or 0.0
    peri = neo["perihelion_distance"] or 0.0

    orbit_class_raw     = neo.get("orbit_class") or "Unknown"
    orbit_class_encoded = ORBIT_CLASS_MAP.get(orbit_class_raw, 0)

    features = {
        "neo_id": neo_id,

        # ── Orbital features (X) ─────────────────────────────────────────
        "absolute_magnitude_h":           neo.get("absolute_magnitude_h"),
        "diameter_mean_km":               neo["diameter_mean_km"],
        "eccentricity":                   ecc,
        "semi_major_axis":                neo.get("semi_major_axis"),
        "inclination":                    inc,
        "perihelion_distance":            peri,
        "aphelion_distance":              neo.get("aphelion_distance"),
        "orbital_period":                 neo.get("orbital_period"),
        "perihelion_crosses_earth_orbit": int(peri < EARTH_AU),
        "ecc_x_inclination":              round(ecc * inc, 6),  # interaction term
        "orbit_class_encoded":            orbit_class_encoded,

        # ── CA aggregates (used to build y, NOT passed to model) ─────────
        "ca_count":               len(ca_list),
        "ca_mean_velocity_km_s":  round(np.mean(velocities),    4) if velocities    else None,
        "ca_max_velocity_km_s":   round(np.max(velocities),     4) if velocities    else None,
        "ca_min_miss_au":         round(np.min(miss_dists_au),  8) if miss_dists_au else None,
        "ca_mean_miss_au":        round(np.mean(miss_dists_au), 8) if miss_dists_au else None,

        # ── Risk score components ────────────────────────────────────────
        "kinetic_energy_proxy":   round(np.mean(ke_proxies), 6) if ke_proxies else None,
        "approach_risk_max":      round(risk_peak, 8),
        "approach_risk_mean":     round(risk_mean, 8),
        "risk_score_raw":         round(risk_raw,  8),

        # Normalized fields written in a second pass after all raws are computed
        "risk_score_normalized":  None,
        "risk_tier":              None,
    }

    return features


# ── Step 4: Normalization (log → z-score) ─────────────────────────────────────
def normalize_risk_scores(records: list[dict]) -> list[dict]:
    """
    Two-stage normalization:

    Stage 1 — Log transform
        log_risk = ln(risk_score_raw + 1)
        Compresses the hyper-skewed long tail. +1 ensures ln(0) = 0, not -inf.
        Without this, z-score is dominated by the few extreme outliers.

    Stage 2 — Z-score
        z = (log_risk - μ) / σ
        Centers the distribution around 0 with unit variance.
        Models learn much better on this than raw values spanning many orders of magnitude.
    """
    raw_scores = np.array([r["risk_score_raw"] for r in records], dtype=np.float64)

    # Stage 1: log transform
    log_scores = np.log1p(raw_scores)   # log1p = ln(x+1), numerically stable at x=0

    # Stage 2: z-score on log-transformed values
    mu    = log_scores.mean()
    sigma = log_scores.std()

    if sigma == 0:
        log.warning("Standard deviation is 0 — all risk scores are identical. Check your data.")
        sigma = 1.0

    z_scores = (log_scores - mu) / sigma

    log.info(f"Risk score stats:")
    log.info(f"  Raw    — min: {raw_scores.min():.4e}, max: {raw_scores.max():.4e}, mean: {raw_scores.mean():.4e}")
    log.info(f"  Log    — min: {log_scores.min():.4f},  max: {log_scores.max():.4f},  mean: {mu:.4f}")
    log.info(f"  Z      — min: {z_scores.min():.4f},   max: {z_scores.max():.4f},   std: {sigma:.4f}")

    for i, record in enumerate(records):
        record["risk_score_normalized"] = round(float(z_scores[i]), 6)

    return records


def assign_risk_tiers(records: list[dict]) -> list[dict]:
    """
    Assign qualitative tier from z-score percentile cuts.
    Percentile-based so tiers are always populated regardless of score distribution.

    CRITICAL : top 2%   (rarest, most dangerous)
    HIGH      : 2–10%
    MEDIUM    : 10–30%
    LOW       : bottom 70%
    """
    z_scores = np.array([r["risk_score_normalized"] for r in records])
    p98 = np.percentile(z_scores, 98)
    p90 = np.percentile(z_scores, 90)
    p70 = np.percentile(z_scores, 70)

    tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for record in records:
        z = record["risk_score_normalized"]
        if   z >= p98: tier = "CRITICAL"
        elif z >= p90: tier = "HIGH"
        elif z >= p70: tier = "MEDIUM"
        else:          tier = "LOW"
        record["risk_tier"] = tier
        tier_counts[tier] += 1

    log.info(f"Tier distribution: {tier_counts}")
    log.info(f"  Thresholds — CRITICAL≥{p98:.3f}, HIGH≥{p90:.3f}, MEDIUM≥{p70:.3f}")
    return records


# ── Step 5: Write to DB ────────────────────────────────────────────────────────
def write_features(conn, records: list[dict]):
    conn.execute("DELETE FROM neo_features")   # fresh write each run
    conn.executemany("""
        INSERT INTO neo_features (
            neo_id,
            absolute_magnitude_h, diameter_mean_km,
            eccentricity, semi_major_axis, inclination,
            perihelion_distance, aphelion_distance, orbital_period,
            perihelion_crosses_earth_orbit, ecc_x_inclination, orbit_class_encoded,
            ca_count, ca_mean_velocity_km_s, ca_max_velocity_km_s,
            ca_min_miss_au, ca_mean_miss_au,
            kinetic_energy_proxy, approach_risk_max, approach_risk_mean,
            risk_score_raw, risk_score_normalized, risk_tier
        ) VALUES (
            :neo_id,
            :absolute_magnitude_h, :diameter_mean_km,
            :eccentricity, :semi_major_axis, :inclination,
            :perihelion_distance, :aphelion_distance, :orbital_period,
            :perihelion_crosses_earth_orbit, :ecc_x_inclination, :orbit_class_encoded,
            :ca_count, :ca_mean_velocity_km_s, :ca_max_velocity_km_s,
            :ca_min_miss_au, :ca_mean_miss_au,
            :kinetic_energy_proxy, :approach_risk_max, :approach_risk_mean,
            :risk_score_raw, :risk_score_normalized, :risk_tier
        )
    """, records)
    conn.commit()
    log.info(f"Wrote {len(records)} feature records to neo_features")


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_feature_engineering():
    conn = get_db()

    # 1. Load
    neos = load_neo_objects(conn)
    if not neos:
        log.error("No NEOs found in DB. Run scraper first.")
        return

    # 2. Engineer features per NEO
    records = []
    skipped = 0

    for i, neo in enumerate(neos):
        ca_list  = load_close_approaches(conn, neo["id"])
        features = engineer_features(neo, ca_list)

        if features is None:
            skipped += 1
            continue

        records.append(features)

        if (i + 1) % 1000 == 0:
            log.info(f"Processed {i+1}/{len(neos)} NEOs...")

    log.info(f"Feature engineering done. {len(records)} records, {skipped} skipped (no CA data).")

    # 3. Normalize risk scores
    records = normalize_risk_scores(records)

    # 4. Assign tiers
    records = assign_risk_tiers(records)

    # 5. Write to DB
    write_features(conn, records)
    conn.close()

    log.info("Step 2 complete. Run with --stats to inspect distribution.")


# ── Stats / Sanity Check ───────────────────────────────────────────────────────
def print_stats():
    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM neo_features").fetchone()[0]
    if total == 0:
        print("No features computed yet. Run: python features.py")
        return

    tiers = conn.execute("""
        SELECT risk_tier, COUNT(*) as cnt
        FROM neo_features
        GROUP BY risk_tier
        ORDER BY cnt ASC
    """).fetchall()

    top10 = conn.execute("""
        SELECT n.name, f.diameter_mean_km, f.eccentricity,
               f.ca_min_miss_au, f.risk_score_normalized, f.risk_tier
        FROM neo_features f
        JOIN neo_objects n ON f.neo_id = n.id
        ORDER BY f.risk_score_normalized DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    print(f"""
╔══════════════════════════════════════════════════════╗
║         Monarch NEO Risk Score — Distribution        ║
╠══════════════════════════════════════════════════════╣
║  Total NEOs with features : {total:<6}                  ║
╠══════════════════════════════════════════════════════╣""")
    for row in tiers:
        print(f"║  {row[0]:<10} : {row[1]:<6}                              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  TOP 10 HIGHEST RISK NEOs                            ║")
    print("╠══════════════════════════════════════════════════════╣")
    for i, r in enumerate(top10, 1):
        name = (r[0] or "Unknown")[:28]
        print(f"║  {i:>2}. {name:<28} [{r[5]}]  ║")
        print(f"║      diameter={r[1]:.4f}km  ecc={r[2]:.3f}  z={r[4]:.3f}       ║")
    print("╚══════════════════════════════════════════════════════╝")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monarch NEO Feature Engineering")
    parser.add_argument("--stats", action="store_true", help="Print distribution stats")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        run_feature_engineering()
