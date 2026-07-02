-- ─────────────────────────────────────────────
--  Monarch Industries | NEO Risk Scorer
--  SQLite Schema
-- ─────────────────────────────────────────────

-- Raw NEO catalog (one row per asteroid)
CREATE TABLE IF NOT EXISTS neo_objects (
    id                          TEXT PRIMARY KEY,   -- NASA NEO reference ID
    name                        TEXT NOT NULL,
    absolute_magnitude_h        REAL,               -- H magnitude (size proxy)
    diameter_min_km             REAL,
    diameter_max_km             REAL,
    diameter_mean_km            REAL,               -- computed: (min+max)/2
    is_potentially_hazardous    INTEGER,            -- NASA's binary label (stored but NOT used as target)
    eccentricity                REAL,
    semi_major_axis             REAL,               -- AU
    inclination                 REAL,               -- degrees
    perihelion_distance         REAL,               -- AU
    aphelion_distance           REAL,               -- AU
    orbital_period              REAL,               -- days
    ascending_node_longitude    REAL,               -- degrees
    argument_of_perihelion      REAL,               -- degrees
    mean_anomaly                REAL,
    orbit_class                 TEXT,               -- e.g. Apollo, Aten, Amor
    first_observation_date      TEXT,
    last_observation_date       TEXT,
    close_approach_count        INTEGER DEFAULT 0,  -- how many CA records fetched
    hydrated                    INTEGER DEFAULT 0,  -- 0 = browse only, 1 = full CA data pulled
    created_at                  TEXT DEFAULT (datetime('now')),
    updated_at                  TEXT DEFAULT (datetime('now'))
);

-- Close approach events (many per NEO)
CREATE TABLE IF NOT EXISTS close_approaches (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    neo_id                      TEXT NOT NULL REFERENCES neo_objects(id),
    epoch_date                  TEXT,               -- ISO date string
    epoch_date_unix             INTEGER,            -- unix ms for sorting
    relative_velocity_km_s      REAL,
    miss_distance_km            REAL,
    miss_distance_au            REAL,
    miss_distance_lunar         REAL,               -- lunar distances
    orbiting_body               TEXT DEFAULT 'Earth',
    created_at                  TEXT DEFAULT (datetime('now')),
    UNIQUE(neo_id, epoch_date, orbiting_body)       -- dedup guard
);

-- Engineered features + computed risk score (written after feature engineering step)
CREATE TABLE IF NOT EXISTS neo_features (
    neo_id                          TEXT PRIMARY KEY REFERENCES neo_objects(id),

    -- Orbital features (X — model inputs)
    absolute_magnitude_h            REAL,
    diameter_mean_km                REAL,
    eccentricity                    REAL,
    semi_major_axis                 REAL,
    inclination                     REAL,
    perihelion_distance             REAL,
    aphelion_distance               REAL,
    orbital_period                  REAL,
    perihelion_crosses_earth_orbit  INTEGER,        -- 1 if perihelion < 1.0 AU
    ecc_x_inclination               REAL,           -- interaction term
    orbit_class_encoded             INTEGER,        -- label-encoded orbit class

    -- Close approach aggregates (used to build y, NOT model inputs)
    ca_count                        INTEGER,
    ca_mean_velocity_km_s           REAL,
    ca_min_miss_au                  REAL,           -- MOID proxy
    ca_max_velocity_km_s            REAL,
    ca_mean_miss_au                 REAL,

    -- Engineered risk score components
    kinetic_energy_proxy            REAL,           -- diameter² × velocity²
    approach_risk_max               REAL,           -- max(KE / miss_distance_au²)
    approach_risk_mean              REAL,           -- mean(KE / miss_distance_au²)

    -- Final target variable
    risk_score_raw                  REAL,           -- approach_risk_max + 0.3 * approach_risk_mean
    risk_score_normalized           REAL,           -- z-scored across dataset
    risk_tier                       TEXT,           -- CRITICAL / HIGH / MEDIUM / LOW

    created_at                      TEXT DEFAULT (datetime('now'))
);

-- Scraper run log (track pages fetched, errors)
CREATE TABLE IF NOT EXISTS scraper_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type        TEXT,       -- 'browse' | 'hydrate'
    pages_fetched   INTEGER DEFAULT 0,
    objects_stored  INTEGER DEFAULT 0,
    errors          INTEGER DEFAULT 0,
    status          TEXT,       -- 'running' | 'complete' | 'failed'
    started_at      TEXT DEFAULT (datetime('now')),
    finished_at     TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ca_neo_id       ON close_approaches(neo_id);
CREATE INDEX IF NOT EXISTS idx_ca_epoch        ON close_approaches(epoch_date_unix);
CREATE INDEX IF NOT EXISTS idx_neo_hydrated    ON neo_objects(hydrated);
CREATE INDEX IF NOT EXISTS idx_features_score  ON neo_features(risk_score_normalized DESC);
