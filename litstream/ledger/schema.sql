-- LitStream cost ledger — paperclip's cost model, scaled to a single user.
-- One SQLite file (litstream.db). All money stored as INTEGER cents.
-- All timestamps stored as ISO-8601 UTC strings.

PRAGMA journal_mode = WAL;        -- safe concurrent reads while a run writes
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- runs: one row per pipeline invocation (scheduled or manual).
-- Mirrors paperclip.heartbeat_runs + continuo's per-session usage roll-up.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,         -- e.g. "weekly-citeseq-20260610T060000Z"
    routine       TEXT,                     -- routine name, or NULL for ad-hoc/manual
    project       TEXT NOT NULL,            -- projects/<project>
    invocation    TEXT NOT NULL             -- 'scheduled' | 'manual' | 'backfill'
                    CHECK (invocation IN ('scheduled','manual','backfill')),
    status        TEXT NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running','completed','failed',
                                      'aborted_budget','coalesced')),
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    -- denormalized roll-ups, recomputed from cost_events on each insert
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    cost_cents          REAL    NOT NULL DEFAULT 0,   -- fractional: many task calls are sub-cent
    papers_new    INTEGER NOT NULL DEFAULT 0,   -- new papers Acquire fed this run
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_routine_started ON runs(routine, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_started         ON runs(started_at);

-- ---------------------------------------------------------------------------
-- cost_events: one row per agent turn (every ResultMessage from the headless
-- driver). This is the source of truth; everything else is a rollup over it.
-- Mirrors paperclip.cost_events.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cost_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    phase         TEXT,                     -- 'mine'|'synthesize'|'evaluate'|'design'|'adjudicate'|...
    role          TEXT,                     -- 'v1'|'v2'|'adjudicator'|'orchestrator'|NULL
    model         TEXT NOT NULL,            -- e.g. 'claude-fable-5'
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    -- cost_cents: prefer the SDK's reported total_cost_usd (→ cents) when present;
    -- otherwise computed from tokens × model_pricing. cost_source records which.
    -- REAL (fractional cents) so sub-cent triage/embedding calls aren't rounded up.
    cost_cents    REAL NOT NULL DEFAULT 0,
    cost_source   TEXT NOT NULL DEFAULT 'computed'
                    CHECK (cost_source IN ('sdk_reported','computed')),
    occurred_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cost_run            ON cost_events(run_id);
CREATE INDEX IF NOT EXISTS idx_cost_occurred       ON cost_events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_cost_model_occurred ON cost_events(model, occurred_at);

-- ---------------------------------------------------------------------------
-- budget_policies: trimmed to the single active policy. Seeded from
-- config/sources.yaml. window is a UTC calendar month (paperclip semantics).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS budget_policies (
    id            INTEGER PRIMARY KEY CHECK (id = 1),   -- singleton row
    cap_cents     INTEGER NOT NULL,                     -- 5000 = $50.00
    warn_pct      INTEGER NOT NULL DEFAULT 80,
    hard_stop     INTEGER NOT NULL DEFAULT 1,           -- bool
    window        TEXT NOT NULL DEFAULT 'calendar_month_utc',
    updated_at    TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- budget_incidents: opened when month-to-date crosses warn or cap.
-- Mirrors paperclip.budget_incidents.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS budget_incidents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    threshold     TEXT NOT NULL CHECK (threshold IN ('warning','hard_stop')),
    window_start  TEXT NOT NULL,            -- UTC month start; (threshold,window_start) is one incident
    cap_cents     INTEGER NOT NULL,
    observed_cents REAL NOT NULL,           -- month-to-date spend when tripped
    status        TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open','acknowledged','resolved')),
    opened_at     TEXT NOT NULL,
    resolved_at   TEXT,
    UNIQUE (threshold, window_start)        -- one warning + one hard_stop per month
);
CREATE INDEX IF NOT EXISTS idx_incident_status ON budget_incidents(status);

-- ---------------------------------------------------------------------------
-- model_pricing: fallback when the SDK does not report total_cost_usd.
-- Prices in cents per 1M tokens. Seed/refresh from the claude-api reference.
-- SDK-reported cost always wins over these.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_pricing (
    model              TEXT PRIMARY KEY,
    input_cents_per_mtok        REAL NOT NULL,
    output_cents_per_mtok       REAL NOT NULL,
    cached_input_cents_per_mtok REAL NOT NULL,   -- cache-read rate
    updated_at         TEXT NOT NULL
);
