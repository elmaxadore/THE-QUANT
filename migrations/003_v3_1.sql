-- =============================================================================
-- THE QUANT v3.1 "Hephaestus" — Database Migrations
-- =============================================================================
-- Adds tables 17-22 and column additions required by the v3.1 spec:
--   * consistency_states      — Prop-firm consistency tracking state
--   * shield_states           — Guardian Shield per-trade loss circuit breaker state
--   * payout_tracking         — Lifetime payout caps & extraction progress
--   * pipeline_accounts       — Account lifecycle & rotation pipeline state
--   * extraction_curves       — Target vs actual equity extraction trajectories
--   * shield_strikes_log      — Log of per-trade loss strikes & gap exceptions
-- Plus column additions to accounts and trades.

-- -----------------------------------------------------------------------------
-- 17. consistency_states
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS consistency_states (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    profit_concentration_max DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    lot_size_cv DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    trade_frequency_7d INTEGER NOT NULL DEFAULT 0,
    avg_hold_time_minutes DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    consistency_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    compliance_status TEXT NOT NULL DEFAULT 'Compliant',
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_consistency_account ON consistency_states (account_id, computed_at DESC);

-- -----------------------------------------------------------------------------
-- 18. shield_states
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shield_states (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    strikes INTEGER NOT NULL DEFAULT 0,
    last_strike_time TIMESTAMPTZ,
    shield_status TEXT NOT NULL DEFAULT 'Active',
    trade_count_since_last_strike INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shield_account ON shield_states (account_id, updated_at DESC);

-- -----------------------------------------------------------------------------
-- 19. payout_tracking
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payout_tracking (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    total_cap DECIMAL NOT NULL,
    already_paid_out DECIMAL NOT NULL DEFAULT 0,
    remaining_cap DECIMAL NOT NULL,
    avg_daily_profit DECIMAL NOT NULL DEFAULT 0,
    extraction_velocity DECIMAL NOT NULL DEFAULT 0,
    payout_progress_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    projected_payout_date TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payout_account ON payout_tracking (account_id, updated_at DESC);

-- -----------------------------------------------------------------------------
-- 20. pipeline_accounts
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_accounts (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    lifecycle_state TEXT NOT NULL DEFAULT 'Acquired',
    pipeline_position INTEGER NOT NULL DEFAULT 0,
    readiness_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    activated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    rotation_reason TEXT,
    next_account_id UUID
);

CREATE INDEX IF NOT EXISTS idx_pipeline_lifecycle ON pipeline_accounts (lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_pipeline_account ON pipeline_accounts (account_id);

-- -----------------------------------------------------------------------------
-- 21. extraction_curves
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS extraction_curves (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    day INTEGER NOT NULL,
    target_equity DECIMAL NOT NULL,
    actual_equity DECIMAL NOT NULL,
    deviation_pct DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_extraction_account_day ON extraction_curves (account_id, day);

-- -----------------------------------------------------------------------------
-- 22. shield_strikes_log
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS shield_strikes_log (
    id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    trade_id UUID REFERENCES trades(id),
    expected_max_loss DECIMAL NOT NULL,
    actual_loss DECIMAL NOT NULL,
    strike_number INTEGER NOT NULL,
    reason TEXT,
    gap_exception BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shield_strikes_account ON shield_strikes_log (account_id, created_at DESC);

-- -----------------------------------------------------------------------------
-- Column additions for v3.1
-- -----------------------------------------------------------------------------

-- accounts table extensions
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS lifecycle_state TEXT DEFAULT 'Acquired';
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS pipeline_position INTEGER DEFAULT 0;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS scaling_level INTEGER DEFAULT 0;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS current_payout_cap DECIMAL;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS guardian_shield_enabled BOOLEAN DEFAULT FALSE;

-- trades table extensions
ALTER TABLE trades ADD COLUMN IF NOT EXISTS consistency_impact_score DOUBLE PRECISION;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS shield_check_passed BOOLEAN DEFAULT TRUE;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS projected_worst_loss DECIMAL;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS extraction_curve_day INTEGER;
