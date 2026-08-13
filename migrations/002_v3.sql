-- =============================================================================
-- THE QUANT v3.0 "Prometheus" — Database Migrations
-- =============================================================================
-- Adds tables 11-16 and column additions required by the v3.0 spec:
--   * rl_policies               — Reinforcement Learning policy registry
--   * rl_experience             — RL experience replay buffer (persistence)
--   * microstructure_features   — Market microstructure features
--   * slippage_model_log        — Slippage model calibration log
--   * changepoints              — Bayesian Online Changepoint Detection results
--   * anomaly_scores            — Isolation Forest anomaly scores
-- Plus column additions to trades and strategies.

-- -----------------------------------------------------------------------------
-- 11. rl_policies
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rl_policies (
    id UUID PRIMARY KEY,
    policy_id TEXT NOT NULL UNIQUE,
    asset TEXT NOT NULL,
    regime TEXT,
    algorithm TEXT NOT NULL DEFAULT 'ppo',
    architecture TEXT NOT NULL DEFAULT '32-16-11',
    state_dim INTEGER NOT NULL,
    action_dim INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}',
    training_cycles INTEGER NOT NULL DEFAULT 0,
    total_steps INTEGER NOT NULL DEFAULT 0,
    last_sharpe REAL,
    status TEXT NOT NULL DEFAULT 'staging',   -- training | staging | production | retired
    is_distilled BOOLEAN NOT NULL DEFAULT FALSE,
    is_multi_agent BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_policies_status ON rl_policies (status);
CREATE INDEX IF NOT EXISTS idx_rl_policies_asset ON rl_policies (asset, status);

-- -----------------------------------------------------------------------------
-- 12. rl_experience
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rl_experience (
    id BIGSERIAL PRIMARY KEY,
    policy_id TEXT NOT NULL,
    state JSONB NOT NULL,
    action JSONB NOT NULL,
    reward DOUBLE PRECISION NOT NULL,
    next_state JSONB NOT NULL,
    done BOOLEAN NOT NULL DEFAULT FALSE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset TEXT,
    regime TEXT
);

CREATE INDEX IF NOT EXISTS idx_rl_experience_policy ON rl_experience (policy_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_rl_experience_time ON rl_experience (timestamp DESC);

-- -----------------------------------------------------------------------------
-- 13. microstructure_features
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS microstructure_features (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    spread_bps DOUBLE PRECISION,
    depth_imbalance DOUBLE PRECISION,
    order_flow_imbalance DOUBLE PRECISION,
    trade_intensity DOUBLE PRECISION,
    realised_spread_bps DOUBLE PRECISION,
    price_impact_1s DOUBLE PRECISION,
    price_impact_5s DOUBLE PRECISION,
    bid_ask_volume_ratio DOUBLE PRECISION,
    microprice DOUBLE PRECISION,
    features_hash TEXT
);

SELECT create_hypertable('microstructure_features', 'time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_microstructure_symbol ON microstructure_features (symbol, time DESC);

-- -----------------------------------------------------------------------------
-- 14. slippage_model_log
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS slippage_model_log (
    id BIGSERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    order_type TEXT NOT NULL,
    volume DECIMAL NOT NULL,
    expected_price DECIMAL,
    actual_price DECIMAL,
    slippage_bps DOUBLE PRECISION,
    spread_bps DOUBLE PRECISION,
    pred_slippage_bps DOUBLE PRECISION,
    model_mae DOUBLE PRECISION,
    regime TEXT,
    model_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_slippage_model_time ON slippage_model_log (time DESC);
CREATE INDEX IF NOT EXISTS idx_slippage_model_symbol ON slippage_model_log (symbol, time DESC);

-- -----------------------------------------------------------------------------
-- 15. changepoints
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS changepoints (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    run_length INTEGER NOT NULL,
    change_probability DOUBLE PRECISION NOT NULL,
    regime_before TEXT,
    regime_after TEXT,
    features_hash TEXT
);

SELECT create_hypertable('changepoints', 'time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_changepoints_symbol ON changepoints (symbol, time DESC);

-- -----------------------------------------------------------------------------
-- 16. anomaly_scores
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomaly_scores (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    anomaly_type TEXT,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    threshold DOUBLE PRECISION,
    contributing_features JSONB,
    model_version TEXT
);

SELECT create_hypertable('anomaly_scores', 'time', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_anomaly_scores_symbol ON anomaly_scores (symbol, time DESC);

-- -----------------------------------------------------------------------------
-- Column additions to existing tables
-- -----------------------------------------------------------------------------

-- trades: add microstructure + slippage context
ALTER TABLE trades ADD COLUMN IF NOT EXISTS slippage_bps DOUBLE PRECISION;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS spread_bps DOUBLE PRECISION;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS execution_algorithm TEXT;   -- market | twap | vwap | rl
ALTER TABLE trades ADD COLUMN IF NOT EXISTS rl_policy_id TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS anomaly_score DOUBLE PRECISION;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS changepoint_probability DOUBLE PRECISION;

-- strategies: add RL & risk metadata
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS risk_model TEXT DEFAULT 'fixed_fraction';
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS kelly_fraction DOUBLE PRECISION;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS cvar_95 DOUBLE PRECISION;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS bayesian_optimized BOOLEAN DEFAULT FALSE;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS rl_policy_id TEXT;
ALTER TABLE strategies ADD COLUMN IF NOT EXISTS trading_symbols JSONB;

-- accounts: add template metadata
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS template_id TEXT;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS provider TEXT;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS profit_split_pct DOUBLE PRECISION;
ALTER TABLE accounts ADD COLUMN IF NOT EXISTS scaling_plan JSONB;
