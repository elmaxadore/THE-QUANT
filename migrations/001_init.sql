-- The Quant — Database Initialization
-- PostgreSQL + TimescaleDB schema

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Accounts table
CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    stage TEXT NOT NULL,
    rules JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'Active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- OHLCV hypertable
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open DECIMAL NOT NULL,
    high DECIMAL NOT NULL,
    low DECIMAL NOT NULL,
    close DECIMAL NOT NULL,
    volume DECIMAL NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);
SELECT create_hypertable('ohlcv', 'time', chunk_time_interval => INTERVAL '1 day');

-- Ticks hypertable (optional, retention 7 days)
CREATE TABLE IF NOT EXISTS ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    bid DECIMAL NOT NULL,
    ask DECIMAL NOT NULL,
    volume DECIMAL
);
SELECT create_hypertable('ticks', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Features hypertable
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('features', 'time', chunk_time_interval => INTERVAL '1 day');

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY,
    account_id UUID REFERENCES accounts(id),
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    volume DECIMAL NOT NULL,
    entry_price DECIMAL NOT NULL,
    exit_price DECIMAL,
    stop_loss DECIMAL,
    take_profit DECIMAL,
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ,
    pnl DECIMAL,
    commission DECIMAL DEFAULT 0,
    swap DECIMAL DEFAULT 0,
    strategy_id UUID,
    regime TEXT,
    model_versions JSONB,
    context_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Regimes hypertable
CREATE TABLE IF NOT EXISTS regimes (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    regime TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    features_hash TEXT
);
SELECT create_hypertable('regimes', 'time', chunk_time_interval => INTERVAL '1 day');

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'lab',
    backtest_metrics JSONB,
    live_metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

-- Model manifests
CREATE TABLE IF NOT EXISTS model_manifests (
    id UUID PRIMARY KEY,
    model_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    regime TEXT,
    algorithm TEXT NOT NULL,
    file_path TEXT,
    metrics JSONB,
    status TEXT NOT NULL DEFAULT 'staging',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Evolution cycles
CREATE TABLE IF NOT EXISTS evolution_cycles (
    id UUID PRIMARY KEY,
    triggered_by TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    phases JSONB,
    status TEXT NOT NULL DEFAULT 'running'
);

-- System logs hypertable
CREATE TABLE IF NOT EXISTS system_logs (
    time TIMESTAMPTZ NOT NULL,
    level TEXT NOT NULL,
    target TEXT,
    message TEXT NOT NULL,
    fields JSONB
);
SELECT create_hypertable('system_logs', 'time', chunk_time_interval => INTERVAL '1 day');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv (symbol, timeframe, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_account_id ON trades (account_id);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades (open_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON trades (strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategies_status ON strategies (status);
CREATE INDEX IF NOT EXISTS idx_model_manifests_status ON model_manifests (status);
CREATE INDEX IF NOT EXISTS idx_evolution_cycles_status ON evolution_cycles (status);
CREATE INDEX IF NOT EXISTS idx_features_symbol ON features (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_regimes_symbol ON regimes (symbol, time DESC);
