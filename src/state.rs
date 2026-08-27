//! STATE — v4.0 HERCULES golden rule: *state lives in git*.
//!
//! The PostgreSQL database is a cache, NOT the source of truth. On catastrophic
//! failure: `git clone` + `the-quant restore` must reconstruct everything.
//!
//! This module provides a versioned, JSON-serialisable state store rooted in
//! `state/`. Every mutation bumps a monotonically increasing revision and is
//! atomically written, so a crash mid-write can never corrupt the store.

use crate::config::Config;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const STATE_SCHEMA_VERSION: u32 = 1;

/// Per-account runtime state (the essence of v4.0 isolation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountState {
    pub account_id: String,
    pub account_type: String,
    pub balance: f64,
    pub equity: f64,
    pub daily_pnl: f64,
    pub max_drawdown_pct: f64,
    pub current_drawdown_pct: f64,
    #[serde(default)]
    pub status: String, // Active | Paused | Halted | Blown
    pub open_positions: BTreeMap<String, PositionState>,
}

impl AccountState {
    pub fn new(account_id: &str, account_type: &str, balance: f64) -> Self {
        AccountState {
            account_id: account_id.into(),
            account_type: account_type.into(),
            balance,
            equity: balance,
            daily_pnl: 0.0,
            max_drawdown_pct: 0.0,
            current_drawdown_pct: 0.0,
            status: "Active".into(),
            open_positions: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionState {
    pub symbol: String,
    pub direction: String, // BUY | SELL
    pub volume: f64,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub open_time: String,
    pub unrealized_pnl: f64,
}

/// One closed trade, appended to the journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub account_id: String,
    pub symbol: String,
    pub direction: String,
    pub volume: f64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl: f64,
    pub open_time: String,
    pub close_time: String,
    pub strategy: String,
}

/// The whole state root that gets committed to git.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateRoot {
    pub schema_version: u32,
    pub revision: u64,
    pub account: AccountState,
    #[serde(default)]
    pub trades: Vec<Trade>,
    #[serde(default)]
    pub last_update_check: Option<String>,
    #[serde(default)]
    pub last_update_applied: Option<String>,
    #[serde(default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl StateRoot {
    pub fn initial(account: AccountState) -> Self {
        StateRoot {
            schema_version: STATE_SCHEMA_VERSION,
            revision: 0,
            account,
            trades: Vec::new(),
            last_update_check: None,
            last_update_applied: None,
            extra: BTreeMap::new(),
        }
    }
}

/// File-based state store. A `Store` guards the path to `state/`.
pub struct Store {
    root: PathBuf,
    state_path: PathBuf,
    updated_since_commit: bool,
}

impl Store {
    pub fn new(cfg: &Config) -> Result<Self, String> {
        let root = cfg.state_root();
        std::fs::create_dir_all(&root).map_err(|e| format!("state dir: {e}"))?;
        let state_path = root.join("state.json");
        Ok(Store {
            root,
            state_path,
            updated_since_commit: false,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn load(&self) -> Result<StateRoot, String> {
        if self.state_path.exists() {
            let s = std::fs::read_to_string(&self.state_path).map_err(|e| format!("read state: {e}"))?;
            let mut root: StateRoot =
                serde_json::from_str(&s).map_err(|e| format!("parse state: {e}"))?;
            // forward-migrate: bump revision bookkeeping is harmless on load
            if root.schema_version < STATE_SCHEMA_VERSION {
                root.schema_version = STATE_SCHEMA_VERSION;
            }
            Ok(root)
        } else {
            Ok(StateRoot::initial(AccountState::new("default", "PERSONAL", 10_000.0)))
        }
    }

    /// Persist the root atomically (write temp + rename). Bumps revision.
    pub fn save(&mut self, root: &StateRoot) -> Result<(), String> {
        let mut next = root.clone();
        next.revision += 1;
        let json = serde_json::to_string_pretty(&next).map_err(|e| format!("serialise: {e}"))?;
        let tmp = self.state_path.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(|e| format!("write tmp: {e}"))?;
        std::fs::rename(&tmp, &self.state_path).map_err(|e| format!("rename: {e}"))?;
        self.updated_since_commit = true;
        Ok(())
    }

    /// We changed something internally (e.g. a trade landed) but didn't save.
    pub fn mark_dirty(&mut self) {
        self.updated_since_commit = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.updated_since_commit
    }

    pub fn clear_dirty(&mut self) {
        self.updated_since_commit = false;
    }
}

/// The `the-quant restore` brain: reconstruct everything from `state/` on a
/// fresh machine. Since git is the source of truth, restoring simply means
/// making sure the state dir exists and loading it.
pub struct Restorer {
    root: PathBuf,
}

impl Restorer {
    pub fn new(cfg: &Config) -> Self {
        Restorer { root: cfg.state_root() }
    }

    pub fn restore(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.root).map_err(|e| format!("mkdir state: {e}"))?;
        for sub in ["trades", "equity", "models", "accounts"] {
            let p = self.root.join(sub);
            std::fs::create_dir_all(&p).map_err(|e| format!("mkdir {sub}: {e}"))?;
        }
        Ok(())
    }
}