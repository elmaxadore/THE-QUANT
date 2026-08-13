//! # State Module (v4.0 "Hercules")
//!
//! Git-centric state architecture. The GitHub repo (plus encrypted vault) is
//! the SINGLE SOURCE OF TRUTH. The PostgreSQL database is a CACHE.
//!
//! All state is serialized to `state/` in the repo:
//!
//! ```text
//! state/
//! ├── accounts/
//! │   └── <account_uuid>/
//! │       ├── account.toml       # Account config + rules
//! │       ├── lifecycle.toml     # Pipeline state
//! │       ├── consistency.toml   # Consistency tracking
//! │       ├── shield.toml        # Guardian shield state
//! │       ├── payout.toml        # Payout economics
//! │       ├── strategies/        # Active strategy configs
//! │       ├── models/            # Model manifests
//! │       ├── lab/               # Lab candidates
//! │       ├── trades/            # Trade journal
//! │       ├── equity/             # Equity snapshots
//! │       └── evolution/          # Evolution logs
//! ├── global/                     # Shared market data, regimes
//! └── manifest.json               # Master state manifest (checksums)
//! ```

pub mod manifest;
pub mod restore;
pub mod store;

pub use manifest::{AccountStateEntry, GlobalState, StateManifest};
pub use restore::{RestoreEngine, RestoreOptions};
pub use store::StateStore;

use serde::Serialize;
use uuid::Uuid;

/// Serialized account state entry builder (simplified for CLI/onboarding)
#[derive(Debug, Clone, Serialize)]
pub struct AccountStateFile {
    pub uuid: Uuid,
    pub name: String,
    pub firm: String,
    pub variant: String,
    pub stage: String,
    pub lifecycle: String,
}
