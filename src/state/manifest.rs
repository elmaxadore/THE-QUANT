//! # State Manifest (v4.0 "Hercules")
//!
//! The ROSETTA STONE of the system. Maps every piece of state to its location,
//! version, and checksum. This file lives at `state/manifest.json` in the repo
//! and is the single file that enables full reconstruction on a fresh machine.

use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Current manifest schema version
pub const MANIFEST_SCHEMA_VERSION: u32 = 4;
/// Current system version
pub const SYSTEM_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Per-account entry in the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountStateEntry {
    pub uuid: Uuid,
    pub name: String,
    pub firm: String,
    pub variant: String,
    pub stage: String,
    pub lifecycle: String,
    pub state_hash: String,
    pub last_commit: String,
    #[serde(default)]
    pub paths: HashMap<String, String>,
}

/// Global system state tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GlobalState {
    #[serde(default)]
    pub regime_detector_version: String,
    #[serde(default)]
    pub correlation_matrix_date: String,
    #[serde(default)]
    pub last_evolution: Option<DateTime<Utc>>,
    #[serde(default)]
    pub market_data_through: Option<DateTime<Utc>>,
}

/// Master state manifest — the single source of truth map
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateManifest {
    pub version: String,
    pub last_updated: DateTime<Utc>,
    pub schema_version: u32,
    #[serde(default)]
    pub accounts: Vec<AccountStateEntry>,
    #[serde(default)]
    pub global_state: GlobalState,
    #[serde(default)]
    pub checksums: HashMap<String, String>,
}

impl Default for StateManifest {
    fn default() -> Self {
        Self {
            version: SYSTEM_VERSION.to_string(),
            last_updated: Utc::now(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            accounts: Vec::new(),
            global_state: GlobalState::default(),
            checksums: HashMap::new(),
        }
    }
}

impl StateManifest {
    /// Load a manifest from a file path
    pub fn load(path: &Path) -> QuantResult<Self> {
        if !path.exists() {
            return Err(QuantError::StateError(format!(
                "State manifest not found at {}",
                path.display()
            )));
        }
        let content = std::fs::read_to_string(path)?;
        let manifest: StateManifest = serde_json::from_str(&content)
            .map_err(|e| QuantError::StateError(format!("Failed to parse manifest: {}", e)))?;
        Ok(manifest)
    }

    /// Save the manifest to a file path (pretty-printed)
    pub fn save(&self, path: &Path) -> QuantResult<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| QuantError::StateError(format!("Failed to serialize manifest: {}", e)))?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Compute a blake3 checksum for a file
    pub fn compute_checksum(path: &Path) -> QuantResult<String> {
        let data = std::fs::read(path)?;
        let hash = blake3::hash(&data);
        Ok(format!("blake3:{}", hash.to_hex()))
    }

    /// Generate the canonical `state/` directory layout for an account
    pub fn account_state_paths(state_root: &Path, uuid: &Uuid) -> HashMap<String, String> {
        let base = state_root.join("accounts").join(uuid.to_string());
        let mut paths = HashMap::new();
        paths.insert("config".into(), base.join("account.toml").to_string_lossy().into());
        paths.insert("lifecycle".into(), base.join("lifecycle.toml").to_string_lossy().into());
        paths.insert("consistency".into(), base.join("consistency.toml").to_string_lossy().into());
        paths.insert("shield".into(), base.join("shield.toml").to_string_lossy().into());
        paths.insert("payout".into(), base.join("payout.toml").to_string_lossy().into());
        paths.insert("strategies".into(), base.join("strategies").to_string_lossy().into());
        paths.insert("models".into(), base.join("models").to_string_lossy().into());
        paths.insert("lab".into(), base.join("lab").to_string_lossy().into());
        paths.insert("trades".into(), base.join("trades").to_string_lossy().into());
        paths.insert("equity".into(), base.join("equity").to_string_lossy().into());
        paths.insert("evolution".into(), base.join("evolution").to_string_lossy().into());
        paths
    }

    /// Ensure the full state directory tree exists
    pub fn ensure_state_tree(state_root: &Path) -> QuantResult<()> {
        let dirs = [
            state_root.join("accounts"),
            state_root.join("global").join("regime_history"),
            state_root.join("global").join("correlation"),
            state_root.join("global").join("market_data"),
            state_root.join("global").join("features"),
            state_root.join("global").join("microstructure"),
            state_root.join("global").join("system"),
        ];
        for dir in dirs.iter() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// Add or update an account entry in the manifest
    pub fn upsert_account(&mut self, entry: AccountStateEntry) {
        if let Some(existing) = self.accounts.iter_mut().find(|a| a.uuid == entry.uuid) {
            *existing = entry;
        } else {
            self.accounts.push(entry);
        }
        self.last_updated = Utc::now();
    }

    /// Remove an account entry
    pub fn remove_account(&mut self, uuid: &Uuid) -> bool {
        let before = self.accounts.len();
        self.accounts.retain(|a| a.uuid != *uuid);
        let removed = self.accounts.len() < before;
        if removed {
            self.last_updated = Utc::now();
        }
        removed
    }

    /// Get an account entry by UUID
    pub fn get_account(&self, uuid: &Uuid) -> Option<&AccountStateEntry> {
        self.accounts.iter().find(|a| a.uuid == *uuid)
    }

    /// Verify all checksums in the manifest against disk
    ///
    /// Returns a list of `(path, expected, actual)` mismatches. Empty = all good.
    pub fn verify_checksums(&self, state_root: &Path) -> Vec<(String, String, String)> {
        let mut mismatches = Vec::new();
        for (rel_path, expected) in &self.checksums {
            let full_path = state_root.join(rel_path);
            if full_path.exists() {
                match Self::compute_checksum(&full_path) {
                    Ok(actual) => {
                        if actual != *expected {
                            mismatches.push((rel_path.clone(), expected.clone(), actual));
                        }
                    }
                    Err(_) => mismatches.push((rel_path.clone(), expected.clone(), "unreadable".into())),
                }
            } else {
                mismatches.push((rel_path.clone(), expected.clone(), "missing".into()));
            }
        }
        mismatches
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_manifest_round_trip() {
        let dir = tempdir().unwrap();
        let manifest_path = dir.path().join("manifest.json");

        let mut manifest = StateManifest::default();
        manifest.upsert_account(AccountStateEntry {
            uuid: Uuid::new_v4(),
            name: "BG-Instant-5K-01".into(),
            firm: "blue_guardian".into(),
            variant: "instant_5k".into(),
            stage: "funded".into(),
            lifecycle: "trading".into(),
            state_hash: "blake3:abc123".into(),
            last_commit: "deadbeef".into(),
            paths: HashMap::new(),
        });

        manifest.save(&manifest_path).unwrap();
        let loaded = StateManifest::load(&manifest_path).unwrap();
        assert_eq!(loaded.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(loaded.accounts.len(), 1);
        assert_eq!(loaded.accounts[0].name, "BG-Instant-5K-01");
    }

    #[test]
    fn test_checksum() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "hello world").unwrap();
        let checksum = StateManifest::compute_checksum(&file).unwrap();
        assert!(checksum.starts_with("blake3:"));
        assert_eq!(checksum.len(), "blake3:".len() + 64); // 32-byte blake3 hex
    }

    #[test]
    fn test_ensure_state_tree() {
        let dir = tempdir().unwrap();
        StateManifest::ensure_state_tree(dir.path()).unwrap();
        assert!(dir.path().join("accounts").exists());
        assert!(dir.path().join("global/market_data").exists());
    }
}