. a //! # State Store (v4.0 "Hercules")
//!
//! Serializes all per-account and global state to the `state/` directory in the
//! repository. The repo IS the source of truth. The DB is a cache.

use crate::error::{QuantError, QuantResult};
use crate::state::manifest::StateManifest;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Versioned state file header — included in every state file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateFileHeader {
    pub schema_version: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl StateFileHeader {
    pub fn new(schema_version: u32) -> Self {
        let now = Utc::now();
        Self {
            schema_version,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }
}

/// Per-account persisted state (mirrors `state/accounts/<uuid>/`)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccountStateFiles {
    #[serde(default)]
    pub account_toml: Option<String>,
    #[serde(default)]
    pub lifecycle_toml: Option<String>,
    #[serde(default)]
    pub consistency_toml: Option<String>,
    #[serde(default)]
    pub shield_toml: Option<String>,
    #[serde(default)]
    pub payout_toml: Option<String>,
}

/// The StateStore handles reading/writing all state files under `state/`
#[derive(Debug, Clone)]
pub struct StateStore {
    /// Root path of the repo (contains `state/`)
    repo_root: PathBuf,
    /// Path to `state/` directory
    state_root: PathBuf,
}

impl StateStore {
    /// Create a new store rooted at the repository directory
    pub fn new(repo_root: PathBuf) -> Self {
        let state_root = repo_root.join("state");
        Self { repo_root, state_root }
    }

    /// Get the state root path
    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    /// Get the repo root path
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// Initialize the state directory tree and create an empty manifest if needed
    pub fn initialize(&self) -> QuantResult<()> {
        StateManifest::ensure_state_tree(&self.state_root)?;
        let manifest_path = self.state_root.join("manifest.json");
        if !manifest_path.exists() {
            let manifest = StateManifest::default();
            manifest.save(&manifest_path)?;
        }
        Ok(())
    }

    /// Load the state manifest
    pub fn load_manifest(&self) -> QuantResult<StateManifest> {
        StateManifest::load(&self.state_root.join("manifest.json"))
    }

    /// Save the state manifest
    pub fn save_manifest(&self, manifest: &StateManifest) -> QuantResult<()> {
        manifest.save(&self.state_root.join("manifest.json"))
    }

    /// Path to an account's state directory
    pub fn account_dir(&self, uuid: &Uuid) -> PathBuf {
        self.state_root.join("accounts").join(uuid.to_string())
    }

    /// Ensure an account's state directory tree exists
    pub fn ensure_account_dirs(&self, uuid: &Uuid) -> QuantResult<()> {
        let base = self.account_dir(uuid);
        let dirs = [
            base.clone(),
            base.join("strategies"),
            base.join("models"),
            base.join("lab"),
            base.join("trades"),
            base.join("equity"),
            base.join("evolution"),
        ];
        for dir in dirs.iter() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// Write a TOML-serializable value to a path under the state tree
    pub fn write_toml<T: Serialize>(&self, rel_path: &Path, value: &T) -> QuantResult<PathBuf> {
        let full = self.state_root.join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(value)
            .map_err(|e| QuantError::StateError(format!("TOML serialize failed: {}", e)))?;
        std::fs::write(&full, content)?;
        Ok(full)
    }

    /// Write a JSON-serializable file to a path under the state tree
    pub fn write_json<T: Serialize>(&self, rel_path: &Path, value: &T) -> QuantResult<PathBuf> {
        let full = self.state_root.join(rel_path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(value)
            .map_err(|e| QuantError::StateError(format!("JSON serialize failed: {}", e)))?;
        std::fs::write(&full, content)?;
        Ok(full)
    }

    /// Read and deserialize a TOML file from the state tree
    pub fn read_toml<T: for<'de> Deserialize<'de>>(&self, rel_path: &Path) -> QuantResult<T> {
        let full = self.state_root.join(rel_path);
        if !full.exists() {
            return Err(QuantError::StateError(format!(
                "State file not found: {}",
                full.display()
            )));
        }
        let content = std::fs::read_to_string(&full)?;
        let value: T = toml::from_str(&content)
            .map_err(|e| QuantError::StateError(format!("TOML parse failed: {}", e)))?;
        Ok(value)
    }

    /// Read and deserialize a JSON file from the state tree
    pub fn read_json<T: for<'de> Deserialize<'de>>(&self, rel_path: &Path) -> QuantResult<T> {
        let full = self.state_root.join(rel_path);
        if !full.exists() {
            return Err(QuantError::StateError(format!(
                "State file not found: {}",
                full.display()
            )));
        }
        let content = std::fs::read_to_string(&full)?;
        let value: T = serde_json::from_str(&content)
            .map_err(|e| QuantError::StateError(format!("JSON parse failed: {}", e)))?;
        Ok(value)
    }

    /// Record a checksum entry for a state file in the manifest
    pub fn record_checksum(&self, rel_path: &Path, manifest: &mut StateManifest) -> QuantResult<()> {
        let full = self.state_root.join(rel_path);
        if full.exists() {
            let checksum = StateManifest::compute_checksum(&full)?;
            manifest
                .checksums
                .insert(rel_path.to_string_lossy().into(), checksum);
        }
        Ok(())
    }

    /// Record all checksums for an account's state files
    pub fn record_account_checksums(
        &self,
        uuid: &Uuid,
        manifest: &mut StateManifest,
    ) -> QuantResult<()> {
        let base = self.account_dir(uuid);
        let rel_base = Path::new("accounts").join(uuid.to_string());

        // Walk the account directory and checksum every file
        for entry in walkdir::WalkDir::new(&base)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let abs_path = entry.path();
            let rel_path = pathdiff::diff_paths(abs_path, &self.state_root)
                .unwrap_or_else(|| rel_base.clone())
                .to_string_lossy()
                .to_string();
            let checksum = StateManifest::compute_checksum(abs_path)?;
            manifest.checksums.insert(rel_path, checksum);
        }

        // Also checksum the account's main config file if it exists
        let account_file = self.state_root.join(&rel_base).join("account.toml");
        if account_file.exists() {
            let rel = rel_base.join("account.toml");
            let checksum = StateManifest::compute_checksum(&account_file)?;
            manifest
                .checksums
                .insert(rel.to_string_lossy().to_string(), checksum);
        }

        Ok(())
    }

    /// Write a trade journal entry (JSON increment file)
    pub fn write_trade_entry<T: Serialize>(
        &self,
        uuid: &Uuid,
        trade_id: &str,
        entry: &T,
    ) -> QuantResult<PathBuf> {
        let dir = self.account_dir(uuid).join("trades");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.json", trade_id));
        self.write_json(
            &Path::new("accounts").join(uuid.to_string()).join("trades").join(format!("{}.json", trade_id)),
            entry,
        )?;
        Ok(path)
    }

    /// Write an equity curve snapshot
    pub fn write_equity<T: Serialize>(
        &self,
        uuid: &Uuid,
        timestamp: &DateTime<Utc>,
        snapshot: &T,
    ) -> QuantResult<PathBuf> {
        let dir = self.account_dir(uuid).join("equity");
        std::fs::create_dir_all(&dir)?;
        let filename = format!("{}.json", timestamp.format("%Y%m%d_%H%M%S"));
        let rel = Path::new("accounts")
            .join(uuid.to_string())
            .join("equity")
            .join(&filename);
        self.write_json(&rel, snapshot)?;
        Ok(self.state_root.join(rel))
    }

    /// Write an evolution cycle log
    pub fn write_evolution_log<T: Serialize>(
        &self,
        uuid: &Uuid,
        cycle: u64,
        log: &T,
    ) -> QuantResult<PathBuf> {
        let rel = Path::new("accounts")
            .join(uuid.to_string())
            .join("evolution")
            .join(format!("cycle_{}.json", cycle));
        self.write_json(&rel, log)?;
        Ok(self.state_root.join(rel))
    }

    /// List all accounts present in the state directory (from subdirs)
    pub fn list_account_uuids(&self) -> QuantResult<Vec<Uuid>> {
        let accounts_dir = self.state_root.join("accounts");
        if !accounts_dir.exists() {
            return Ok(Vec::new());
        }
        let mut uuids = Vec::new();
        for entry in std::fs::read_dir(&accounts_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Ok(uuid) = Uuid::parse_str(&entry.file_name().to_string_lossy()) {
                    uuids.push(uuid);
                }
            }
        }
        Ok(uuids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_initialize_state_tree() {
        let dir = tempdir().unwrap();
        let store = StateStore::new(dir.path().to_path_buf());
        store.initialize().unwrap();
        assert!(store.state_root().join("manifest.json").exists());
        assert!(store.state_root().join("accounts").exists());
    }

    #[test]
    fn test_account_dirs() {
        let dir = tempdir().unwrap();
        let store = StateStore::new(dir.path().to_path_buf());
        store.initialize().unwrap();

        let uuid = Uuid::new_v4();
        store.ensure_account_dirs(&uuid).unwrap();
        assert!(store.account_dir(&uuid).exists());
        assert!(store.account_dir(&uuid).join("strategies").exists());
        assert!(store.account_dir(&uuid).join("trades").exists());
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestState {
        name: String,
        value: u64,
    }

    #[test]
    fn test_write_read_round_trip() {
        let dir = tempdir().unwrap();
        let store = StateStore::new(dir.path().to_path_buf());
        store.initialize().unwrap();

        let uuid = Uuid::new_v4();
        store.ensure_account_dirs(&uuid).unwrap();

        let state = TestState {
            name: "test".into(),
            value: 42,
        };

        let rel = Path::new("accounts").join(uuid.to_string()).join("consistency.toml");
        store.write_toml(&rel, &state).unwrap();
        let loaded: TestState = store.read_toml(&rel).unwrap();
        assert_eq!(loaded, state);
    }

    #[test]
    fn test_manifest_checksum_record() {
        let dir = tempdir().unwrap();
        let store = StateStore::new(dir.path().to_path_buf());
        store.initialize().unwrap();

        let uuid = Uuid::new_v4();
        store.ensure_account_dirs(&uuid).unwrap();

        let state = TestState { name: "checksum".into(), value: 7 };
        let rel = Path::new("accounts").join(uuid.to_string()).join("account.toml");
        store.write_toml(&rel, &state).unwrap();

        let mut manifest = store.load_manifest().unwrap();
        store.record_account_checksums(&uuid, &mut manifest).unwrap();
        assert!(!manifest.checksums.is_empty());

        // Verify checksums pass
        let mismatches = manifest.verify_checksums(store.state_root());
        assert!(mismatches.is_empty());
    }
}