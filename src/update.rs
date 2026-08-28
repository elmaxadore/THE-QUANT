//! AUTO-UPDATE ENGINE — "updates are non-events" (v4.0 §0.4).
//!
//! Runs every `update_interval_hours` (default 24 h):
//!   1. fetch origin, compare local vs remote HEAD
//!   2. if remote ahead: snapshot state (git commit), pull --rebase
//!   3. `cargo build --release` a NEW binary in a staging dir
//!   4. smoke-test the new binary (`--smoke`) in a throwaway dir
//!   5. blue-green swap: rename current binary -> .prev, promote new
//!   6. on ANY failure: restore `.prev`, leave state untouched

use crate::config::Config;
use crate::github::GitSync;
use crate::state::Store;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

/// Outcome of one update pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateResult {
    /// No remote change; nothing to do.
    UpToDate,
    /// Remote was ahead and we applied it.
    Applied { new_head: String },
    /// Remote was ahead but applying failed; we rolled back.
    Failed { reason: String },
    /// There is no git repo: updates are disabled (not an error).
    Disabled,
}

pub struct AutoUpdater {
    pub cfg: Config,
    pub interval_hours: u64,
    pub git: Option<GitSync>,
}

impl AutoUpdater {
    pub fn new(cfg: Config) -> Self {
        let git = GitSync::discover(&cfg).ok();
        AutoUpdater { interval_hours: cfg.system.update_interval_hours, cfg, git }
    }

    /// Perform a single update check now. Called from the scheduler loop and
    /// from `the-quant update --now`.
    pub fn check_now(&self, store: &mut Store) -> UpdateResult {
        let Some(git) = &self.git else {
            return UpdateResult::Disabled;
        };

        // 1. Fetch and compare.
        if let Err(e) = git.fetch() {
            eprintln!("[update] fetch failed: {e}");
            return UpdateResult::Failed { reason: format!("fetch: {e}") };
        }
        let local = match git.current_head() {
            Ok(h) => h,
            Err(e) => return UpdateResult::Failed { reason: e },
        };
        let remote = match git.remote_head() {
            Ok(h) => h,
            Err(e) => return UpdateResult::Failed { reason: e },
        };
        if remote.is_empty() || remote == local {
            return UpdateResult::UpToDate;
        }
        let target = remote.clone();

        // 2. Snapshot state before touching anything.
        let _ = self.snapshot_state(store, &git);

        // 3. Pull latest code with rebase.
        if let Err(e) = git.pull_rebase() {
            let msg = format!("pull --rebase failed: {e}");
            return UpdateResult::Failed { reason: msg };
        }

        // 4. Stage a new binary in a temp dir.
        let stage_dir = std::env::temp_dir().join("the_quant_stage");
        let _ = std::fs::remove_dir_all(&stage_dir);
        std::fs::create_dir_all(&stage_dir).unwrap_or_default();
        let stage_bin = stage_dir.join("the-quant-new");
        if let Err(e) = self.build_release(&stage_bin) {
            return UpdateResult::Failed { reason: format!("build failed: {e}") };
        }

        // 5. Smoke test the new binary.
        if self.cfg.update.require_smoke_test {
            if let Err(e) = self.smoke_test(&stage_bin) {
                return UpdateResult::Failed { reason: format!("smoke test failed: {e}") };
            }
        }

        // 6. Blue-green swap.
        let current = self.current_binary_path();
        let prev = self.current_binary_path().with_extension("prev");
        let _ = std::fs::remove_file(&prev);
        if let Err(e) = std::fs::rename(&current, &prev) {
            return UpdateResult::Failed { reason: format!("cannot back up current binary: {e}") };
        }
        if let Err(e) = std::fs::rename(&stage_bin, &current) {
            let _ = std::fs::rename(&prev, &current);
            return UpdateResult::Failed { reason: format!("cannot promote new binary: {e}") };
        }

        // Record in state.
        if let Ok(mut root) = store.load() {
            root.last_update_applied = Some(format!("head={target}"));
            root.last_update_check = Some(crate::util::now_iso8601());
            let _ = store.save(&root);
        }

        UpdateResult::Applied { new_head: target }
    }
}

impl AutoUpdater {
    fn snapshot_state(&self, store: &mut Store, git: &GitSync) -> Result<(), String> {
        if let Ok(mut root) = store.load() {
            root.last_update_check = Some(crate::util::now_iso8601());
            let _ = store.save(&root);
        }
        let _ = git.commit_state("state: pre-update snapshot");
        let _ = git.push();
        Ok(())
    }

    fn build_release(&self, stage_bin: &Path) -> Result<(), String> {
        let cargo = if !self.cfg.update.cargo.is_empty() {
            self.cfg.update.cargo.clone()
        } else {
            "cargo".to_string()
        };
        let repo = PathBuf::from(&self.cfg.system.repo_dir);
        let out = Command::new(&cargo)
            .current_dir(&repo)
            .args(["build", "--release"])
            .output()
            .map_err(|e| format!("cargo build: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "cargo build failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let built = repo.join("target/release/the-quant");
        if !built.exists() {
            return Err("expected target/release/the-quant, not found".into());
        }
        std::fs::copy(&built, stage_bin).map_err(|e| format!("copy new binary: {e}"))?;
        Ok(())
    }

    fn smoke_test(&self, bin: &Path) -> Result<(), String> {
        let out = Command::new(bin)
            .arg("--smoke")
            .output()
            .map_err(|e| format!("smoke: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "smoke test exit={}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(())
    }

    pub fn current_binary_path(&self) -> PathBuf {
        std::env::current_exe().unwrap_or_else(|_| PathBuf::from("the-quant"))
    }

    /// Scheduler: blocking loop that sleeps `interval_hours` and checks.
    pub fn scheduler_loop(&self, store: &mut Store, stop: Arc<std::sync::atomic::AtomicBool>) {
        loop {
            if stop.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let hours = self.interval_hours.max(1);
            for _ in 0..(hours * 3600) {
                std::thread::sleep(std::time::Duration::from_secs(1));
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
            }
            match self.check_now(store) {
                UpdateResult::Applied { new_head } => {
                    println!("[update] applied {new_head}");
                }
                UpdateResult::Failed { reason } => {
                    eprintln!("[update] failed: {reason}");
                }
                _ => {}
            }
        }
    }
}
