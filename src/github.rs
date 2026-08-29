//! GITHUB / GIT SYNC ENGINE — the system backbone (v4.0 §1).
//!
//! Every piece of state that matters lives in the repo. This module wraps the
//! `git` CLI (no heavy git2 FFI) to commit+push state changes and to fetch the
//! latest code. It is deliberately small and dependency-free.

use crate::config::Config;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct GitSync {
    pub repo_dir: PathBuf,
    pub git_bin: String,
    /// If true, we also push to origin (requires PAT configured in the vault).
    pub push_enabled: bool,
}

pub fn run_git(repo: &Path, git_bin: &str, args: &[&str]) -> Result<String, String> {
    let mut cmd = Command::new(git_bin);
    cmd.arg("-C").arg(repo);
    cmd.args(args);
    let out = cmd
        .output()
        .map_err(|e| format!("git {}: {e}", args.first().unwrap_or(&"")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("git {} failed: {}", args.join(" "), stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

impl GitSync {
    pub fn discover(cfg: &Config) -> Result<Self, String> {
        let repo_dir = PathBuf::from(&cfg.system.repo_dir);
        let git_bin = if !cfg.update.git.is_empty() {
            cfg.update.git.clone()
        } else {
            "git".to_string()
        };
        // Check the repo is actually a git repository.
        if repo_dir.join(".git").exists() {
            Ok(GitSync { repo_dir, git_bin, push_enabled: !cfg.system.repo_remote.is_empty() })
        } else {
            // Provide a usable error describing how to bootstrap.
            Err(format!(
                "no git repo at '{}'. Run `git init` here and add a remote, or set system.repo_remote.",
                repo_dir.display()
            ))
        }
    }

    pub fn current_head(&self) -> Result<String, String> {
        run_git(&self.repo_dir, &self.git_bin, &["rev-parse", "HEAD"])
    }

    /// Current branch name, e.g. "main". Falls back to "main" when detached.
    pub fn current_branch(&self) -> String {
        run_git(&self.repo_dir, &self.git_bin, &["rev-parse", "--abbrev-ref", "HEAD"])
            .map(|s| s.trim().to_string())
            .ok()
            .filter(|s| !s.is_empty() && s != "HEAD")
            .unwrap_or_else(|| "main".to_string())
    }

    /// The commit the remote's tracking branch points at.
    ///
    /// NOTE: `ls-remote origin HEAD` is NOT reliable — some transports (notably
    /// local file remotes) do not advertise a HEAD symref, which would make the
    /// updater silently think there is nothing to update. We therefore ask for
    /// the current branch's ref explicitly.
    pub fn remote_head(&self) -> Result<String, String> {
        let branch = self.current_branch();
        let spec = format!("refs/heads/{branch}");
        let out = run_git(&self.repo_dir, &self.git_bin, &["ls-remote", "origin", &spec])?;
        // Fall back to any advertised head if the branch is missing upstream.
        if out.is_empty() {
            let all = run_git(&self.repo_dir, &self.git_bin, &["ls-remote", "origin", "refs/heads/*"])?;
            return Ok(all.lines().next().unwrap_or("").split_whitespace().next().unwrap_or("").to_string());
        }
        Ok(out.split_whitespace().next().unwrap_or("").to_string())
    }

    /// Fetch origin without touching working tree.
    pub fn fetch(&self) -> Result<(), String> {
        run_git(&self.repo_dir, &self.git_bin, &["fetch", "origin", "--prune"]).map(|_| ())
    }

    /// Commit any dirty state dir files. Returns true if something was committed.
    pub fn commit_state(&self, message: &str) -> Result<bool, String> {
        run_git(&self.repo_dir, &self.git_bin, &["add", "-A"]).map_err(|e| e.clone())?;
        let status = run_git(&self.repo_dir, &self.git_bin, &["status", "--porcelain"])?;
        if status.is_empty() {
            return Ok(false);
        }
        run_git(&self.repo_dir, &self.git_bin, &["commit", "-m", message])?;
        Ok(true)
    }

    /// Push committed state to origin (only if push_enabled).
    pub fn push(&self) -> Result<(), String> {
        if !self.push_enabled {
            return Ok(());
        }
        run_git(&self.repo_dir, &self.git_bin, &["push", "origin", "HEAD"]).map(|_| ())
    }

    /// Pull latest code with rebase. On conflict returns Err so update.rs can
    /// roll back safely.
    pub fn pull_rebase(&self) -> Result<(), String> {
        run_git(&self.repo_dir, &self.git_bin, &["pull", "--rebase", "origin", "HEAD"]).map(|_| ())
    }

    /// Status of working tree: dirty or clean.
    pub fn is_dirty(&self) -> Result<bool, String> {
        let s = run_git(&self.repo_dir, &self.git_bin, &["status", "--porcelain"])?;
        Ok(!s.is_empty())
    }
}