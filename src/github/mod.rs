//! # GitHub Integration Module (Layer 11) — v4.0 "Hercules"
//!
//! Git-centric state architecture. The GitHub repo IS the single source of
//! truth for ALL state. This module provides:
//!
//! - `GitHubSync` — low-level git commit/push/tag operations
//! - `AutoCommitEngine` — event-driven state commit queueing
//! - `GitHubCommitHandler` — bridges AutoCommitEngine to GitHubSync

pub mod auto_commit;
pub mod push;

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use crate::github::auto_commit::CommitHandler;
use chrono::{DateTime, Utc};
use git2::{IndexAddOption, Repository, Signature};
use octocrab::Octocrab;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

pub use auto_commit::{AutoCommitEngine, CommitEvent, CommitQueue, CommitRecord};
pub use push::GitHubPush;

/// Status of the GitHub sync
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyncStatus {
    Idle,
    Syncing { phase: String },
    Success { last_sync: DateTime<Utc> },
    Failed { error: String, last_attempt: DateTime<Utc> },
}

/// GitHub sync manager
#[derive(Debug)]
pub struct GitHubSync {
    /// Repository handle
    repo: Arc<RwLock<Option<Repository>>>,
    /// GitHub API client
    octocrab: Arc<RwLock<Option<Octocrab>>>,
    /// Current sync status
    status: Arc<RwLock<SyncStatus>>,
    /// Repository path
    repo_path: PathBuf,
    /// Configuration
    config: QuantConfig,
    /// Whether initial setup is complete
    initialized: Arc<RwLock<bool>>,
    /// Push helper
    pusher: GitHubPush,
}

impl GitHubSync {
    pub fn new(config: &QuantConfig) -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let repo_path = PathBuf::from(&home).join("the-quant");

        let pusher = GitHubPush::new(config);

        Self {
            repo: Arc::new(RwLock::new(None)),
            octocrab: Arc::new(RwLock::new(None)),
            status: Arc::new(RwLock::new(SyncStatus::Idle)),
            repo_path,
            config: config.clone(),
            initialized: Arc::new(RwLock::new(false)),
            pusher,
        }
    }

    /// Initialize GitHub integration
    pub async fn initialize(&self) -> QuantResult<()> {
        if !self.config.github.enabled {
            info!("GitHub integration disabled");
            return Ok(());
        }

        // Initialize GitHub API client
        let octocrab = Octocrab::builder()
            .personal_token(self.config.github.pat.clone())
            .build()
            .map_err(|e| QuantError::GitHubError(format!("Failed to create GitHub client: {}", e)))?;

        *self.octocrab.write().await = Some(octocrab);

        // Open or initialize local repository
        if self.repo_path.join(".git").exists() {
            let repo = Repository::open(&self.repo_path)
                .map_err(|e| QuantError::GitError(format!("Failed to open repo: {}", e)))?;
            *self.repo.write().await = Some(repo);
            info!("Opened existing repository at {:?}", self.repo_path);
        } else {
            info!(
                "Repository not found at {:?}, will clone on first sync",
                self.repo_path
            );
        }

        *self.initialized.write().await = true;
        info!("GitHub integration initialized");
        Ok(())
    }

    /// Sync all changes to GitHub (commit + push)
    pub async fn sync(&self, message: &str) -> QuantResult<SyncStatus> {
        if !*self.initialized.read().await {
            self.initialize().await?;
        }

        *self.status.write().await = SyncStatus::Syncing {
            phase: "committing".into(),
        };

        // Ensure repository exists
        if self.repo.read().await.is_none() {
            self.clone_or_init_repo().await?;
        }

        let hash = {
            let repo = self.repo.read().await;
            let repo = repo
                .as_ref()
                .ok_or_else(|| QuantError::GitError("Repository not available".into()))?;

            // Stage all changes
            let mut index = repo
                .index()
                .map_err(|e| QuantError::GitError(format!("Failed to get index: {}", e)))?;
            index
                .add_all(["*"].iter(), IndexAddOption::DEFAULT, None)
                .map_err(|e| QuantError::GitError(format!("Failed to add files: {}", e)))?;
            index
                .write()
                .map_err(|e| QuantError::GitError(format!("Failed to write index: {}", e)))?;

            let tree_id = index
                .write_tree()
                .map_err(|e| QuantError::GitError(format!("Failed to write tree: {}", e)))?;
            let tree = repo
                .find_tree(tree_id)
                .map_err(|e| QuantError::GitError(format!("Failed to find tree: {}", e)))?;

            // Create commit
            let signature = Signature::now("The Quant", "quant@thequant.dev")
                .map_err(|e| QuantError::GitError(format!("Failed to create signature: {}", e)))?;

            // Skip if no changes
            let head = repo.head().ok();
            let parent_commit = head
                .as_ref()
                .and_then(|head| head.target().map(|oid| repo.find_commit(oid).ok()))
                .flatten();

            if let Some(parent) = &parent_commit {
                let parent_tree = parent.tree().ok();
                if let Some(parent_tree) = parent_tree {
                    if parent_tree.id() == tree.id() {
                        info!("No changes to commit — skipping empty commit");
                        return Ok(SyncStatus::Idle);
                    }
                }
            }

            let parents: Vec<&git2::Commit> = parent_commit.as_ref().into_iter().collect();

            let commit = repo
                .commit(
                    Some("HEAD"),
                    &signature,
                    &signature,
                    message,
                    &tree,
                    &parents,
                )
                .map_err(|e| QuantError::GitError(format!("Failed to commit: {}", e)))?;

            commit.to_string()
        };

        info!("Committed: {} ({})", message, hash);

        // Push to remote
        *self.status.write().await = SyncStatus::Syncing {
            phase: "pushing".into(),
        };

        if self.config.github.enabled {
            if let Err(e) = self.pusher.push().await {
                warn!("Push failed (will retry later): {}", e);
            }
        }

        let status = SyncStatus::Success {
            last_sync: Utc::now(),
        };
        *self.status.write().await = status.clone();

        info!("GitHub sync completed: {}", message);
        Ok(status)
    }

    /// Clone the repository or initialize a new one
    pub async fn clone_or_init_repo(&self) -> QuantResult<()> {
        if self.config.github.repo_url.is_empty() {
            // Initialize a new local repository
            std::fs::create_dir_all(&self.repo_path)?;
            let repo = Repository::init(&self.repo_path)
                .map_err(|e| QuantError::GitError(format!("Failed to init repo: {}", e)))?;
            *self.repo.write().await = Some(repo);
            info!("Initialized new repository at {:?}", self.repo_path);
        } else {
            // Clone existing repository
            let url = if self.config.github.pat.is_empty() {
                self.config.github.repo_url.clone()
            } else {
                // Insert PAT into URL for authentication
                let mut parts: Vec<&str> = self.config.github.repo_url.split("://").collect();
                if parts.len() == 2 {
                    format!("{}://{}@{}", parts[0], self.config.github.pat, parts[1])
                } else {
                    self.config.github.repo_url.clone()
                }
            };

            info!("Cloning repository from {}...", self.config.github.repo_url);
            let repo = Repository::clone(&url, &self.repo_path)
                .map_err(|e| QuantError::GitError(format!("Failed to clone repo: {}", e)))?;
            *self.repo.write().await = Some(repo);
            info!("Repository cloned successfully");
        }
        Ok(())
    }

    /// Force-push all pending changes to the remote
    pub async fn force_sync(&self) -> QuantResult<SyncStatus> {
        let message = format!(
            "Auto-commit: The Quant sync - {}",
            Utc::now().format("%Y-%m-%d %H:%M:%S")
        );
        self.sync(&message).await
    }

    /// Create a git tag for evolution cycles
    pub async fn tag_evolution(&self, version: &str, cycle_num: u64) -> QuantResult<()> {
        let repo = self.repo.read().await;
        if let Some(repo) = repo.as_ref() {
            let tag_name = format!("v{}-evolution-{}", version, cycle_num);

            let head = repo
                .head()
                .map_err(|e| QuantError::GitError(format!("Failed to get HEAD: {}", e)))?
                .target()
                .ok_or_else(|| QuantError::GitError("No HEAD target".into()))?;

            let obj = repo
                .find_object(head, None)
                .map_err(|e| QuantError::GitError(format!("Failed to find object: {}", e)))?;

            let signature = Signature::now("The Quant", "quant@thequant.dev")
                .map_err(|e| QuantError::GitError(format!("Failed to create signature: {}", e)))?;

            repo.tag(
                &tag_name,
                &obj,
                &signature,
                &format!("Evolution cycle {}", cycle_num),
                false,
            )
            .map_err(|e| QuantError::GitError(format!("Failed to create tag: {}", e)))?;

            info!("Created tag: {}", tag_name);
        }
        Ok(())
    }

    /// Get current sync status
    pub async fn get_status(&self) -> SyncStatus {
        self.status.read().await.clone()
    }

    /// Add a file pattern to .gitignore
    pub async fn update_gitignore(&self) -> QuantResult<()> {
        let gitignore_path = self.repo_path.join(".gitignore");
        let content = if gitignore_path.exists() {
            std::fs::read_to_string(&gitignore_path)?
        } else {
            String::new()
        };

        let needed_entries = [
            "data/raw/",
            "data/cache/",
            "*.enc",
            "*.log",
            "target/",
            ".env",
        ];

        let mut needs_update = false;
        for entry in &needed_entries {
            if !content.contains(entry) {
                needs_update = true;
            }
        }

        if needs_update {
            let mut new_content = content;
            for entry in &needed_entries {
                if !new_content.contains(entry) {
                    new_content.push_str(entry);
                    new_content.push('\n');
                }
            }
            std::fs::write(&gitignore_path, new_content)?;
            info!("Updated .gitignore");
        }

        Ok(())
    }

    /// Return the repo path for the current environment
    pub fn repo_path(&self) -> &PathBuf {
        &self.repo_path
    }
}

/// Adapter that bridges the AutoCommitEngine to GitHubSync
#[derive(Clone)]
pub struct GitHubCommitHandler {
    sync: Arc<GitHubSync>,
}

impl GitHubCommitHandler {
    pub fn new(sync: Arc<GitHubSync>) -> Self {
        Self { sync }
    }
}

#[async_trait::async_trait]
impl CommitHandler for GitHubCommitHandler {
    async fn commit_and_push(&self, message: &str) -> QuantResult<String> {
        match self.sync.sync(message).await {
            Ok(_) => {
                // Get the last commit hash for the record
                let path = self.sync.repo_path().clone();
                let repo = Repository::open(&path)?;
                let head = repo.head()?;
                let oid = head
                    .target()
                    .ok_or_else(|| QuantError::GitError("No HEAD".into()))?;
                Ok(oid.to_string())
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init_disabled() {
        let config = QuantConfig::default(); // GitHub disabled by default
        let sync = GitHubSync::new(&config);
        assert!(sync.initialize().await.is_ok());
    }
}