//! # GitHub Push (v4.0 "Hercules")
//!
//! Provides authenticated push-to-remote for the Git-centric state architecture.
//! Uses git2 with credential callbacks for HTTPS PAT authentication.

use crate::config::QuantConfig;
use crate::error::{QuantError, QuantResult};
use git2::{PushOptions, RemoteCallbacks, Repository};
use std::path::PathBuf;
use tracing::info;

/// Handles pushing local commits to a remote GitHub repository
#[derive(Debug, Clone)]
pub struct GitHubPush {
    repo_path: PathBuf,
    remote_url: String,
    pat: Option<String>,
    branch: String,
}

impl GitHubPush {
    pub fn new(config: &QuantConfig) -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let repo_path = PathBuf::from(&home).join("the-quant");

        // Determine branch from config (defaults to "main" in GitHubConfig::default)
        let branch = config.github.branch.clone();

        Self {
            repo_path,
            remote_url: config.github.repo_url.clone(),
            pat: if config.github.pat.is_empty() {
                None
            } else {
                Some(config.github.pat.clone())
            },
            branch,
        }
    }

    /// Push all committed changes to the configured remote
    pub async fn push(&self) -> QuantResult<()> {
        if self.remote_url.is_empty() {
            info!("No remote URL configured — skipping push");
            return Ok(());
        }

        let repo = Repository::open(&self.repo_path)
            .map_err(|e| QuantError::GitError(format!("Failed to open repo for push: {}", e)))?;

        // Find remote
        let mut remote = repo.find_remote("origin")
            .map_err(|e| QuantError::GitError(format!("Failed to find remote 'origin': {}", e)))?;

        // Set up callbacks with credentials
        let mut callbacks = RemoteCallbacks::new();

        // If PAT is set, use it for authentication
        if let Some(pat) = &self.pat {
            let pat_clone = pat.clone();
            callbacks.credentials(move |_url, username_from_url, _allowed_types| {
                git2::Cred::userpass_plaintext(
                    username_from_url.unwrap_or("elmaxadore"),
                    &pat_clone,
                )
            });
        }

        // Configure push options
        let mut options = PushOptions::new();
        options.remote_callbacks(callbacks);

        let refspec = format!("refs/heads/{}:refs/heads/{}", self.branch, self.branch);

        info!("Pushing to {}", self.remote_url);
        remote.push(&[&refspec], Some(&mut options))
            .map_err(|e| QuantError::GitError(format!("Failed to push: {}", e)))?;

        info!("Push completed successfully");
        Ok(())
    }

    /// Fetch the latest from the remote (for update checks)
    pub async fn fetch(&self) -> QuantResult<()> {
        if self.remote_url.is_empty() {
            return Ok(());
        }

        let repo = Repository::open(&self.repo_path)
            .map_err(|e| QuantError::GitError(format!("Failed to open repo for fetch: {}", e)))?;

        let mut remote = repo.find_remote("origin")
            .map_err(|e| QuantError::GitError(format!("Failed to find remote 'origin': {}", e)))?;

        let mut callbacks = RemoteCallbacks::new();

        if let Some(pat) = &self.pat {
            let pat_clone = pat.clone();
            callbacks.credentials(move |_url, _username_url, _cred_types| {
                git2::Cred::userpass_plaintext("elmaxadore", &pat_clone)
            });
        }

        let mut options = git2::FetchOptions::new();
        options.remote_callbacks(callbacks);
        options.depth(1);

        info!("Fetching from {}", self.remote_url);
        remote.fetch(&["refs/heads/*:refs/remotes/origin/*"], Some(&mut options), None)
            .map_err(|e| QuantError::GitError(format!("Failed to fetch: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_new_disabled() {
        let config = QuantConfig::default();
        let push = GitHubPush::new(&config);
        assert!(push.remote_url.is_empty());
    }
}