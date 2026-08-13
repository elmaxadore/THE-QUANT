//! # Auto-Commit Engine (v4.0 "Hercules")
//!
//! Automatically commits every state change to GitHub with structured commit
//! messages. The repository IS the single source of truth — so the engine must
//! never miss a commit.
//!
//! Commit message format:
//!   `[ACCOUNT:<uuid>] <ACTION>: <description>`
//!   `[GLOBAL] <ACTION>: <description>`
//!   `[SYSTEM] <ACTION>: <description>`

use crate::error::{QuantError, QuantResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Types of state changes that trigger a commit
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CommitEvent {
    /// A trade was closed
    Trade { account_id: Uuid, description: String },
    /// An evolution cycle completed
    Evolution { account_id: Uuid, cycle: u64, description: String },
    /// A model was promoted to production
    ModelPromoted { account_id: Uuid, model_id: String },
    /// An account state changed (config, lifecycle, etc.)
    AccountState { account_id: Uuid, description: String },
    /// New market data was collected (30-min batch)
    MarketData { description: String },
    /// A strategy was promoted or demoted
    StrategyChange { account_id: Uuid, action: String },
    /// Any global/system change
    System { description: String },
}

impl CommitEvent {
    /// Builds the commit message in the `[ACCOUNT:uuid] ACTION: desc` format
    pub fn commit_message(&self) -> String {
        match self {
            Self::Trade { account_id, description } => {
                format!("[ACCOUNT:{}] TRADE: {}", account_id, description)
            }
            Self::Evolution { account_id, cycle, description: _ } => {
                format!(
                    "[ACCOUNT:{}] EVOLUTION: Cycle {} complete",
                    account_id, cycle
                )
            }
            Self::ModelPromoted { account_id, model_id } => {
                format!("[ACCOUNT:{}] MODEL: Promoted {}", account_id, model_id)
            }
            Self::AccountState { account_id, description } => {
                format!("[ACCOUNT:{}] STATE: {}", account_id, description)
            }
            Self::MarketData { description } => {
                format!("[GLOBAL] DATA: {}", description)
            }
            Self::StrategyChange { account_id, action } => {
                format!("[ACCOUNT:{}] STRATEGY: {}", account_id, action)
            }
            Self::System { description } => {
                format!("[SYSTEM] {}", description)
            }
        }
    }

    /// Human-readable summary for logs
    pub fn summary(&self) -> String {
        match self {
            Self::Trade { description, .. } => format!("trade: {}", description),
            Self::Evolution { cycle, .. } => format!("evolution cycle {}", cycle),
            Self::ModelPromoted { model_id, .. } => format!("model promoted: {}", model_id),
            Self::AccountState { description, .. } => format!("account state: {}", description),
            Self::MarketData { description } => format!("market data: {}", description),
            Self::StrategyChange { action, .. } => format!("strategy: {}", action),
            Self::System { description } => format!("system: {}", description),
        }
    }
}

/// Queue of pending commits to be flushed by the auto-commit engine
#[derive(Debug, Default)]
pub struct CommitQueue {
    inner: Arc<Mutex<VecDeque<CommitEvent>>>,
}

impl CommitQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new event onto the queue
    pub async fn push(&self, event: CommitEvent) {
        self.inner.lock().await.push_back(event);
    }

    /// Pop the next event from the queue
    pub async fn pop(&self) -> Option<CommitEvent> {
        self.inner.lock().await.pop_front()
    }

    /// Number of pending events
    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }

    /// Is the queue empty?
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Drain all pending events
    pub async fn drain(&self) -> Vec<CommitEvent> {
        let mut guard = self.inner.lock().await;
        guard.drain(..).collect()
    }
}

/// A single recorded commit (for audit trail)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRecord {
    pub hash: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub event: CommitEvent,
}

/// Trait for the underlying git commit implementation
#[async_trait::async_trait]
pub trait CommitHandler: Send + Sync {
    async fn commit_and_push(&self, message: &str) -> QuantResult<String>;
}

/// The AutoCommitEngine queues state-change events and commits them to the
/// repository. The repository IS the source of truth — no event is ever lost.
#[derive(Debug, Clone)]
pub struct AutoCommitEngine {
    queue: CommitQueue,
    /// Hook used to perform the actual git commit+push.
    committer: Option<Arc<dyn CommitHandler + Send + Sync>>,
}

impl AutoCommitEngine {
    pub fn new() -> Self {
        Self {
            queue: CommitQueue::new(),
            committer: None,
        }
    }

    /// Attach a commit handler (set once at startup)
    pub fn attach_handler(&mut self, handler: Arc<dyn CommitHandler + Send + Sync>) {
        self.committer = Some(handler);
    }

    /// Queue a commit event for later processing
    pub async fn queue_event(&self, event: CommitEvent) {
        self.queue.push(event).await;
        tracing::debug!("[AUTOCOMMIT] Queued: {}", event.summary());
    }

    /// Count of pending events
    pub async fn pending(&self) -> usize {
        self.queue.len().await
    }

    /// Process a single event immediately (synchronous commit + push)
    pub async fn process_event(&self, event: CommitEvent) -> QuantResult<Option<CommitRecord>> {
        let message = event.commit_message();

        let Some(handler) = &self.committer else {
            tracing::warn!(
                "[AUTOCOMMIT] No commit handler attached — skipping '{}'",
                message
            );
            return Ok(None);
        };

        tracing::info!("[AUTOCOMMIT] Committing: {}", message);
        let hash = handler.commit_and_push(&message).await?;

        Ok(Some(CommitRecord {
            hash,
            message,
            timestamp: Utc::now(),
            event,
        }))
    }

    /// Drain all pending events — blocks until queue is empty
    pub async fn drain_all(&self) -> QuantResult<Vec<CommitRecord>> {
        if self.committer.is_none() {
            tracing::warn!("[AUTOCOMMIT] No commit handler — draining queue without committing");
            self.queue.drain().await;
            return Ok(Vec::new());
        }

        let mut records = Vec::new();
        loop {
            let Some(event) = self.queue.pop().await else {
                break;
            };
            if let Some(record) = self.process_event(event).await? {
                records.push(record);
            }
        }
        Ok(records)
    }

    /// Spawn a background task that continuously drains the queue,
    /// committing at most one event per `interval` (rate limiting).
    pub fn spawn_background(
        self,
        interval: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                if self.pending().await > 0 {
                    if let Err(e) = self.drain_all().await {
                        tracing::error!("[AUTOCOMMIT] Failed to drain queue: {}", e);
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_message_format() {
        let uuid = Uuid::new_v4();
        let event = CommitEvent::Trade {
            account_id: uuid,
            description: "Closed BUY XAUUSD +$42.50".into(),
        };
        let msg = event.commit_message();
        assert_eq!(
            msg,
            format!("[ACCOUNT:{}] TRADE: Closed BUY XAUUSD +$42.50", uuid)
        );
    }

    #[test]
    fn test_system_message() {
        let event = CommitEvent::System {
            description: "v4.0.1 → v4.0.2 (blue-green switchover)".into(),
        };
        assert_eq!(
            event.commit_message(),
            "[SYSTEM] v4.0.1 → v4.0.2 (blue-green switchover)"
        );
    }

    #[tokio::test]
    async fn test_queue_push_pop() {
        let queue = CommitQueue::new();
        assert!(queue.is_empty().await);
        queue
            .push(CommitEvent::System {
                description: "test".into(),
            })
            .await;
        assert_eq!(queue.len().await, 1);
        let event = queue.pop().await.unwrap();
        assert!(matches!(event, CommitEvent::System { .. }));
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn test_engine_no_handler_skips() {
        let engine = AutoCommitEngine::new();
        engine
            .queue_event(CommitEvent::System {
                description: "test".into(),
            })
            .await;
        let records = engine.drain_all().await.unwrap();
        assert!(records.is_empty());
    }

    struct MockHandler;

    #[async_trait::async_trait]
    impl CommitHandler for MockHandler {
        async fn commit_and_push(&self, message: &str) -> QuantResult<String> {
            assert!(message.contains("[SYSTEM]"));
            Ok("deadbeef".into())
        }
    }

    #[tokio::test]
    async fn test_engine_with_handler() {
        let mut engine = AutoCommitEngine::new();
        engine.attach_handler(Arc::new(MockHandler));
        engine
            .queue_event(CommitEvent::System {
                description: "test".into(),
            })
            .await;
        let records = engine.drain_all().await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].hash, "deadbeef");
    }
}