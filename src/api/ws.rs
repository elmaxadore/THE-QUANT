//! # WebSocket Streaming (v3.0 "Prometheus")
//!
//! Real-time streaming of quotes, positions, regimes, and system status to
//! connected web clients via WebSocket. Broadcasts are pushed from the
//! internal message bus to subscribed clients.

use crate::api::server::WsEvent;
use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn};

/// A single connected WebSocket client.
#[derive(Debug)]
pub struct WsClient {
    pub id: String,
    pub subscriptions: Vec<String>,
    pub sender: mpsc::UnboundedSender<String>,
}

/// The WebSocket hub — manages clients and broadcasts events.
#[derive(Debug)]
pub struct WsHub {
    clients: Arc<RwLock<HashMap<String, WsClient>>>,
    /// Event history for late-joining clients (bounded).
    history: Arc<RwLock<std::collections::VecDeque<String>>>,
    max_history: usize,
}

impl WsHub {
    pub fn new() -> Self {
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            max_history: 1000,
        }
    }

    /// Register a client.
    pub async fn register(&self, client: WsClient) {
        self.clients.write().await.insert(client.id.clone(), client);
    }

    /// Unregister a client.
    pub async fn unregister(&self, client_id: &str) {
        self.clients.write().await.remove(client_id);
    }

    /// Broadcast an event to all clients.
    pub async fn broadcast(&self, event: &WsEvent) -> QuantResult<()> {
        let json = serde_json::to_string(event).map_err(|e| QuantError::Internal(e.to_string()))?;

        // Push to history
        {
            let mut history = self.history.write().await;
            history.push_back(json.clone());
            if history.len() > self.max_history {
                history.pop_front();
            }
        }

        // Send to all clients
        let clients = self.clients.read().await;
        for (_, client) in clients.iter() {
            let _ = client.sender.send(json.clone());
        }
        Ok(())
    }

    /// Broadcast to clients subscribed to a channel.
    pub async fn broadcast_to(&self, channel: &str, event: &WsEvent) -> QuantResult<()> {
        let json = serde_json::to_string(event).map_err(|e| QuantError::Internal(e.to_string()))?;
        let clients = self.clients.read().await;
        for (_, client) in clients.iter() {
            if client.subscriptions.iter().any(|s| s == channel) {
                let _ = client.sender.send(json.clone());
            }
        }
        Ok(())
    }

    /// Send a private event to a single client.
    pub async fn send_to(&self, client_id: &str, event: &WsEvent) -> QuantResult<()> {
        let json = serde_json::to_string(event).map_err(|e| QuantError::Internal(e.to_string()))?;
        let clients = self.clients.read().await;
        if let Some(client) = clients.get(client_id) {
            let _ = client.sender.send(json);
        }
        Ok(())
    }

    /// Number of connected clients.
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Get recent event history.
    pub async fn history(&self, limit: usize) -> Vec<String> {
        let history = self.history.read().await;
        history.iter().rev().take(limit).cloned().collect()
    }
}

impl Default for WsHub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_broadcast() {
        let hub = WsHub::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let client = WsClient {
            id: "c1".into(),
            subscriptions: vec!["quotes".into()],
            sender: tx,
        };
        hub.register(client).await;
        assert_eq!(hub.client_count().await, 1);

        hub.broadcast(&WsEvent::Quote {
            symbol: "EURUSD".into(),
            bid: 1.1,
            ask: 1.11,
            time: 0,
        }).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert!(received.contains("EURUSD"));
    }

    #[tokio::test]
    async fn test_broadcast_to_channel() {
        let hub = WsHub::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let client = WsClient {
            id: "c1".into(),
            subscriptions: vec!["quotes".into()],
            sender: tx,
        };
        hub.register(client).await;

        hub.broadcast_to("quotes", &WsEvent::Quote {
            symbol: "XAUUSD".into(),
            bid: 2000.0,
            ask: 2000.1,
            time: 0,
        }).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert!(received.contains("XAUUSD"));
    }

    #[tokio::test]
    async fn test_history() {
        let hub = WsHub::new();
        hub.broadcast(&WsEvent::System { memory_pct: 50.0, cpu_pct: 20.0 }).await.unwrap();
        let history = hub.history(10).await;
        assert_eq!(history.len(), 1);
    }
}
