//! # RL Experience Replay Buffer
//!
//! A bounded, percentage-scaled replay buffer for storing (state, action,
//! reward, next_state, done) transitions. When full, the oldest samples are
//! dropped. Optional persistence to PostgreSQL for offline training runs.

use crate::error::{QuantError, QuantResult};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single transition in the replay buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: TransitionAction,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// Serialized action for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionAction {
    pub index: usize,
    pub scale: f64,
}

/// Bounded replay buffer with uniform sampling.
#[derive(Debug)]
pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
    rng: rand::rngs::StdRng,
}

impl ReplayBuffer {
    /// Create a new buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let seed = rand::random::<u64>();
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Create a buffer sized as a percentage of available memory.
    ///
    /// `budget_bytes` is the RL module's memory budget; each transition is
    /// estimated at `transition_bytes` (default ~1KB). Capacity is derived so
    /// the buffer fits within the budget.
    pub fn new_scaled(budget_bytes: u64, transition_bytes: u64) -> Self {
        let capacity = ((budget_bytes / transition_bytes.max(1)) as usize).clamp(1000, 5_000_000);
        Self::new(capacity)
    }

    /// Push a transition, dropping the oldest if full.
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }

    /// Sample a random batch of transitions.
    pub fn sample(&mut self, batch_size: usize) -> Vec<Transition> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let n = batch_size.min(self.buffer.len());
        let mut batch = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = rand::Rng::gen_range(&mut self.rng, 0..self.buffer.len());
            if let Some(t) = self.buffer.get(idx) {
                batch.push(t.clone());
            }
        }
        batch
    }

    /// Sample the most recent batch (for prioritized replay simplicity).
    pub fn sample_recent(&self, batch_size: usize) -> Vec<Transition> {
        let n = batch_size.min(self.buffer.len());
        self.buffer.iter().rev().take(n).cloned().collect()
    }

    /// Number of transitions in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Serialize the buffer to JSON (for persistence).
    pub fn to_json(&self) -> QuantResult<String> {
        let transitions: Vec<&Transition> = self.buffer.iter().collect();
        serde_json::to_string(&transitions).map_err(|e| QuantError::Internal(e.to_string()))
    }

    /// Load transitions from JSON.
    pub fn from_json(json: &str) -> QuantResult<Self> {
        let transitions: Vec<Transition> = serde_json::from_str(json)
            .map_err(|e| QuantError::Internal(e.to_string()))?;
        let capacity = transitions.len().max(1000);
        let mut buf = Self::new(capacity);
        for t in transitions {
            buf.push(t);
        }
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transition() -> Transition {
        Transition {
            state: vec![0.1, 0.2, 0.3],
            action: TransitionAction { index: 1, scale: 0.5 },
            reward: 0.01,
            next_state: vec![0.2, 0.3, 0.4],
            done: false,
        }
    }

    #[test]
    fn test_push_and_len() {
        let mut buf = ReplayBuffer::new(10);
        for _ in 0..5 {
            buf.push(make_transition());
        }
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_overflow_drops_oldest() {
        let mut buf = ReplayBuffer::new(3);
        for i in 0..5 {
            let mut t = make_transition();
            t.reward = i as f64;
            buf.push(t);
        }
        assert_eq!(buf.len(), 3);
        // Oldest (reward 0,1) should be dropped; newest (2,3,4) remain
        let recent = buf.sample_recent(3);
        let rewards: Vec<f64> = recent.iter().map(|t| t.reward).collect();
        assert_eq!(rewards, vec![4.0, 3.0, 2.0]);
    }

    #[test]
    fn test_sample() {
        let mut buf = ReplayBuffer::new(10);
        for _ in 0..10 {
            buf.push(make_transition());
        }
        let batch = buf.sample(5);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut buf = ReplayBuffer::new(10);
        for _ in 0..4 {
            buf.push(make_transition());
        }
        let json = buf.to_json().unwrap();
        let loaded = ReplayBuffer::from_json(&json).unwrap();
        assert_eq!(loaded.len(), 4);
    }

    #[test]
    fn test_scaled_capacity() {
        // 1MB budget, 1KB per transition → 1000 transitions
        let buf = ReplayBuffer::new_scaled(1_048_576, 1024);
        assert_eq!(buf.capacity(), 1000);
    }
}
