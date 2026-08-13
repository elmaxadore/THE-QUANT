//! # Account Pipeline & Rotation System (v3.1 Hephaestus)
//!
//! Manages accounts as fungible assets in a pipeline:
//! ACQUIRED → WARMING → TRADING → EXTRACTING → PAYOUT_PENDING → RETIRED / BLOWN
//! Maintains a bench of warm accounts and automatically rotates capital/focus when an account reaches payout cap or blows.

pub mod manager;

pub use manager::{
    AccountLifecycle, ManagedAccount, PipelineConfig, PipelineManager, ReadinessScore,
};
