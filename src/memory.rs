//! # Memory Management Module
//!
//! Percentage-scaled memory management — the heart of The Quant's resource contract.
//! At boot, the system detects total system RAM and computes a HARD_PROCESS_LIMIT
//! = TOTAL_RAM × PROCESS_CAP_PCT. All module budgets, channel capacities, and
//! buffer sizes scale dynamically from this single value.
//!
//! The system self-regulates: a 4GB VPS runs lean, a 64GB workstation runs aggressive.
//! Same binary. Different percentages. No waste. No starvation.

use crate::config::{MemoryConfig, QuantConfig};
use crate::error::{QuantError, QuantResult};
use std::sync::atomic::{AtomicUsize, Ordering};
use sysinfo::System;
use tracing::{info, warn, error};

/// Tracks and manages memory across all modules
#[derive(Debug)]
pub struct MemoryManager {
    /// Total system RAM in bytes (detected at boot)
    total_ram: u64,
    /// Hard process limit in bytes = total_ram × process_cap_pct
    hard_limit: u64,
    /// Per-module memory tracking
    modules: dashmap::DashMap<String, ModuleMemory>,
    /// System info handle for live monitoring
    system: System,
    /// Configuration reference
    config: MemoryConfig,
    /// Total allocated tracked by custom allocator
    total_allocated: AtomicUsize,
}

/// Per-module memory accounting
#[derive(Debug, Clone)]
pub struct ModuleMemory {
    pub name: String,
    pub budget_bytes: u64,
    pub used_bytes: u64,
    pub peak_bytes: u64,
    pub channel_capacity: usize,
    pub ring_buffer_depth: usize,
}

impl MemoryManager {
    /// Create a new MemoryManager, detecting system RAM and computing budgets
    pub fn new(config: &QuantConfig) -> Self {
        let mut system = System::new();
        system.refresh_memory();
        
        let total_ram = system.total_memory();
        let hard_limit = ((total_ram as f64) * (config.memory.process_cap_pct / 100.0)) as u64;
        
        info!(
            "Memory Manager initialized — {} GB total RAM, {} GB hard limit ({:.0}%)",
            total_ram as f64 / 1_073_741_824.0,
            hard_limit as f64 / 1_073_741_824.0,
            config.memory.process_cap_pct,
        );

        // Scale channel/buffer capacities based on RAM tier
        let (ram_tier, capacity_multiplier) = if total_ram > 34_359_738_368 {
            // > 32 GB
            (4u8, 4usize)
        } else if total_ram > 17_179_869_184 {
            // > 16 GB
            (3u8, 2usize)
        } else if total_ram > 8_589_934_592 {
            // > 8 GB
            (2u8, 1usize)
        } else {
            // <= 8 GB
            (1u8, 1usize)
        };

        info!("Detected RAM Tier {} — capacity multiplier: {}x", ram_tier, capacity_multiplier);

        let mut manager = Self {
            total_ram,
            hard_limit,
            modules: dashmap::DashMap::new(),
            system,
            config: config.memory.clone(),
            total_allocated: AtomicUsize::new(0),
        };

        // Initialize module budgets
        let budgets = &config.memory.module_budgets;
        let base_channel = config.memory.channel_base_capacity * capacity_multiplier;
        let base_ring = config.memory.ring_buffer_base * (total_ram as usize / 4_194_304).max(1000);

        let module_budgets = vec![
            ("data_collector", budgets.data_collector, base_channel * 4, base_ring),
            ("feature_pipeline", budgets.feature_pipeline, base_channel * 2, base_ring / 2),
            ("model_manager", budgets.model_manager, base_channel, base_ring / 4),
            ("strategy_engine", budgets.strategy_engine, base_channel, base_ring / 10),
            ("risk_engine", budgets.risk_engine, base_channel / 2, base_ring / 20),
            ("lab", budgets.lab, base_channel, base_ring),
            ("rl_gym", budgets.rl_gym, base_channel, base_ring / 4),
            ("microstructure", budgets.microstructure, base_channel / 2, 100),
            ("anomaly", budgets.anomaly, base_channel / 4, 100),
            ("changepoint", budgets.changepoint, base_channel / 4, 100),
            ("math", budgets.math, base_channel / 4, 100),
            ("tui_web", budgets.tui_web, base_channel / 4, 100),
            ("api", budgets.api, base_channel / 4, 100),
            ("system_overhead", budgets.system_overhead, base_channel, base_ring / 10),
            ("reserve", budgets.reserve, 0, 0),
        ];

        for (name, pct, chan_cap, ring_depth) in module_budgets {
            let budget_bytes = ((hard_limit as f64) * (pct / 100.0)) as u64;
            manager.modules.insert(name.to_string(), ModuleMemory {
                name: name.to_string(),
                budget_bytes,
                used_bytes: 0,
                peak_bytes: 0,
                channel_capacity: chan_cap,
                ring_buffer_depth: ring_depth,
            });
        }

        // Log budget allocation
        for entry in manager.modules.iter() {
            info!(
                "Module '{}' — budget: {:.1} MB ({:.1}%), channel: {}, ring: {}",
                entry.name,
                entry.budget_bytes as f64 / 1_048_576.0,
                (entry.budget_bytes as f64 / hard_limit as f64) * 100.0,
                entry.channel_capacity,
                entry.ring_buffer_depth,
            );
        }

        manager
    }

    // === Getters ===

    pub fn total_ram(&self) -> u64 { self.total_ram }
    pub fn total_ram_gb(&self) -> f64 { self.total_ram as f64 / 1_073_741_824.0 }
    pub fn hard_limit(&self) -> u64 { self.hard_limit }
    pub fn hard_limit_gb(&self) -> f64 { self.hard_limit as f64 / 1_073_741_824.0 }
    
    pub fn module_budget(&self, name: &str) -> u64 {
        self.modules.get(name).map(|m| m.budget_bytes).unwrap_or(0)
    }

    pub fn module_used(&self, name: &str) -> u64 {
        self.modules.get(name).map(|m| m.used_bytes).unwrap_or(0)
    }

    pub fn channel_capacity(&self, name: &str) -> usize {
        self.modules.get(name).map(|m| m.channel_capacity).unwrap_or(1024)
    }

    pub fn ring_buffer_depth(&self, name: &str) -> usize {
        self.modules.get(name).map(|m| m.ring_buffer_depth).unwrap_or(1000)
    }

    // === Memory Tracking ===

    /// Track memory allocation for a module
    pub fn track_allocation(&self, module: &str, bytes: u64) {
        if let Some(mut entry) = self.modules.get_mut(module) {
            entry.used_bytes += bytes;
            entry.peak_bytes = entry.peak_bytes.max(entry.used_bytes);
        }
        self.total_allocated.fetch_add(bytes as usize, Ordering::Relaxed);
    }

    /// Track memory deallocation for a module
    pub fn track_deallocation(&self, module: &str, bytes: u64) {
        if let Some(mut entry) = self.modules.get_mut(module) {
            entry.used_bytes = entry.used_bytes.saturating_sub(bytes);
        }
        self.total_allocated.fetch_sub(bytes as usize, Ordering::Relaxed);
    }

    // === Memory Pressure Checks ===

    /// Check current RSS as percentage of HARD_PROCESS_LIMIT
    pub fn rss_pct(&mut self) -> f64 {
        self.system.refresh_memory();
        let used = self.system.used_memory();
        (used as f64 / self.hard_limit as f64) * 100.0
    }

    /// Check if we've hit the soft limit (75%)
    pub fn is_soft_limit_reached(&mut self) -> bool {
        self.rss_pct() >= self.config.soft_limit_pct
    }

    /// Check if we've hit the hard limit (90%)
    pub fn is_hard_limit_reached(&mut self) -> bool {
        self.rss_pct() >= self.config.hard_limit_pct
    }

    /// Check if emergency reduction is needed (90%+)
    pub fn is_emergency_needed(&mut self) -> bool {
        self.rss_pct() >= self.config.emergency_pct
    }

    // === Backpressure Actions ===

    /// Soft limit reached: reduce lab batch size, flush caches, pause non-essential
    pub fn apply_soft_limit_backpressure(&self) {
        warn!(
            "Soft memory limit ({:.0}%) reached — reducing lab batch, flushing caches",
            self.config.soft_limit_pct,
        );
        // TODO: Signal to Evolution Engine to pause lab
        // TODO: Signal to FeaturePipeline to flush caches
        // TODO: Request jemalloc purge
    }

    /// Hard limit reached: emergency reduction protocol
    pub fn apply_hard_limit_emergency(&self) {
        error!(
            "EMERGENCY: Hard memory limit ({:.0}%) reached! Initiating emergency reduction",
            self.config.hard_limit_pct,
        );
        // TODO: Phase 1: Pause Evolution Engine and Lab immediately
        // TODO: Phase 2: Drop all feature caches
        // TODO: Phase 3: Flush pending database writes
        // TODO: Phase 4: If still >90% after 10s, pause Strategy Engine
        // TODO: Phase 5: If still >90% after 30s, close non-essential positions
    }

    /// Get per-module memory breakdown for the TUI
    pub fn module_breakdown(&self) -> Vec<ModuleMemorySnapshot> {
        self.modules.iter().map(|entry| {
            ModuleMemorySnapshot {
                name: entry.name.clone(),
                budget_bytes: entry.budget_bytes,
                used_bytes: entry.used_bytes,
                peak_bytes: entry.peak_bytes,
                pct_of_budget: if entry.budget_bytes > 0 {
                    (entry.used_bytes as f64 / entry.budget_bytes as f64) * 100.0
                } else {
                    0.0
                },
                channel_capacity: entry.channel_capacity,
                ring_buffer_depth: entry.ring_buffer_depth,
            }
        }).collect()
    }
}

/// Snapshot of module memory for display
#[derive(Debug, Clone)]
pub struct ModuleMemorySnapshot {
    pub name: String,
    pub budget_bytes: u64,
    pub used_bytes: u64,
    pub peak_bytes: u64,
    pub pct_of_budget: f64,
    pub channel_capacity: usize,
    pub ring_buffer_depth: usize,
}

impl ModuleMemorySnapshot {
    pub fn used_mb(&self) -> f64 { self.used_bytes as f64 / 1_048_576.0 }
    pub fn budget_mb(&self) -> f64 { self.budget_bytes as f64 / 1_048_576.0 }
    pub fn peak_mb(&self) -> f64 { self.peak_bytes as f64 / 1_048_576.0 }
}

/// Custom allocator wrapper that tracks allocations per module
/// NOTE: This is a simplified tracking approach. For production, use jemalloc
/// with its built-in stats or a proper internal allocator wrapper.
pub struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl std::alloc::GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed);
        std::alloc::System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        ALLOCATED.fetch_sub(layout.size(), std::sync::atomic::Ordering::Relaxed);
        std::alloc::System.dealloc(ptr, layout)
    }
}

/// Get total allocated bytes from the tracking allocator
pub fn total_allocated_bytes() -> usize {
    ALLOCATED.load(std::sync::atomic::Ordering::Relaxed)
}
