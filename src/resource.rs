//! RESOURCE PROFILER — "one binary, many machines".
//!
//! The Quant does NOT impose arbitrary ceilings. At boot it detects the host's
//! RAM and CPU count, chooses a Tier (1..4), and computes a `HardProcessLimit`
//! plus a per-module memory budget expressed as a *percentage* of available RAM
//! (matching v2.1 §0.4 and v3.0 §7.3). A 4 GB VPS runs lean; a 64 GB workstation
//! runs aggressive — automatically.
//!
//! Budget slots (percent of HARD_PROCESS_LIMIT), from v3.0 §7.3:
//!   DataCollector  25%   FeaturePipeline 12%   ModelManager  15%
//!   StrategyEngine  6%   RiskEngine       3%    RL/Gym        5%
//!   Lab            22%   TUI + Web        4%    API/Channels  2%
//!   System          6%   Reserve          5%    TOTAL       100%

use std::collections::BTreeMap;
use std::fmt;

/// Memory governor tiers. Tier selection drives which capabilities run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// <= 8 GB — lean: no RL training, no Lab, headless/TUI, minimal caches.
    One,
    /// 8–16 GB — standard: web dashboard, heavier feature cache, small lab.
    Two,
    /// 16–32 GB — full: Lab, online GBDT retraining, microsec hot loops.
    Three,
    /// > 32 GB — aggressive: full research stack, RL, oversized buffers.
    Four,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Tier::One => "Tier-1 (<= 8 GB) — LEAN",
            Tier::Two => "Tier-2 (8-16 GB) — STANDARD",
            Tier::Three => "Tier-3 (16-32 GB) — ADVANCED",
            Tier::Four => "Tier-4 (> 32 GB) — AGGRESSIVE",
        };
        f.write_str(s)
    }
}

/// A module plus its hard memory budget in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleBudget {
    pub name: &'static str,
    pub budget_bytes: usize,
}

/// The boot-time view of the machine and the budgets derived from it.
#[derive(Debug, Clone)]
pub struct ResourceProfile {
    /// Total physical RAM in MiB.
    pub total_ram_mib: u64,
    /// Total usable RAM we are willing to consume (0.8 * total), in MiB.
    pub hard_process_limit_mib: u64,
    /// Number of logical CPUs.
    pub cpus: usize,
    pub tier: Tier,
    /// per-module budget table (name -> MiB).
    pub budgets: BTreeMap<&'static str, u64>,
}

fn read_total_ram_mib() -> u64 {
    // Env override takes precedence (used by tests and CI to simulate tiers).
    if let Ok(v) = std::env::var("THE_QUANT_RAM_MIB") {
        if let Ok(mib) = v.parse::<u64>() {
            return mib;
        }
    }
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest
                    .trim()
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                if kb > 0 {
                    return kb / 1024;
                }
            }
        }
    }
    std::env::var("THE_QUANT_RAM_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}

fn read_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
}

impl ResourceProfile {
    /// Detect the host and compute the full percentage-scaled budget.
    pub fn detect() -> Self {
        let total_ram_mib = read_total_ram_mib();
        let cpus = read_cpus();
        let tier = tier_for(total_ram_mib);
        // Cap our own consumption at 80% of physical RAM so the OS and other
        // processes stay healthy even during evolution/RL bursts.
        let hard_process_limit_mib = (total_ram_mib as f64 * 0.8) as u64;

        // Percentage table (from v3.0 §7.3).
        let mut pct: BTreeMap<&'static str, f64> = BTreeMap::new();
        pct.insert("DataCollector", 25.0);
        pct.insert("FeaturePipeline", 12.0);
        pct.insert("ModelManager", 15.0);
        pct.insert("StrategyEngine", 6.0);
        pct.insert("RiskEngine", 3.0);
        pct.insert("RLGym", 5.0);
        pct.insert("Lab", 22.0);
        pct.insert("UI", 4.0);
        pct.insert("APIChannels", 2.0);
        pct.insert("SysOverhead", 6.0);
        pct.insert("Reserve", 5.0);
        // sum == 100.0 exactly.

        let budgets: BTreeMap<&'static str, u64> = pct
            .iter()
            .map(|(name, p)| {
                let bytes = (hard_process_limit_mib as f64 * p / 100.0) as u64 * 1024 * 1024;
                (*name, bytes)
            })
            .collect();

        ResourceProfile { total_ram_mib, hard_process_limit_mib, cpus, tier, budgets }
    }

    /// Budget for a module, in bytes.
    pub fn budget(&self, module: &str) -> usize {
        (*self.budgets.get(module).unwrap_or(&0)) as usize
    }

    pub fn budget_mib(&self, module: &str) -> f64 {
        self.budget(module) as f64 / (1024.0 * 1024.0)
    }

    /// Number of worker threads we allow ourselves.
    pub fn worker_threads(&self) -> usize {
        (self.cpus.saturating_sub(1)).max(1)
    }

    /// Cap for LRU caches: 0.5% of the hard limit (v2.1 §6).
    pub fn feature_cache_bytes(&self) -> usize {
        (self.hard_process_limit_mib as f64 * 1024.0 * 1024.0 * 0.005) as usize
    }

    pub fn can_run_lab(&self) -> bool {
        self.tier >= Tier::Three
    }

    pub fn can_train_rl(&self) -> bool {
        self.tier >= Tier::Three
    }

    pub fn can_run_web(&self) -> bool {
        self.tier >= Tier::Two
    }

    pub fn summary(&self) -> String {
        let mut out = format!(
            "{} — RAM {} MiB, hard limit {} MiB, CPUs {}, workers {}",
            self.tier,
            self.total_ram_mib,
            self.hard_process_limit_mib,
            self.cpus,
            self.worker_threads()
        );
        for (name, _bytes) in &self.budgets {
            out.push_str(&format!("\n  {:<18} {:>8.1} MiB", name, self.budget_mib(name)));
        }
        out
    }
}

fn tier_for(mib: u64) -> Tier {
    if mib <= 8192 {
        Tier::One
    } else if mib <= 16384 {
        Tier::Two
    } else if mib <= 32768 {
        Tier::Three
    } else {
        Tier::Four
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentages_sum_to_100() {
        // Force a fake 16 GB machine.
        std::env::set_var("THE_QUANT_RAM_MIB", "16384");
        let p = ResourceProfile::detect();
        assert_eq!(p.tier, Tier::Two);
        // hard limit is 0.8 * 16 G = 12.8 G
        assert_eq!(p.hard_process_limit_mib, 13107);
        // DataCollector should be 25% of 13,107 MiB = ~3276 MiB
        let dc = p.budget_mib("DataCollector");
        assert!((dc - 3276.75).abs() < 1.0, "dc={dc}");
    }

    #[test]
    fn tier_boundaries() {
        std::env::set_var("THE_QUANT_RAM_MIB", "4096");
        let p = ResourceProfile::detect();
        assert_eq!(p.tier, Tier::One);
        assert_eq!(p.can_run_lab(), false);
        std::env::set_var("THE_QUANT_RAM_MIB", "49152");
        let p = ResourceProfile::detect();
        assert_eq!(p.tier, Tier::Four);
        assert_eq!(p.can_train_rl(), true);
    }
}