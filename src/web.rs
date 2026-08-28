//! Local web dashboard — v3.0 hybrid GUI (feature `web`, non-default).
//!
//! A dependency-free-at-runtime module compiled only when `--features web` is
//! used. It exposes a tiny JSON status endpoint that the (optional) Python
//! research server and any monitoring can poll. In a full deployment this is
//! where Axum + SSE would render live equity/regime charts; this lean module
//! keeps the default build small and compiles only when requested.

// placeholder — satisfy the module when the feature is enabled.
pub fn enabled() -> bool {
    true
}