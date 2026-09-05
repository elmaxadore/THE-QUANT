# DERIV INTEGRATION GUIDE

## Overview
Deriv.com API integration has been added to THE QUANT v4.1 Hercules as an optional feature module. This enables trading on Deriv's platform, which offers:

- **Synthetic Indices**: 24/7 trading on simulated markets (Volatility indices, Boom/Crash, etc.)
- **Forex**: Major, minor, and exotic currency pairs
- **Commodities**: Gold, silver, oil, etc.
- **Cryptocurrencies**: BTC, ETH, and other major cryptos
- **Stocks & Indices**: Global stock markets and indices

## Features Added

### 1. Core Module (`src/deriv.rs`)
- WebSocket client for Deriv API v3
- Symbol mapping (Deriv broker symbols → canonical assets)
- Real-time tick and OHLC data streaming
- Order execution (Rise/Fall, Higher/Lower, Touch/NoTouch contracts)
- Account balance monitoring
- Auto-reconnect with exponential backoff
- Rate limiting compliance

### 2. Configuration (`Cargo.toml`)
New optional feature flag:
```toml
[features]
deriv = ["dep:tokio", "dep:tokio-tungstenite", "dep:url"]

[dependencies]
tokio-tungstenite = { version = "0.21", optional = true }
url = { version = "2.5", optional = true }
```

### 3. Module Declaration (`src/main.rs`)
```rust
#[cfg(feature = "deriv")]
mod deriv;
```

## Installation

### Step 1: Register Deriv API App
1. Visit https://api.deriv.com/
2. Create a new application
3. Note your `app_id` and generate an `api_token` if you want trading access

### Step 2: Build with Deriv Feature
```bash
# Build with Deriv support
cargo build --release --features deriv

# Or add to default features in Cargo.toml if always needed
default = ["crypto", "deriv"]
```

### Step 3: Configure Credentials
Add to your `config/system.toml`:

```toml
[deriv]
app_id = 12345  # Your Deriv app ID
api_token = "your_secret_token_here"  # Optional for read-only, required for trading
use_ticks = true  # true for tick stream, false for OHLC candles
granularity = 300  # Candle granularity in seconds (M5 = 300)
symbols = ["R_50", "frxEURUSD", "XAUUSD"]  # Symbols to subscribe to
```

## Usage

### As a Data Feed Source
The Deriv module integrates with the existing feed system similar to `Mt5Feed`:

```rust
use crate::deriv::{DerivClient, DerivFeedConfig, ConnectionState};

let config = DerivFeedConfig {
    app_id: 12345,
    api_token: Some("your_token".into()),
    symbols: vec!["R_50".into(), "frxEURUSD".into()],
    granularity: 300,
    use_ticks: true,
};

let mut client = DerivClient::new(config);

// Connect and subscribe (implementation requires async runtime)
// client.connect().await?;
// client.subscribe_all().await?;

// Process incoming messages
// while let Some(bar) = client.next_bar().await {
//     // Feed bar into the engine
// }
```

### Symbol Mapping
The module includes automatic symbol conversion:

| Deriv Symbol | Canonical Asset |
|--------------|----------------|
| `R_50` | SYNTHETIC_50 |
| `frxEURUSD` | EURUSD |
| `XAUUSD` | XAUUSD |
| `BTCUSD` | BTCUSD |
| `VOL75` | VOLATILITY_75 |
| `BOOM1000` | BOOM_1000 |

Add custom mappings in `src/deriv.rs::deriv_symbol_map()`.

### Contract Types Supported
1. **Rise/Fall**: Vanilla call/put options
2. **Higher/Lower**: Barrier options
3. **Touch/NoTouch**: Binary barrier contracts

## Integration Points

### To Wire Into Engine
To fully integrate Deriv as a live feed source:

1. **Add to `src/simfeed.rs`**: Implement `Feed` trait for `DerivFeed`
2. **Update `src/engine.rs::build_feed()`**: Add "deriv" case
3. **Update `config/system.toml`**: Add `data_source = "deriv"` option
4. **Configure symbols**: Map Deriv symbols to Aegis universe assets

Example engine integration:
```rust
// In src/engine.rs::build_feed()
"deriv" => {
    let symbols: Vec<&str> = cfg.system.symbols.iter().map(|s| s.as_str()).collect();
    let feed = DerivFeed::new(&cfg.deriv, &symbols)
        .map_err(|e| format!("init deriv feed: {e}"))?;
    Ok(Box::new(feed))
}
```

## Testing

Run the included tests:
```bash
cargo test --features deriv deriv::tests
```

Tests cover:
- Symbol mapping
- Rate limiter initialization
- Client creation
- Message building (authorize, subscribe, etc.)

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit your `api_token` to git
- Use environment variables or encrypted vault for tokens
- The `security` module provides AES-256-GCM encryption for credentials
- Start with read-only mode (no token) for testing

Example secure storage:
```bash
export DERIV_API_TOKEN="your_secret_token"
```

Then load in code:
```rust
let token = std::env::var("DERIV_API_TOKEN").ok();
```

## Rate Limits

Deriv API enforces:
- **1 request/second** for most endpoints
- **1000 requests/day** for free tier
- WebSocket streams don't count against limits

The `RateLimiter` struct handles this automatically.

## Troubleshooting

### Connection Issues
- Check firewall allows WebSocket connections (port 443)
- Verify `app_id` is correct (default 1089 is public/test only)
- Ensure network allows `wss://ws.derivws.com`

### Authorization Errors
- Verify `api_token` is valid and not expired
- Check token has required scopes (read/trade)
- Try read-only mode first (omit token, use `balance: 1`)

### Symbol Not Found
- Use correct Deriv symbol format (e.g., `frxEURUSD` not `EURUSD`)
- Check symbol is available in your region
- Verify market is open (synthetic indices are 24/7, forex is not)

## Next Steps

To complete the integration:

1. ✅ **DONE**: Create `src/deriv.rs` module
2. ✅ **DONE**: Add feature flag to `Cargo.toml`
3. ✅ **DONE**: Declare module in `src/main.rs`
4. ⏳ **TODO**: Implement `Feed` trait for async streaming
5. ⏳ **TODO**: Wire into `engine.rs::build_feed()`
6. ⏳ **TODO**: Add Deriv-specific execution backend
7. ⏳ **TODO**: Update config schema with `[deriv]` section
8. ⏳ **TODO**: Add integration tests with mock server

## Resources

- [Deriv API Documentation](https://developers.deriv.com/api)
- [WebSocket API Explorer](https://ws.derivws.com/v3/websockets/docs)
- [Available Markets](https://deriv.com/markets)
- [API Rate Limits](https://developers.deriv.com/api#rate-limit)

## Support

For issues specific to Deriv integration:
1. Check Deriv API status: https://status.deriv.com
2. Review API changelog: https://developers.deriv.com/changelog
3. Contact Deriv support: api@deriv.com

---
**Added in**: THE QUANT v4.1 Hercules  
**Feature Flag**: `--features deriv`  
**Status**: Module created, awaiting full engine wiring
