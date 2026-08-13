//! # Web UI Dashboard (v3.0 "Prometheus")
//!
//! A lightweight HTML/HTMX dashboard served by the Axum server. It uses
//! Server-Sent Events (SSE) and WebSocket for real-time updates. The UI is
//! intentionally minimal — no heavy JS framework — keeping it fast and simple.
//!
//! The dashboard is served at `/` and the API at `/api/*`.

/// The web UI module's memory budget as a percentage of HARD_PROCESS_LIMIT.
pub const WEB_UI_MEMORY_BUDGET_PCT: f64 = 4.0;

/// HTML template for the dashboard shell.
pub const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Quant — Dashboard</title>
<style>
  :root { --bg:#0f1115; --panel:#1a1d24; --border:#2a2e37; --text:#e6e8eb; --green:#4ade80; --red:#f87171; --yellow:#facc15; --blue:#60a5fa; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; }
  .header { display:flex; align-items:center; justify-content:space-between; padding:14px 24px; background:var(--panel); border-bottom:1px solid var(--border); }
  .logo { font-weight:700; font-size:20px; letter-spacing:1px; }
  .logo span { color:var(--blue); }
  .status-badge { padding:4px 12px; border-radius:12px; font-size:12px; background:var(--green); color:#000; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; padding:20px; }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:16px; }
  .panel h3 { font-size:13px; text-transform:uppercase; letter-spacing:1px; color:#8b93a1; margin-bottom:12px; }
  .metric { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border); font-size:14px; }
  .metric:last-child { border-bottom:none; }
  .metric .label { color:#8b93a1; }
  .metric .value { font-weight:600; }
  .bar { height:8px; background:var(--border); border-radius:4px; overflow:hidden; margin-top:8px; }
  .bar-fill { height:100%; border-radius:4px; transition:width 0.4s; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); }
  th { color:#8b93a1; font-weight:500; text-transform:uppercase; font-size:11px; }
  .pos { color:var(--green); } .neg { color:var(--red); }
  .nav { display:flex; gap:8px; padding:0 24px; padding-top:16px; }
  .nav button { background:var(--panel); border:1px solid var(--border); color:var(--text); padding:8px 16px; border-radius:6px; cursor:pointer; font-size:13px; }
  .nav button.active { background:var(--blue); color:#000; border-color:var(--blue); }
  .conn { display:flex; align-items:center; gap:6px; font-size:12px; color:#8b93a1; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green); }
</style>
<script>window.addEventListener('DOMContentLoaded', () => {
  const es = new EventSource('/api/stream');
  es.onmessage = (e) => { try { const d = JSON.parse(e.data); window.dispatchEvent(new CustomEvent('quant:'+d.type, {detail:d.data})); } catch(_){} };
});</script>
</head>
<body>
  <div class="header">
    <div class="logo">THE <span>QUANT</span> v3.0</div>
    <div class="conn"><span class="dot"></span> Live</div>
    <div class="status-badge">● Trading</div>
  </div>
  <div class="nav">
    <button class="active" data-view="overview">Overview</button>
    <button data-view="accounts">Accounts</button>
    <button data-view="positions">Positions</button>
    <button data-view="regimes">Regimes</button>
    <button data-view="models">Models</button>
    <button data-view="trades">Trades</button>
  </div>
  <div class="grid" id="app">
    <div class="panel"><h3>System Status</h3>
      <div class="metric"><span class="label">CPU</span><span class="value" id="cpu">0%</span></div>
      <div class="metric"><span class="label">RAM</span><span class="value" id="ram">0%</span></div>
      <div class="metric"><span class="label">Uptime</span><span class="value" id="uptime">0s</span></div>
      <div class="metric"><span class="label">MT5</span><span class="value" id="mt5">—</span></div>
      <div class="metric"><span class="label">DB</span><span class="value" id="db">—</span></div>
      <div class="bar"><div class="bar-fill" id="rambar" style="width:0%;background:var(--green)"></div></div>
    </div>
    <div class="panel"><h3>Accounts</h3>
      <table><thead><tr><th>Name</th><th>Equity</th><th>DD%</th><th>Status</th></tr></thead>
      <tbody id="accounts"></tbody></table>
    </div>
    <div class="panel"><h3>Open Positions</h3>
      <table><thead><tr><th>Symbol</th><th>Dir</th><th>Size</th><th>PnL</th></tr></thead>
      <tbody id="positions"></tbody></table>
    </div>
    <div class="panel"><h3>Market Regimes</h3>
      <div id="regimes"></div>
    </div>
  </div>
<script>
  const fmt = (v) => typeof v === 'number' ? v.toFixed(2) : v;
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'<','>':'>','"':'"'}[c]));
  window.addEventListener('quant:Quote', e => {});
  window.addEventListener('quant:System', e => { document.getElementById('cpu').textContent = fmt(e.cpu_pct)+'%'; document.getElementById('ram').textContent = fmt(e.memory_pct)+'%'; const r=document.getElementById('rambar'); r.style.width=Math.min(e.memory_pct,100)+'%'; r.style.background = e.memory_pct>90?'var(--red)':e.memory_pct>75?'var(--yellow)':'var(--green)'; });
  window.addEventListener('quant:Account', e => { const t=document.getElementById('accounts'); t.innerHTML = `<tr><td>${esc(e.account_id)}</td><td>${fmt(e.equity)}</td><td class="${e.drawdown_pct>5?'neg':'pos'}">${fmt(e.drawdown_pct)}%</td><td>Active</td></tr>`; });
  window.addEventListener('quant:Position', e => { const t=document.getElementById('positions'); t.innerHTML = `<tr><td>${esc(e.symbol)}</td><td>—</td><td>—</td><td class="${e.pnl>=0?'pos':'neg'}">${fmt(e.pnl)}</td></tr>`; });
  window.addEventListener('quant:Regime', e => { const r=document.getElementById('regimes'); r.innerHTML = `<div class="metric"><span class="label">${esc(e.symbol)}</span><span class="value">${esc(e.regime)} (${fmt(e.probability*100)}%)</span></div>`; });
</script>
</body>
</html>"#;

/// Returns the dashboard HTML.
pub fn dashboard_html() -> &'static str {
    DASHBOARD_HTML
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_html() {
        let html = dashboard_html();
        assert!(html.contains("THE QUANT"));
        assert!(html.contains("System Status"));
        assert!(html.contains("/api/stream"));
    }
}
