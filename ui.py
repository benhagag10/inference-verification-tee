"""Self-contained HTML UI for the inference verification API."""


def get_ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TEE Inference Verification</title>
<style>
  :root {
    --safe: #22c55e; --suspicious: #f59e0b; --dangerous: #ef4444;
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 960px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); font-size: 0.875rem; margin-bottom: 1.5rem; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 0.5rem; padding: 1.25rem; margin-bottom: 1rem; }
  .card h2 { font-size: 1rem; margin-bottom: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
  .config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.5rem; }
  .config-item { font-size: 0.875rem; }
  .config-item .label { color: var(--muted); }
  .config-item .value { font-weight: 600; }
  .form-row { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: end; margin-bottom: 0.75rem; }
  .form-group { display: flex; flex-direction: column; gap: 0.25rem; }
  .form-group label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .form-group input { background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 0.5rem 0.75rem; border-radius: 0.375rem; font-size: 0.875rem; width: 120px; }
  .form-group input:focus { outline: none; border-color: #3b82f6; }
  .toggle-advanced { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 0.8rem; padding: 0.25rem 0; }
  .toggle-advanced:hover { color: var(--text); }
  .advanced { display: none; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border); }
  .advanced.show { display: block; }
  button.primary { background: #3b82f6; color: white; border: none; padding: 0.5rem 1.25rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.875rem; font-weight: 600; }
  button.primary:hover { background: #2563eb; }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .error-banner { background: #7f1d1d; border: 1px solid var(--dangerous); border-radius: 0.375rem; padding: 0.75rem 1rem; margin-bottom: 1rem; display: none; font-size: 0.875rem; }
  .loading { display: none; text-align: center; padding: 2rem; }
  .loading.show { display: block; }
  .spinner { width: 2rem; height: 2rem; border: 3px solid var(--border); border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 0.75rem; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text { color: var(--muted); font-size: 0.875rem; }
  .results { display: none; }
  .results.show { display: block; }
  .summary { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .stat { flex: 1; min-width: 120px; text-align: center; padding: 1rem; border-radius: 0.375rem; background: var(--bg); }
  .stat .count { font-size: 1.75rem; font-weight: 700; }
  .stat .pct { font-size: 0.8rem; color: var(--muted); }
  .stat .stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; margin-top: 0.25rem; }
  .stat.safe .count { color: var(--safe); }
  .stat.suspicious .count { color: var(--suspicious); }
  .stat.dangerous .count { color: var(--dangerous); }
  .meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 1rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { text-align: left; padding: 0.5rem; border-bottom: 2px solid var(--border); color: var(--muted); text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05em; }
  td { padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--border); }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; }
  .badge.safe { background: #14532d; color: var(--safe); }
  .badge.suspicious { background: #78350f; color: var(--suspicious); }
  .badge.dangerous { background: #7f1d1d; color: var(--dangerous); }
  .table-wrap { max-height: 400px; overflow-y: auto; border-radius: 0.375rem; }
</style>
</head>
<body>
<div class="container">
  <h1>TEE Inference Verification</h1>
  <p class="subtitle">Verify LLM outputs for model weight exfiltration detection</p>

  <div class="card" id="config-card">
    <h2>Configuration</h2>
    <div class="config-grid" id="config-grid">
      <div class="config-item"><span class="label">Loading...</span></div>
    </div>
  </div>

  <div class="card">
    <h2>Run Verification</h2>
    <div class="form-row">
      <div class="form-group">
        <label>Prompts</label>
        <input type="number" id="n_prompts" value="5" min="1" max="100">
      </div>
      <div class="form-group">
        <label>Max Tokens</label>
        <input type="number" id="max_tokens" value="50" min="1" max="500">
      </div>
      <button class="primary" id="run-btn" onclick="runVerify()">Run Verification</button>
    </div>
    <button class="toggle-advanced" onclick="toggleAdvanced()">&#9660; Advanced Options</button>
    <div class="advanced" id="advanced">
      <div class="form-row">
        <div class="form-group">
          <label>Temperature</label>
          <input type="number" id="temperature" value="1.0" step="0.1" min="0">
        </div>
        <div class="form-group">
          <label>Top K</label>
          <input type="number" id="top_k" value="50" min="1">
        </div>
        <div class="form-group">
          <label>Top P</label>
          <input type="number" id="top_p" value="0.95" step="0.05" min="0" max="1">
        </div>
        <div class="form-group">
          <label>Seed</label>
          <input type="number" id="seed" value="42">
        </div>
        <div class="form-group">
          <label>GLS Threshold</label>
          <input type="number" id="gls_threshold" value="-5.0" step="0.5">
        </div>
        <div class="form-group">
          <label>Rank Threshold</label>
          <input type="number" id="logit_rank_threshold" value="10" min="1">
        </div>
      </div>
    </div>
  </div>

  <div class="error-banner" id="error"></div>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <div class="loading-text">Running verification &mdash; this may take a few minutes...</div>
  </div>

  <div class="results" id="results">
    <div class="card">
      <h2>Results</h2>
      <div class="meta" id="results-meta"></div>
      <div class="summary" id="summary"></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>GLS Score</th><th>Logit Rank</th><th>Classification</th></tr></thead>
          <tbody id="tokens-body"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
async function loadConfig() {
  try {
    const res = await fetch('/config');
    const cfg = await res.json();
    const grid = document.getElementById('config-grid');
    grid.innerHTML = Object.entries(cfg).map(([k, v]) =>
      `<div class="config-item"><span class="label">${k.replace(/_/g, ' ')}:</span> <span class="value">${v}</span></div>`
    ).join('');
    if (cfg.seed !== undefined) document.getElementById('seed').value = cfg.seed;
    if (cfg.gls_threshold !== undefined) document.getElementById('gls_threshold').value = cfg.gls_threshold;
    if (cfg.logit_rank_threshold !== undefined) document.getElementById('logit_rank_threshold').value = cfg.logit_rank_threshold;
  } catch (e) {
    document.getElementById('config-grid').innerHTML = '<div class="config-item"><span class="label">Failed to load config</span></div>';
  }
}

function toggleAdvanced() {
  document.getElementById('advanced').classList.toggle('show');
}

function showError(msg) {
  const el = document.getElementById('error');
  el.textContent = msg;
  el.style.display = 'block';
}

async function runVerify() {
  const btn = document.getElementById('run-btn');
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const errorEl = document.getElementById('error');

  btn.disabled = true;
  errorEl.style.display = 'none';
  results.classList.remove('show');
  loading.classList.add('show');

  const body = {
    n_prompts: parseInt(document.getElementById('n_prompts').value),
    max_tokens: parseInt(document.getElementById('max_tokens').value),
    config: {
      temperature: parseFloat(document.getElementById('temperature').value),
      top_k: parseInt(document.getElementById('top_k').value),
      top_p: parseFloat(document.getElementById('top_p').value),
      seed: parseInt(document.getElementById('seed').value),
      gls_threshold: parseFloat(document.getElementById('gls_threshold').value),
      logit_rank_threshold: parseInt(document.getElementById('logit_rank_threshold').value),
    }
  };

  try {
    const res = await fetch('/verify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({detail: res.statusText}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    showError('Verification failed: ' + e.message);
  } finally {
    btn.disabled = false;
    loading.classList.remove('show');
  }
}

function renderResults(data) {
  document.getElementById('results-meta').textContent =
    `Model: ${data.model_name} | Seed: ${data.seed} | Prompts: ${data.n_prompts} | Total tokens: ${data.total_tokens} | GLS threshold: ${data.gls_threshold} | Rank threshold: ${data.logit_rank_threshold}`;

  document.getElementById('summary').innerHTML = [
    {cls: 'safe', count: data.num_safe, pct: data.safe_pct},
    {cls: 'suspicious', count: data.num_suspicious, pct: data.suspicious_pct},
    {cls: 'dangerous', count: data.num_dangerous, pct: data.dangerous_pct},
  ].map(s => `<div class="stat ${s.cls}"><div class="count">${s.count}</div><div class="pct">${s.pct}%</div><div class="stat-label">${s.cls}</div></div>`).join('');

  document.getElementById('tokens-body').innerHTML = data.tokens.map((t, i) =>
    `<tr><td>${i + 1}</td><td>${t.gls_score != null ? t.gls_score.toFixed(4) : 'N/A'}</td><td>${t.logit_rank}</td><td><span class="badge ${t.classification}">${t.classification}</span></td></tr>`
  ).join('');

  document.getElementById('results').classList.add('show');
}

loadConfig();
</script>
</body>
</html>"""
