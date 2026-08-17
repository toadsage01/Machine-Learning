"""
report
======

HTML report builder for AutoInsight.

Given a ``DatasetProfile`` (and an optional ``DriftReport``), this module
renders a single self-contained ``.html`` file with:

1. **Overview cards** — rows / cols / missing cells / memory / duplicates.
2. **Per-column distributions** — interactive Plotly histograms for numeric
   columns, bar charts for categoricals, line charts for datetimes.
3. **Missingness matrix** — matplotlib heatmap (binary, encoded as base64 PNG).
4. **Correlation heatmap** — matplotlib Pearson correlation for numeric cols.
5. **Drift report** — bar chart of PSI per column, colour-coded by severity.

All figures are inlined as base64 (matplotlib) or Plotly JSON embedded
behind ``<script>`` tags, so the HTML file is fully self-contained and
emailable.

The Jinja2 template lives inline as a Python string in this file so the
project remains a single package — no separate ``templates/`` directory
to manage at install time.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless rendering before pyplot is imported.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, BaseLoader, select_autoescape

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared import apply_style  # noqa: E402
from dataset import ColumnType  # noqa: E402
from model import (  # noqa: E402
    DatasetProfile,
    DriftReport,
    PSI_NO_DRIFT,
    PSI_MODERATE_DRIFT,
)

# Apply the project-wide theme before any figure is created.
apply_style()


# ---------------------------------------------------------------------------
# Plotly is optional — degrade gracefully if it is missing.
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAVE_PLOTLY = True
except Exception:  # pragma: no cover
    HAVE_PLOTLY = False


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _fig_to_base64_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plotly_to_html_div(fig: "go.Figure", div_id: str) -> str:
    """Embed a Plotly figure as a self-contained <div> with inline JS."""
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )
    return html


def _missingness_matrix(profile: DatasetProfile, max_rows: int = 5000) -> str:
    """Render the missingness matrix as a base64 PNG.

    For large datasets we sample ``max_rows`` rows uniformly — the visual
    pattern of missingness is preserved well below the full row count.
    """
    matrix = profile.missingness_matrix
    n_rows = matrix.shape[0]
    if n_rows > max_rows:
        idx = np.linspace(0, n_rows - 1, max_rows).astype(int)
        matrix = matrix[idx]

    fig, ax = plt.subplots(figsize=(max(8, min(20, matrix.shape[1] * 0.35)), 5))
    # 1 = missing (highlight in amber), 0 = present (light grey)
    ax.imshow(matrix.T, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#f4f4f4", "#D55E00"]), interpolation="nearest")
    ax.set_xlabel(f"Row index{' (sampled)' if n_rows > max_rows else ''}")
    ax.set_ylabel("Column")
    ax.set_title("Missingness Matrix — amber = missing cell", pad=10)
    ax.set_yticks(range(matrix.shape[1]))
    ax.set_yticklabels(list(profile.column_types.keys()), fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    return _fig_to_base64_png(fig)


def _correlation_heatmap(profile: DatasetProfile) -> Optional[str]:
    """Render the numeric correlation matrix as a base64 PNG."""
    if profile.correlation_numeric is None:
        return None
    cols = profile.correlation_numeric["columns"]
    mat = np.array(profile.correlation_numeric["matrix"])
    if len(cols) == 0:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.7), max(5, len(cols) * 0.6)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_title("Pearson Correlation — numeric columns", pad=10)

    # Annotate cells with values (only if there are <= 14 cols).
    if len(cols) <= 14:
        for i in range(len(cols)):
            for j in range(len(cols)):
                value = mat[i, j]
                if pd.isna(value):
                    continue
                color = "white" if abs(value) > 0.55 else "#2b2b2b"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_base64_png(fig)


def _drift_bar_chart(drift: DriftReport) -> str:
    """Render the drift report as a horizontal bar chart of PSI values."""
    cols = drift.columns
    if not cols:
        return ""

    names = [c.name for c in cols]
    psis = [c.psi for c in cols]
    # Colour by severity.
    colours = []
    for c in cols:
        if c.psi_label == "no_drift":
            colours.append("#009E73")
        elif c.psi_label == "moderate_drift":
            colours.append("#E69F00")
        else:
            colours.append("#D55E00")

    fig, ax = plt.subplots(figsize=(max(8, max(len(names) * 0.6, 8)), max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, psis, color=colours)
    ax.axvline(PSI_NO_DRIFT, color="#2b2b2b", linestyle=":", linewidth=0.9, label=f"no-drift ({PSI_NO_DRIFT})")
    ax.axvline(PSI_MODERATE_DRIFT, color="#2b2b2b", linestyle="--", linewidth=0.9, label=f"moderate ({PSI_MODERATE_DRIFT})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("PSI (Population Stability Index)")
    ax.set_title(f"Drift Report — {drift.current_name} vs {drift.reference_name}", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    return _fig_to_base64_png(fig)


def _plotly_numeric_histogram(series_name: str, profile_col, div_id: str, raw_series: pd.Series) -> Optional[str]:
    if not HAVE_PLOTLY:
        return None
    values = pd.to_numeric(raw_series, errors="coerce").dropna()
    if values.empty:
        return None
    fig = go.Figure(data=[go.Histogram(x=values.values, nbinsx=40, marker_color="#0072B2", opacity=0.85)])
    fig.update_layout(
        title=f"Distribution of <b>{series_name}</b> (numeric)",
        xaxis_title=series_name,
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        template="plotly_white",
    )
    return _plotly_to_html_div(fig, div_id)


def _plotly_categorical_bar(series_name: str, profile_col, div_id: str, raw_series: pd.Series, top_n: int = 25) -> Optional[str]:
    if not HAVE_PLOTLY:
        return None
    counts = raw_series.dropna().astype(str).value_counts().head(top_n)
    if counts.empty:
        return None
    fig = go.Figure(data=[go.Bar(x=counts.index.astype(str), y=counts.values, marker_color="#009E73")])
    fig.update_layout(
        title=f"Top-{top_n} categories of <b>{series_name}</b>",
        xaxis_title=series_name,
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=50, b=80),
        height=320,
        template="plotly_white",
        xaxis_tickangle=-30,
    )
    return _plotly_to_html_div(fig, div_id)


def _plotly_datetime(series_name: str, profile_col, div_id: str, raw_series: pd.Series) -> Optional[str]:
    if not HAVE_PLOTLY:
        return None
    ts = pd.to_datetime(raw_series, errors="coerce").dropna()
    if ts.empty:
        return None
    counts = ts.dt.to_period("D").value_counts().sort_index()
    fig = go.Figure(data=[go.Scatter(x=counts.index.astype(str), y=counts.values, mode="lines", line=dict(color="#CC79A7", width=2))])
    fig.update_layout(
        title=f"Daily count of <b>{series_name}</b> (datetime)",
        xaxis_title="Date",
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        template="plotly_white",
    )
    return _plotly_to_html_div(fig, div_id)


# ---------------------------------------------------------------------------
# HTML template (Jinja2, inline)
# ---------------------------------------------------------------------------
_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ profile.name }} — AutoInsight Report</title>
<style>
:root{
  --bg:#fafafa; --panel:#ffffff; --ink:#1a1a1a; --muted:#6b6b6b;
  --border:#e4e4e4; --accent:#0072B2; --warn:#E69F00; --bad:#D55E00; --ok:#009E73;
}
*{box-sizing:border-box}
body{font-family:'Inter','Helvetica Neue',Arial,'Noto Sans SC',sans-serif;
     margin:0;background:var(--bg);color:var(--ink);line-height:1.55}
.container{max-width:1200px;margin:0 auto;padding:32px 24px}
header{padding:24px 0;border-bottom:2px solid var(--ink)}
header h1{margin:0 0 8px 0;font-size:28px;font-weight:700;letter-spacing:-0.01em}
header .meta{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px 18px}
.card .label{font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted)}
.card .value{font-size:24px;font-weight:700;margin-top:4px}
.card.warn .value{color:var(--warn)}
.card.bad .value{color:var(--bad)}
.card.ok .value{color:var(--ok)}
section{margin:36px 0;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px 24px}
section h2{font-size:18px;margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid var(--border)}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;background:#eef2f6;color:#2b2b2b;margin-right:4px}
.tag.numeric{background:#e7f1fb;color:#0072B2}
.tag.categorical{background:#e6f5ec;color:#009E73}
.tag.datetime{background:#fdeee9;color:#D55E00}
.tag.boolean{background:#fbf0e3;color:#b35e00}
.tag.text{background:#efeef6;color:#5b5b8a}
.tag.empty{background:#f1f1f1;color:#888}
.tag.high-card{background:#fdecea;color:#cc2936}
.pill{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.pill.no_drift{background:#e6f5ec;color:#009E73}
.pill.moderate_drift{background:#fbf0e3;color:#E69F00}
.pill.severe_drift{background:#fdeee9;color:#D55E00}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top}
th{font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);font-weight:600;background:#fafafa}
tr:hover td{background:#fafafa}
.code{font-family:'Sarasa Mono SC',Menlo,Consolas,monospace;font-size:12px;background:#f6f6f6;padding:1px 4px;border-radius:3px}
.figure{margin:16px 0}
.figure img{max-width:100%;height:auto;border:1px solid var(--border);border-radius:6px}
.plotly{margin:16px 0}
.kvtable td:first-child{width:38%;color:var(--muted)}
.preview td{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px}
footer{padding:24px 0 40px 0;color:var(--muted);font-size:12px;text-align:center}
.drift-summary{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.drift-summary .pill{font-size:13px;padding:6px 12px}
details summary{cursor:pointer;font-size:13px;color:var(--muted);padding:8px 0}
details[open] summary{color:var(--ink)}
ul.cleanlist{margin:0;padding-left:18px}
ul.cleanlist li{margin:2px 0}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>🪄 AutoInsight Report — <span style="color:var(--accent)">{{ profile.name }}</span></h1>
  <div class="meta">
    Generated {{ generated_at }} &middot; {{ profile.n_rows|fmt_int }} rows &middot; {{ profile.n_cols }} columns &middot;
    {{ profile.memory_mb }} MB &middot; {{ profile.missing_pct }}% cells missing
  </div>
</header>

<!-- Overview cards -->
<div class="cards">
  <div class="card"><div class="label">Rows</div><div class="value">{{ profile.n_rows|fmt_int }}</div></div>
  <div class="card"><div class="label">Columns</div><div class="value">{{ profile.n_cols }}</div></div>
  <div class="card {{ 'warn' if profile.missing_pct > 5 else '' }}"><div class="label">Missing cells</div><div class="value">{{ profile.missing_pct }}%</div></div>
  <div class="card {{ 'warn' if profile.duplicate_pct > 1 else '' }}"><div class="label">Duplicate rows</div><div class="value">{{ profile.duplicate_pct }}%</div></div>
  <div class="card"><div class="label">Memory</div><div class="value">{{ profile.memory_mb }} MB</div></div>
  <div class="card"><div class="label">SHA-256</div><div class="value" style="font-size:13px;font-family:monospace">{{ sha_short }}</div></div>
</div>

<!-- Missingness matrix -->
<section>
  <h2>Missingness Map</h2>
  <p style="color:var(--muted);font-size:13px;margin:0 0 8px 0">
    Each cell is a row × column pair. Amber = missing, grey = present.
    The matrix is sampled down to 5,000 rows for large datasets to keep the chart legible.
  </p>
  <div class="figure"><img alt="missingness matrix" src="data:image/png;base64,{{ missingness_png }}"></div>
</section>

<!-- Correlation heatmap -->
{% if correlation_png %}
<section>
  <h2>Numeric Correlation</h2>
  <p style="color:var(--muted);font-size:13px;margin:0 0 8px 0">
    Pearson correlation between numeric columns. Use this to spot redundant
    features (|r| > 0.95) or potential leakage channels.
  </p>
  <div class="figure"><img alt="correlation heatmap" src="data:image/png;base64,{{ correlation_png }}"></div>
</section>
{% endif %}

<!-- Drift report -->
{% if drift %}
<section>
  <h2>Drift Report</h2>
  <p style="color:var(--muted);font-size:13px;margin:0 0 12px 0">
    Comparing <b>{{ drift.current_name }}</b> against reference <b>{{ drift.reference_name }}</b> using
    Population Stability Index (PSI). Thresholds:
    <span class="pill no_drift">no drift &lt; 0.10</span>
    <span class="pill moderate_drift">moderate 0.10–0.25</span>
    <span class="pill severe_drift">severe &gt; 0.25</span>
  </p>
  <div class="drift-summary">
    <span class="pill no_drift">{{ drift.n_no_drift }} no drift</span>
    <span class="pill moderate_drift">{{ drift.n_moderate }} moderate</span>
    <span class="pill severe_drift">{{ drift.n_severe }} severe</span>
    <span class="pill" style="background:#eef2f6">max PSI = {{ drift.max_psi }}</span>
  </div>
  <div class="figure"><img alt="drift bar chart" src="data:image/png;base64,{{ drift_png }}"></div>

  <table>
    <thead><tr><th>Column</th><th>Type</th><th>PSI</th><th>Verdict</th><th>KS stat</th><th>KS p-value</th><th>n (current)</th><th>n (reference)</th></tr></thead>
    <tbody>
      {% for c in drift.columns %}
      <tr>
        <td class="code">{{ c.name }}</td>
        <td><span class="tag {{ c.type }}">{{ c.type }}</span></td>
        <td>{{ c.psi }}</td>
        <td><span class="pill {{ c.psi_label }}">{{ c.psi_label|replace('_',' ') }}</span></td>
        <td>{{ c.ks_stat if c.ks_stat is not none else '—' }}</td>
        <td>{{ c.ks_pvalue if c.ks_pvalue is not none else '—' }}</td>
        <td>{{ c.n_current|fmt_int }}</td>
        <td>{{ c.n_reference|fmt_int }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</section>
{% endif %}

<!-- Per-column profile table -->
<section>
  <h2>Column Profiles</h2>
  <table>
    <thead>
      <tr>
        <th>Column</th><th>Type</th><th>Missing</th><th>Unique</th>
        <th>Mean / Median</th><th>Std</th><th>p05 / p95</th><th>Skew / Kurt</th>
        <th>Top category (count)</th>
      </tr>
    </thead>
    <tbody>
      {% for c in profile.columns %}
      <tr>
        <td class="code">{{ c.name }}{% if c.cardinality_flag == 'high' %} <span class="tag high-card">high-card</span>{% endif %}</td>
        <td><span class="tag {{ c.type }}">{{ c.type }}</span></td>
        <td>{{ c.missing_pct }}%</td>
        <td>{{ c.n_unique|fmt_int }} ({{ c.unique_pct }}%)</td>
        <td>
          {% if c.mean is not none %}{{ c.mean|fmt_float }} / {{ c.median|fmt_float }}{% else %}—{% endif %}
        </td>
        <td>{% if c.std is not none %}{{ c.std|fmt_float }}{% else %}—{% endif %}</td>
        <td>{% if c.p05 is not none %}{{ c.p05|fmt_float }} / {{ c.p95|fmt_float }}{% else %}—{% endif %}</td>
        <td>{% if c.skew is not none %}{{ c.skew|fmt_float }} / {{ c.kurtosis|fmt_float }}{% else %}—{% endif %}</td>
        <td>
          {% if c.top_categories %}
            {% for label, count in c.top_categories[:3] %}
              <span class="code">{{ label }}</span> <span style="color:var(--muted)">({{ count|fmt_int }})</span>{% if not loop.last %}, {% endif %}
            {% endfor %}
          {% else %}—{% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</section>

<!-- Per-column distribution figures -->
<section>
  <h2>Per-column Distributions</h2>
  <p style="color:var(--muted);font-size:13px;margin:0 0 8px 0">
    Interactive Plotly charts. Click legend entries to toggle series, hover for values.
  </p>
  {% for fig_html in distribution_figs %}
    <div class="plotly">{{ fig_html|safe }}</div>
  {% else %}
    <p style="color:var(--muted)">No numeric/categorical/datetime columns to plot.</p>
  {% endfor %}
</section>

<!-- Sample rows -->
{% if profile.sample_rows %}
<section>
  <h2>Sample Rows (first 5)</h2>
  <details>
    <summary>Show first 5 rows</summary>
    <div style="overflow-x:auto">
      <table class="preview">
        <thead><tr>{% for col in profile.sample_rows[0].keys() %}<th>{{ col }}</th>{% endfor %}</tr></thead>
        <tbody>
          {% for row in profile.sample_rows %}
          <tr>{% for col in profile.sample_rows[0].keys() %}<td>{{ row[col] }}</td>{% endfor %}</tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </details>
</section>
{% endif %}

<footer>
  AutoInsight P1 &middot; generated {{ generated_at }} &middot;
  {{ profile.n_rows|fmt_int }} × {{ profile.n_cols }} dataset &middot;
  provenance SHA-256 <span class="code">{{ sha256 }}</span>
</footer>

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Jinja2 environment with custom filters
# ---------------------------------------------------------------------------
def _make_env() -> Environment:
    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
    env.filters["fmt_int"] = lambda v: f"{int(v):,}"
    env.filters["fmt_float"] = lambda v: (f"{v:.4f}" if isinstance(v, (int, float)) and not pd.isna(v) else "—")
    return env


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class HTMLReportBuilder:
    """Render a ``DatasetProfile`` (and optional ``DriftReport``) to HTML."""

    MAX_DISTRIBUTION_FIGS = 50  # cap to keep HTML small for wide datasets

    def __init__(self, profile: DatasetProfile, raw_df: pd.DataFrame,
                 drift: Optional[DriftReport] = None, sha256: str = ""):
        self.profile = profile
        self.raw_df = raw_df
        self.drift = drift
        self.sha256 = sha256 or "—"
        self.env = _make_env()

    def _build_distribution_figs(self) -> List[str]:
        """Return list of Plotly HTML <div> snippets for the most interesting columns."""
        figs: List[str] = []
        n_built = 0
        for col_profile in self.profile.columns:
            if n_built >= self.MAX_DISTRIBUTION_FIGS:
                break
            name = col_profile.name
            if name not in self.raw_df.columns:
                continue
            raw = self.raw_df[name]
            div_id = f"plot_{name}_{n_built}"
            html: Optional[str] = None
            if col_profile.type == ColumnType.NUMERIC.value:
                html = _plotly_numeric_histogram(name, col_profile, div_id, raw)
            elif col_profile.type in (ColumnType.CATEGORICAL.value, ColumnType.BOOLEAN.value):
                html = _plotly_categorical_bar(name, col_profile, div_id, raw)
            elif col_profile.type == ColumnType.DATETIME.value:
                html = _plotly_datetime(name, col_profile, div_id, raw)
            if html:
                figs.append(html)
                n_built += 1
        return figs

    def render(self) -> str:
        missingness_png = _missingness_matrix(self.profile)
        correlation_png = _correlation_heatmap(self.profile)
        distribution_figs = self._build_distribution_figs()
        drift_png = _drift_bar_chart(self.drift) if self.drift else None

        template = self.env.from_string(_TEMPLATE)
        html = template.render(
            profile=self.profile,
            drift=self.drift,
            missingness_png=missingness_png,
            correlation_png=correlation_png or "",
            drift_png=drift_png or "",
            distribution_figs=distribution_figs,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            sha256=self.sha256,
            sha_short=self.sha256[:16] if self.sha256 else "—",
        )
        return html

    def save(self, output_path: Path) -> Path:
        """Render and write to ``output_path``. Returns the resolved path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = self.render()
        output_path.write_text(html, encoding="utf-8")
        return output_path


__all__ = ["HTMLReportBuilder"]
