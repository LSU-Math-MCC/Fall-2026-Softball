# pip install dash pandas plotly lightgbm scipy
# Run the model first so grading script first -> pitch_grade_output/ exists and then: python3 this file
# dont forget the arsenal stuff features 
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

OUT_DIR = "../pitch_grade_output after location"
MIN_PITCHES_FOR_PERCENTILES = 50      # pitchers in the percentile comparison pool

# --- Color Palette ---

PRIMARY       = "#461D7C"   # LSU purple: header, tabs, table headers, titles, Stuff+
PRIMARY_LIGHT = "#653C9C"   # lighter purple: header gradient, sliders, dropdown accents
ACCENT        = "#FDD023"   # LSU gold: CSW+, KPI numbers, selected pitchers, tab underline

PAGE_BG   = "#f3f0f8"       # page background
PANEL_BG  = "#ffffff"       # white panels / search box / tiles
CARD_BG   = "#f8f6fb"       # small cards (profile, percentile, distribution)
BORDER    = "#e5e1ee"       # card and tile borders
GRID      = "#e5e7eb"       # chart gridlines

TEXT        = "#1f2937"     # normal text
TEXT_SOFT   = "#374151"     # subplot titles
MUTED       = "#6b7280"     # labels, small text
ON_PRIMARY  = "#ffffff"     # text on purple (header, table header, selected tab)
CHART_TEXT  = PRIMARY       # chart font color
CHART_TITLE = PRIMARY       # chart titles

TAB_BG   = "#e6e0f0"        # unselected tab
TAB_TEXT = "#3b2a5a"        # unselected tab text

CHIP_BG     = "#ece4f7"
CHIP_BORDER = "#c9b6e6"

ROW_ALT      = "#f8f6fb"    # zebra stripe
SELECTED_ROW = "#fdf1b8"    # row of a pitcher picked in the search box (light gold)

NEUTRAL      = "#cbd5e1"    # "other" dots on the leaderboard scatter
REF_LINE     = "#9ca3af"    # dotted line at 100
OUTLINE      = "black"      # outline on selected pitchers (all charts)
SELECTED_DOT = ACCENT       # selected pitchers on the leaderboard scatter

STUFF_COLOR = PRIMARY
CSW_COLOR   = ACCENT

GRADE_LOW   = "#dc2626"
GRADE_MID   = "#ffea48"
GRADE_HIGH  = "#5ccf23"
GRADE_MIN = 85    # this grade or lower = fully GRADE_LOW (red)
GRADE_MAX = 110   # this grade or higher = fully GRADE_HIGH (purple)

TILE_GREAT = PRIMARY
TILE_GOOD  = PRIMARY_LIGHT
TILE_AVG   = MUTED
TILE_POOR  = GRADE_LOW

HEADER_FROM = PRIMARY
HEADER_TO   = PRIMARY_LIGHT
HEADER_STRIPE = ACCENT      # gold line under the header


FILL_ALPHA = 0.7        # fill opacity (1 = solid)
BORDER_W = 2           # border width on non-selected bars
SELECTED_W = 2        # border width on selected bars

PITCH_COLORS = {
    "fastball": "#d62839", "fourseam": "#d62839", "fourseamfastball": "#d62839",
    "sinker": "#f77f00", "twoseam": "#f77f00",
    "cutter": "#DA70D6",
    "slider": "#eddb62",
    "curveball": "#1764c8",
    "changeup": "#2e9e48",
    "splitter": "#81e3ee",
    "sweeper": "#6f4f28",
    "knuckleball": "#8f44cd",
}
PITCH_FALLBACK = px.colors.qualitative.Plotly   # for pitch names not listed above


FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

NICE = {"stuff_plus": "Stuff+", "csw_plus": "CSW+"}
METRIC_COLORS = {"Stuff+": STUFF_COLOR, "CSW+": CSW_COLOR}
GRADE_SCALE = [[0.0, GRADE_LOW], [0.5, GRADE_MID], [1.0, GRADE_HIGH]]
LABELS = {"stuff_plus": "Stuff+", "csw_plus": "CSW+", "pitches": "Pitches",
          "pitch_types": "Pitch types", "CorrectedPitchType": "Pitch type",
          "grade": "Grade", "metric": ""}
METRIC_OPTIONS = [{"label": "Stuff+", "value": "stuff_plus"},
                  {"label": "CSW+", "value": "csw_plus"}]

# ---------- Load outputs from script (once, at startup) ----------
lb = pd.read_csv(os.path.join(OUT_DIR, "pitcher_leaderboard.csv"))
grades = pd.read_csv(os.path.join(OUT_DIR, "pitcher_pitch_grades.csv"))
scored_full = pd.read_csv(os.path.join(OUT_DIR, "scored_pitches.csv"))
scored = scored_full.sample(60_000, random_state=0) if len(scored_full) > 60_000 else scored_full

pitchers = sorted(grades["Pitcher"].unique())

TYPES = sorted(grades["CorrectedPitchType"].dropna().unique())
CMAP = {}
for i, t in enumerate(TYPES):
    key = str(t).lower().replace(" ", "").replace("-", "")
    CMAP[t] = PITCH_COLORS.get(key, PITCH_FALLBACK[i % len(PITCH_FALLBACK)])

# Pitcher-level stats (all pitches) used for the percentile rankings
pstats = (scored_full.groupby("Pitcher")
          .agg(pitches=("stuff_plus", "size"),
               stuff_plus=("stuff_plus", "mean"),
               csw_plus=("csw_plus", "mean"),
               csw_pct=("csw", "mean"),
               run_value=("run_value", "mean"))
          .reset_index())
pool = pstats[pstats["pitches"] >= MIN_PITCHES_FOR_PERCENTILES]

# (label, column, higher_is_better, formatter)
PCT_METRICS = [
    ("Stuff+", "stuff_plus", True, lambda v: f"{v:.1f}"),
    ("CSW+", "csw_plus", True, lambda v: f"{v:.1f}"),
    ("CSW%", "csw_pct", True, lambda v: f"{v:.1%}"),
    ("xRV / pitch", "run_value", False, lambda v: f"{v:.3f}"),
]


# ---------- Style helpers ----------
def style_fig(fig, height=None):
    """Apply the shared look to any figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=13, color=CHART_TEXT),
        title_font=dict(size=17, color=CHART_TITLE),
        title=dict(y=0.97, yanchor="top"),          # NEW: pin title to the top
        margin=dict(t=90),                          # NEW: more room above the plot
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    if height:
        fig.update_layout(height=height)
    return fig


def ref_h(fig, **kw):
    fig.add_hline(y=100, line_dash="dot", line_color=REF_LINE, **kw)


def ref_v(fig, **kw):
    fig.add_vline(x=100, line_dash="dot", line_color=REF_LINE, **kw)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def scale_color(t):
    t = max(0.0, min(1.0, t))
    for (p0, c0), (p1, c1) in zip(GRADE_SCALE, GRADE_SCALE[1:]):
        if t <= p1:
            f = (t - p0) / (p1 - p0)
            a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return "#%02x%02x%02x" % tuple(round(x + (y - x) * f) for x, y in zip(a, b))
    return GRADE_SCALE[-1][1]


def grade_color(v):
    return scale_color((v - GRADE_MIN) / (GRADE_MAX  - GRADE_MIN))

def tile(value, label):
    return html.Div([
        html.Div(f"{value:.1f}", className="tile-num", style={"color": grade_color(value)}),
        html.Div(label, className="tile-label"),
    ], className="tile")


def make_tab(label, value, children):
    return dcc.Tab(label=label, value=value, children=html.Div(children, className="panel"),
                   className="custom-tab", selected_className="custom-tab--selected")


ZEBRA = [{"if": {"row_index": "odd"}, "backgroundColor": ROW_ALT}]
TABLE_STYLE = dict(
    style_as_list_view=True,
    style_header={"backgroundColor": PRIMARY, "color": ON_PRIMARY, "fontWeight": "600",
                  "border": "none", "textAlign": "left", "fontFamily": FONT},
    style_cell={"fontFamily": FONT, "padding": "9px 14px", "border": "none",
                "textAlign": "left", "fontSize": "14px"},
    style_data_conditional=ZEBRA,
)

def with_alpha(hex_color, alpha):
    """'#d62839' -> 'rgba(214,40,57,0.7)'"""
    h = hex_color.lstrip("#")
    r, gr, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{gr},{b},{alpha})"

def add_type_legend(fig, types):
    """Add a legend for the pitch type colors to the bottom of the page."""
    fig.update_traces(showlegend=False)
    for t in TYPES:
        if t in types:
            fig.add_trace(go.Scatter(x=[None], y=[None], marker=dict(size=10, color=CMAP[t]), 
                                     mode="markers", name=t, legendgroup=t, showlegend=True, 
                                     hoverinfo="skip",))




# ---------- CSS: palette -> CSS variables, so the stylesheet never hardcodes colors ----------
CSS_VARS = {
    "primary": PRIMARY, "primary-light": PRIMARY_LIGHT, "accent": ACCENT,
    "page-bg": PAGE_BG, "panel-bg": PANEL_BG, "card-bg": CARD_BG, "border": BORDER,
    "text": TEXT, "muted": MUTED, "on-primary": ON_PRIMARY,
    "tab-bg": TAB_BG, "tab-text": TAB_TEXT,
    "chip-bg": CHIP_BG, "chip-border": CHIP_BORDER,
    "header-from": HEADER_FROM, "header-to": HEADER_TO, "header-stripe": HEADER_STRIPE,
}
ROOT_CSS = ":root{" + "".join(f"--{k}:{v};" for k, v in CSS_VARS.items()) + "}"

INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  /*ROOT_VARS*/
  body { margin: 0; background: var(--page-bg); font-family: Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: var(--text); }
  .container { max-width: 1300px; margin: 0 auto; padding: 20px 20px 40px; }
  .header { background: linear-gradient(120deg, var(--header-from) 0%, var(--header-to) 100%); color: var(--on-primary); border-radius: 16px;
            border-bottom: 5px solid var(--header-stripe);
            padding: 22px 28px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; }
  .header h1 { margin: 0; font-size: 28px; letter-spacing: -0.5px; }
  .header p { margin: 4px 0 0; opacity: 0.85; }
  .kpis { display: flex; gap: 12px; }
  .kpi { background: rgba(255,255,255,0.14); border-radius: 12px; padding: 10px 18px; text-align: center; }
  .kpi b { display: block; font-size: 22px; color: var(--accent); }
  .kpi span { font-size: 12px; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.6px; }
  .search { background: var(--panel-bg); border-radius: 14px; padding: 14px 18px; margin: 16px 0; box-shadow: 0 1px 3px rgba(16,24,40,.08); }
  .ctrl-label { display: block; font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }
  .panel { background: var(--panel-bg); border-radius: 14px; padding: 20px; box-shadow: 0 1px 3px rgba(16,24,40,.08); margin-top: 6px; }
  .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; padding: 14px; }
  .row { display: flex; gap: 14px; margin-top: 14px; flex-wrap: wrap; }
  .tiles { display: flex; gap: 10px; margin-top: 14px; }
  .tile { flex: 1; background: var(--panel-bg); border: 1px solid var(--border); border-radius: 12px; padding: 10px; text-align: center; }
  .tile-num { font-size: 30px; font-weight: 700; line-height: 1.1; }
  .tile-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .section-title { margin: 22px 0 8px; font-size: 17px; color: var(--primary); }
  .custom-tabs-container { border: none !important; }
  .custom-tab { border: none !important; border-radius: 10px !important; padding: 10px 18px !important; margin-right: 6px !important;
                background: var(--tab-bg) !important; color: var(--tab-text) !important; font-weight: 600 !important; font-family: inherit !important; }
  .custom-tab--selected { background: var(--primary) !important; color: var(--on-primary) !important; box-shadow: inset 0 -3px 0 var(--accent); }
  .muted { color: var(--muted); }
  /* ---- Recolor Dash's built-in controls (sliders, dropdowns, radios) ---- */
  :root {
    --Dash-Fill-Interactive-Strong: var(--primary-light) !important;
    --Dash-Fill-Interactive-Weak: var(--chip-bg) !important;
    --Dash-Fill-Primary-Active: var(--chip-bg) !important;
    --Dash-Fill-Primary-Hover: var(--card-bg) !important;
    --Dash-Stroke-Strong: var(--primary-light) !important;
  }
  input[type="radio"], input[type="checkbox"] { accent-color: var(--primary-light); }
  .rc-slider-track, .dash-slider-range { background-color: var(--primary-light) !important; }
  .rc-slider-handle, .dash-slider-thumb { border-color: var(--primary-light) !important; }
  .rc-slider-dot-active { border-color: var(--primary-light) !important; }
  .Select--multi .Select-value { background-color: var(--chip-bg) !important; border-color: var(--chip-border) !important; color: var(--primary) !important; }
  .Select--multi .Select-value-icon { border-color: var(--chip-border) !important; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

app = Dash(__name__)
app.title = "Pitch Grades"
app.index_string = INDEX_TEMPLATE.replace("/*ROOT_VARS*/", ROOT_CSS)

# ---------- Tabs ----------
tabs = [
    make_tab("Leaderboard", "lb", [
        html.Label("Minimum pitches (selected pitchers are always shown)", className="ctrl-label"),
        dcc.Slider(0, int(lb["pitches"].max()), 25, value=100, id="min-pitches",
                   tooltip={"placement": "bottom"}),
        dcc.Graph(id="lb-scatter"),
        dash_table.DataTable(
            id="lb-table", sort_action="native", page_size=25,
            columns=[{"name": LABELS.get(c, c), "id": c} for c in lb.columns],
            **TABLE_STYLE,
        ),
    ]),

    make_tab("Top pitchers", "top", [
        html.Div([
            html.Div([
                html.Label("Metric", className="ctrl-label"),
                dcc.RadioItems(METRIC_OPTIONS, "stuff_plus", id="top-metric", inline=True),
            ], style={"flex": "1", "minWidth": "180px"}),
            html.Div([
                html.Label("Min pitches", className="ctrl-label"),
                dcc.Input(id="top-min", type="number", value=50, min=0),
            ], style={"flex": "1", "minWidth": "120px"}),
            html.Div([
                html.Label("Pitch types (blank = all)", className="ctrl-label"),
                dcc.Dropdown(TYPES, [], multi=True, id="top-types"),
            ], style={"flex": "3", "minWidth": "260px"}),
            html.Div([
                dcc.Checklist(["Zoom axis"], [], id="top-zoom"),
            ], style={"flex": "1", "minWidth": "100px"}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end", "flexWrap": "wrap"}),
        html.Label("How many to show", className="ctrl-label", style={"marginTop": "14px"}),
        dcc.Slider(5, 30, 1, value=15, id="top-n", tooltip={"placement": "bottom"}),
        dcc.Graph(id="top-graph"),
    ]),

    make_tab("Pitcher", "pitcher", [
        html.Label("Pitcher", className="ctrl-label"),
        dcc.Dropdown(pitchers, pitchers[0], id="pitcher-dd", clearable=False),
        html.Div([
            html.Div(id="profile-card", className="card", style={"flex": "1", "minWidth": "240px"}),
            html.Div(dcc.Graph(id="pct-graph"), className="card", style={"flex": "2", "minWidth": "340px"}),
            html.Div(dcc.Graph(id="dist-graph"), className="card", style={"flex": "2", "minWidth": "340px"}),
        ], className="row"),
        dcc.Graph(id="pitcher-bar"),
        html.H4("Arsenal", className="section-title"),
        dash_table.DataTable(
            id="arsenal-table", sort_action="native",
            columns=[
                {"name": "Pitch type", "id": "CorrectedPitchType"},
                {"name": "Pitches", "id": "pitches"},
                {"name": "Stuff+", "id": "stuff_plus"},
                {"name": "CSW+", "id": "csw_plus"},
            ],
            **TABLE_STYLE,
        ),
    ]),

    make_tab("Pitch types", "types", [
        html.Label("Metric", className="ctrl-label"),
        dcc.RadioItems(METRIC_OPTIONS + [{"label": "Both", "value": "Both"}],
                       "Both", id="metric", inline=True),
        dcc.Graph(id="type-box"),
        dcc.Graph(id="scatter"),
    ]),
]



# ---------- Layout ----------
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Pitch Grades"),
            html.P("Stuff+ and CSW+ for every pitcher and pitch type"),
        ]),
        html.Div([
            html.Div([html.B(f"{len(scored_full):,}"), html.Span("pitches")], className="kpi"),
            html.Div([html.B(f"{len(pstats):,}"), html.Span("pitchers")], className="kpi"),
            html.Div([html.B(f"{len(TYPES)}"), html.Span("pitch types")], className="kpi"),
        ], className="kpis"),
    ], className="header"),

    # Search box shared by the Leaderboard, Top pitchers and Pitch types tabs
    html.Div([
        html.Label("Find pitchers", className="ctrl-label"),
        dcc.Dropdown(
            pitchers, [], multi=True, id="pitcher-select",
            placeholder="Type to search and select pitchers...",
        ),
        dcc.RadioItems(["Highlight", "Show only"], "Highlight",
                       id="pitcher-mode", inline=True, style={"marginTop": "8px"}),
    ], className="search"),

    dcc.Tabs(tabs, value="lb", parent_className="custom-tabs",
             className="custom-tabs-container"),
], className="container")



# ---------- Callbacks ----------
@app.callback(
    Output("lb-table", "data"),
    Output("lb-table", "style_data_conditional"),
    Output("lb-scatter", "figure"),
    Input("min-pitches", "value"),
    Input("pitcher-select", "value"),
    Input("pitcher-mode", "value"),
)


def update_leaderboard(min_p, selected, mode):
    d = lb.copy()
    is_sel = d["Pitcher"].isin(selected)
    d["group"] = is_sel.map({True: "Selected", False: "Other"})

    # keep pitchers above the minimum, plus any selected pitcher regardless
    d = d[(d["pitches"] >= min_p) | is_sel]
    if mode == "Show only" and selected:
        d = d[d["group"] == "Selected"]

    d_plot = d.sort_values("group", ascending=True)       # "Other" first, selected drawn on top
    d_plot["label"] = d_plot["Pitcher"].where(d_plot["group"] == "Selected", "")
    fig = px.scatter(
        d_plot, x="stuff_plus", y="csw_plus", color="group", text="label",
        hover_name="Pitcher",
        hover_data={"pitches": True, "pitch_types": True, "stuff_plus": ":.1f",
                    "csw_plus": ":.1f", "group": False, "label": False},
        color_discrete_map={"Selected": SELECTED_DOT, "Other": NEUTRAL},
        labels=LABELS, title="Stuff+ vs CSW+ (arsenal-weighted)",
    )
    fig.update_traces(marker_size=8, textposition="top center")
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color=OUTLINE)),
                      selector=dict(name="Selected"))
    style_fig(fig, height=480)
    fig.update_layout(showlegend=False, hovermode="closest")
    ref_h(fig)
    ref_v(fig)

    d_tab = d.sort_values(["group", "stuff_plus"], ascending=[False, False])
    data = d_tab.drop(columns="group").to_dict("records")
    styles = ZEBRA + [{"if": {"filter_query": "{Pitcher} = " + json.dumps(p)},
                       "backgroundColor": SELECTED_ROW, "fontWeight": "bold"}
                      for p in selected]
    return data, styles, fig




@app.callback(
    Output("top-graph", "figure"),
    Input("top-metric", "value"),
    Input("top-n", "value"),
    Input("top-types", "value"),
    Input("top-min", "value"),
    Input("top-zoom", "value"),
    Input("pitcher-select", "value"),
)


def update_top(metric, n, types_sel, min_p, zoom, selected):
    # 1) The normal top N
    pool_g = grades[grades["pitches"] >= (min_p or 0)]
    if types_sel:
        pool_g = pool_g[pool_g["CorrectedPitchType"].isin(types_sel)]
    top_g = pool_g.nlargest(n, metric)

    # 2) Selected pitchers not already in the top N, added at the bottom
    #    (ignores the min-pitches filter so they always show up)
    extra = grades[grades["Pitcher"].isin(selected)]
    if types_sel:
        extra = extra[extra["CorrectedPitchType"].isin(types_sel)]
    extra = extra[~extra.index.isin(top_g.index)].sort_values(metric, ascending=False)

    g = pd.concat([top_g, extra]).copy()        # top N first, selected last
    g["label"] = g["Pitcher"] + " (" + g["CorrectedPitchType"] + ")"
    g["txt"] = g[metric].map(lambda v: f"{v:.1f}")
    nice = NICE[metric]

    title = f"Top {len(top_g)} pitcher / pitch types by {nice}"
    if len(extra):
        title += f"  (+ {len(extra)} selected)"

    # Colored by pitch type (same colors as every other tab)
    fig = px.bar(
        g, x=metric, y="label", color="CorrectedPitchType", orientation="h",
        text="txt", color_discrete_map=CMAP, labels=LABELS,
        hover_data={"pitches": True, metric: ":.1f", "label": False, "txt": False},
        category_orders={"label": g["label"].tolist()},
        title=title,                      # keep opacity out of here
    )
    fig.update_traces(textposition="outside", cliponaxis=False)

    sel_labels = set(g.loc[g["Pitcher"].isin(selected), "label"])
    for tr in fig.data:                                   # one trace per pitch type
        base = CMAP[tr.name]
        tr.marker.color = with_alpha(base, FILL_ALPHA)    # faded fill
        # non-selected: solid border in the pitch's base color
        # selected: black border
        tr.marker.line.color = [OUTLINE if lab in sel_labels else base for lab in tr.y]
        tr.marker.line.width = [SELECTED_W if lab in sel_labels else BORDER_W for lab in tr.y]
    style_fig(fig, height=max(350, 34 * len(g) + 130))
    fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=g["label"].tolist(),
                   autorange="reversed", title="Pitcher (Pitch Type)"),
        xaxis=dict(title=f"Average {nice}"),
        margin=dict(l=10, r=40),
    )
    if zoom and len(g):
        fig.update_xaxes(range=[g[metric].min() - 3, g[metric].max() + 2])
    elif len(g):
        fig.update_xaxes(range=[0, g[metric].max() * 1.12])
    return fig




@app.callback(
    Output("profile-card", "children"),
    Output("pct-graph", "figure"),
    Output("dist-graph", "figure"),
    Output("pitcher-bar", "figure"),
    Output("arsenal-table", "data"),
    Input("pitcher-dd", "value"),
)


def update_pitcher(p):
    gp = grades[grades["Pitcher"] == p].sort_values("pitches", ascending=False)
    row = pstats[pstats["Pitcher"] == p]

    # ---- Left card ----
    if len(row):
        r = row.iloc[0]
        card = [
            html.H3(p, style={"margin": "0 0 4px", "color": PRIMARY}),
            html.Div(f"{int(r['pitches']):,} pitches  •  {gp['CorrectedPitchType'].nunique()} pitch types",
                     className="muted"),
            html.Div([tile(r["stuff_plus"], "Stuff+"), tile(r["csw_plus"], "CSW+")],
                     className="tiles"),
        ]
    else:
        card = [html.H3(p)]

    # ---- Percentile rankings (vs. pitchers in the pool) ----
    labels, pcts, texts = [], [], []
    if len(row):
        for label, col, higher, fmt in PCT_METRICS:
            v = r[col]
            pct = (pool[col] < v).mean() * 100 if higher else (pool[col] > v).mean() * 100
            labels.append(label)
            pcts.append(round(pct))
            texts.append(f"{round(pct)}  ({fmt(v)})")
    pct_fig = go.Figure(go.Bar(
        x=pcts, y=labels, orientation="h", text=texts, textposition="outside",
        marker=dict(color=pcts, colorscale=GRADE_SCALE, cmin=0, cmax=100),
        width=0.6, hoverinfo="skip",
    ))
    style_fig(pct_fig, height=320)
    pct_fig.update_layout(
        title="Percentile rankings",
        xaxis=dict(range=[0, 125], showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=10, r=10, t=50, b=20), showlegend=False,
    )

    # ---- Stuff+ distribution by pitch type ----
    d = scored_full[scored_full["Pitcher"] == p]
    order = [t for t in gp["CorrectedPitchType"] if (d["CorrectedPitchType"] == t).sum() >= 5]
    n = max(len(order), 1)
    dist = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=order if order else ["Not enough data"])
    xs = np.linspace(50, 150, 200)
    for i, t in enumerate(order, start=1):
        vals = d.loc[d["CorrectedPitchType"] == t, "stuff_plus"].dropna().values
        if len(vals) < 5 or np.std(vals) < 1e-6:
            continue
        ys = gaussian_kde(vals)(xs)
        color = CMAP.get(t, MUTED)
        dist.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy",
                                  line=dict(color=color, width=2),
                                  showlegend=False, hoverinfo="skip"), row=i, col=1)
        dist.add_vline(x=float(vals.mean()), line_dash="dash", line_color=color,
                       row=i, col=1)
    style_fig(dist, height=max(320, 95 * n + 70))
    dist.update_yaxes(visible=False)
    dist.update_xaxes(range=[50, 150])
    dist.update_annotations(font=dict(size=13, color=TEXT_SOFT))
    dist.update_layout(title="Stuff+ distribution (per pitch)",
                       margin=dict(l=10, r=10, t=60, b=30), showlegend=False)

    # ---- Grouped bars: Stuff+ and CSW+ per pitch type ----
    long = gp.melt(id_vars=["CorrectedPitchType", "pitches"],
                   value_vars=["stuff_plus", "csw_plus"],
                   var_name="metric", value_name="grade")
    long["metric"] = long["metric"].map(NICE)
    bar = px.bar(long, x="CorrectedPitchType", y="grade", color="metric", barmode="group",
                 hover_data={"pitches": True, "grade": ":.1f"}, labels=LABELS,
                 title=f"{p}: grades by pitch type", color_discrete_map=METRIC_COLORS)
    ref_h(bar)
    style_fig(bar, height=420)

    return card, pct_fig, dist, bar, gp[["CorrectedPitchType", "pitches", "stuff_plus", "csw_plus"]].to_dict("records")


@app.callback(Output("type-box", "figure"), Output("scatter", "figure"),
              Input("metric", "value"),
              Input("pitcher-select", "value"),
              Input("pitcher-mode", "value"))
def update_types(metric, selected, mode):
    # ---- Box plot ----
    if metric == "Both":
        long = scored.melt(id_vars=["CorrectedPitchType"],
                           value_vars=["stuff_plus", "csw_plus"],
                           var_name="metric", value_name="grade")
        long["metric"] = long["metric"].map(NICE)
        box = px.box(long, x="CorrectedPitchType", y="grade", color="metric",
                     title="Stuff+ and CSW+ by pitch type", labels=LABELS,
                     color_discrete_map=METRIC_COLORS,
                     category_orders={"CorrectedPitchType": TYPES})
    else:
        box = px.box(scored, x="CorrectedPitchType", y=metric, color="CorrectedPitchType",
                     color_discrete_map=CMAP, title=f"{NICE[metric]} by pitch type",
                     labels=LABELS, category_orders={"CorrectedPitchType": TYPES})
        box.update_layout(showlegend=False)
    style_fig(box, height=420)

    is_sel = grades["Pitcher"].isin(selected)
    sel = grades[is_sel]
    other = grades[~is_sel]
    show_base = not (mode == "Show only" and selected)
    dim = 0.35 if selected else 0.8

    # ---- Scatter, colored by pitch type ----
    if metric == "Both":
        sc = go.Figure()
        if show_base:
            sc = px.scatter(
                other, x="stuff_plus", y="csw_plus", color="CorrectedPitchType",
                color_discrete_map=CMAP, hover_name="Pitcher", labels=LABELS,
                hover_data={"pitches": True, "stuff_plus": ":.1f", "csw_plus": ":.1f"},
                opacity=dim, category_orders={"CorrectedPitchType": TYPES},
            )
            sc.update_traces(marker_size=8)
        if len(sel):
            top = px.scatter(
                sel, x="stuff_plus", y="csw_plus", color="CorrectedPitchType",
                color_discrete_map=CMAP, hover_name="Pitcher", labels=LABELS,
                hover_data={"pitches": True, "stuff_plus": ":.1f", "csw_plus": ":.1f"},
                text="Pitcher",
            )
            top.update_traces(marker=dict(size=15, line=dict(width=2, color=OUTLINE)),
                              textposition="top center", showlegend=False)
            for t in top.data:
                sc.add_trace(t)
        sc.update_layout(title="Stuff+ vs CSW+ by pitch type", hovermode="closest",
                         xaxis_title="Stuff+", yaxis_title="CSW+")
        ref_h(sc)
        ref_v(sc)
    else:
        sc = go.Figure()
        if show_base:
            base = px.strip(
                other, x="CorrectedPitchType", y=metric, color="CorrectedPitchType",
                color_discrete_map=CMAP, hover_name="Pitcher", labels=LABELS,
                hover_data={"pitches": True, metric: ":.1f"},
                category_orders={"CorrectedPitchType": TYPES}, stripmode="overlay",
            )
            base.update_traces(marker=dict(size=7, opacity=dim))
            for t in base.data:
                sc.add_trace(t)
        for t_name in TYPES:
            s = sel[sel["CorrectedPitchType"] == t_name]
            if len(s):
                sc.add_trace(go.Scatter(
                    x=s["CorrectedPitchType"], y=s[metric], mode="markers+text",
                    text=s["Pitcher"], textposition="middle right",
                    marker=dict(size=15, color=CMAP[t_name],
                                line=dict(width=2, color=OUTLINE)),
                    name=t_name, legendgroup=t_name, showlegend=False,
                    hovertemplate="%{text}<br>" + NICE[metric] + ": %{y:.1f}<extra></extra>",
                ))
        sc.update_layout(title=f"{NICE[metric]} by pitch type (each dot = one pitcher's pitch type)",
                         hovermode="closest", yaxis_title=NICE[metric], xaxis_title="Pitch type")
        sc.update_xaxes(categoryorder="array", categoryarray=TYPES)
        ref_h(sc)
    shown = set(sel["CorrectedPitchType"])         
    if show_base:
        shown |= set(other["CorrectedPitchType"])
    add_type_legend(sc, shown)
    style_fig(sc, height=520)

    return box, sc


if __name__ == "__main__":
    app.run(debug=True)