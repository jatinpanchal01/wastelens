"""Visual layer for WasteLens.

Pure presentation: nothing here reads or writes app state. All HTML below is
static (no user data is interpolated), so unsafe_allow_html is safe to use.

Design language: warm paper background, forest-green primary, ochre accent,
serif headings over Streamlit's humanist sans body. System serif stacks are
used on purpose so no font is fetched from a third party. To self-host a
display face later, add [[theme.fontFaces]] to .streamlit/config.toml.
"""
from __future__ import annotations

import streamlit as st

PAPER = "#F6F3EC"
SURFACE = "#FFFDF8"
INK = "#1F2A22"
MUTED = "#5E6B60"
LINE = "#E1DAC8"
FOREST = "#2F5D3A"
OCHRE = "#C28F2C"

PAGE_ICON = ":material/restaurant:"

_CSS = """
<style>
:root{
  --wl-paper:#F6F3EC; --wl-surface:#FFFDF8; --wl-ink:#1F2A22; --wl-muted:#5E6B60;
  --wl-line:#E1DAC8; --wl-forest:#2F5D3A; --wl-forest-deep:#234a2d; --wl-ochre:#C28F2C;
  --wl-serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --wl-ease:cubic-bezier(.22,.8,.3,1);
}

/* ---- Layout ---- */
.stApp{background:var(--wl-paper)}
header[data-testid="stHeader"]{background:transparent}
.block-container{position:relative;max-width:1280px;padding-top:2.25rem;padding-bottom:4rem}

/* Quiet "lens" motif: concentric rings, top right. Decorative only. */
.block-container::before{
  content:"";position:absolute;top:-2.5rem;right:-1rem;width:300px;height:300px;z-index:0;
  pointer-events:none;opacity:.9;
  background:
    radial-gradient(circle at center, var(--wl-paper) 0 21%, transparent 21.5%),
    repeating-radial-gradient(circle at center, transparent 0 22px, rgb(47 93 58 / .10) 22px 23px);
  -webkit-mask-image:radial-gradient(circle at center,#000 35%,transparent 72%);
          mask-image:radial-gradient(circle at center,#000 35%,transparent 72%);
}
.block-container > div{position:relative;z-index:1}

/* ---- Type ---- */
.stApp h1,.stApp h2,.stApp h3{font-family:var(--wl-serif);color:var(--wl-ink);letter-spacing:-.015em}
.stApp h1{font-size:clamp(2.4rem,4.6vw,3.6rem);font-weight:600;line-height:1.02;padding:.15rem 0 .5rem}
.stApp h2{font-weight:600}
.stApp h3{font-weight:600;font-size:1.45rem}
.stApp h4{font-family:var(--wl-serif);font-weight:600;letter-spacing:-.01em}
.wl-eyebrow{display:inline-flex;align-items:center;gap:.55rem;margin:0 0 .35rem;
  font-size:.74rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--wl-forest)}
.wl-eyebrow::before{content:"";width:1.6rem;height:2px;background:var(--wl-ochre);border-radius:2px}
[data-testid="stCaptionContainer"]{color:var(--wl-muted)}
[data-testid="stCaptionContainer"] p{line-height:1.5}

/* ---- Sidebar ---- */
[data-testid="stSidebar"]{border-right:1px solid var(--wl-line)}
[data-testid="stSidebar"] h2{font-size:1.25rem}

/* ---- Metrics ---- */
[data-testid="stMetric"]{
  background:var(--wl-surface);border:1px solid var(--wl-line);border-radius:14px;
  padding:1rem 1.1rem;box-shadow:0 1px 0 rgb(31 42 34 / .03);
  transition:border-color .25s var(--wl-ease), transform .25s var(--wl-ease);
}
[data-testid="stMetric"]:hover{border-color:#cfc6ad;transform:translateY(-1px)}
[data-testid="stMetricLabel"]{color:var(--wl-muted)}
[data-testid="stMetricLabel"] p{font-size:.74rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase}
[data-testid="stMetricValue"]{font-family:var(--wl-serif);font-weight:600;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}

/* ---- Cards: forms, bordered containers, expanders, tables ---- */
[data-testid="stForm"]{background:var(--wl-surface);border:1px solid var(--wl-line);border-radius:18px;padding:1.4rem 1.4rem 1.2rem}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--wl-line)!important;border-radius:16px!important}
[data-testid="stExpander"]{border:1px solid var(--wl-line);border-radius:14px;background:var(--wl-surface)}
[data-testid="stExpander"] summary{font-weight:600}
[data-testid="stDataFrame"],[data-testid="stTable"]{border-radius:12px;overflow:hidden}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--wl-line)}
hr{border-color:var(--wl-line)!important}

/* ---- Tabs ---- */
[data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid var(--wl-line)}
button[role="tab"]{padding:.7rem .9rem;border-radius:10px 10px 0 0}
button[role="tab"] p{font-weight:600;color:var(--wl-muted);transition:color .2s var(--wl-ease)}
button[role="tab"][aria-selected="true"] p,button[role="tab"]:hover p{color:var(--wl-forest)}
[data-baseweb="tab-highlight"]{background-color:var(--wl-forest)!important;height:3px;border-radius:3px 3px 0 0}
[data-baseweb="tab-border"]{display:none}

/* ---- Buttons ---- */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{
  border-radius:10px;font-weight:600;min-height:44px;
  transition:transform .15s var(--wl-ease), box-shadow .25s var(--wl-ease), background-color .2s var(--wl-ease), border-color .2s var(--wl-ease);
}
.stButton>button:active,.stDownloadButton>button:active,[data-testid="stFormSubmitButton"]>button:active{transform:translateY(1px) scale(.99)}
button[kind="primary"],button[kind="primaryFormSubmit"]{background:var(--wl-forest);border:1px solid var(--wl-forest-deep);color:#fff;
  box-shadow:inset 0 1px 0 rgb(255 255 255 / .16), 0 6px 16px -8px rgb(47 93 58 / .6)}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover{background:var(--wl-forest-deep);border-color:var(--wl-forest-deep);color:#fff}
button[kind="secondary"],button[kind="secondaryFormSubmit"]{background:var(--wl-surface);border:1px solid var(--wl-line);color:var(--wl-ink)}
button[kind="secondary"]:hover,button[kind="secondaryFormSubmit"]:hover{border-color:var(--wl-forest);color:var(--wl-forest)}
button:focus-visible,[role="tab"]:focus-visible,summary:focus-visible{outline:2px solid var(--wl-ochre)!important;outline-offset:2px}

/* ---- Inputs ---- */
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"]{border-radius:10px!important}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed #c9bfa5;border-radius:14px;background:var(--wl-surface)}

/* ---- Empty state ---- */
.wl-start{background:var(--wl-surface);border:1px solid var(--wl-line);border-radius:20px;padding:1.6rem 1.8rem;margin:.5rem 0 1.25rem;max-width:760px}
.wl-start h3{margin:0 0 .25rem;font-size:1.5rem}
.wl-start p.wl-sub{margin:0 0 1rem;color:var(--wl-muted)}
.wl-step{display:grid;grid-template-columns:3.2rem 1fr;gap:.5rem;align-items:baseline;padding:.85rem 0;border-top:1px solid var(--wl-line)}
.wl-step .n{font-family:var(--wl-serif);font-size:1.7rem;color:var(--wl-ochre);font-variant-numeric:tabular-nums;line-height:1}
.wl-step b{display:block;color:var(--wl-ink)}
.wl-step span{color:var(--wl-muted);font-size:.95rem}

/* ---- Motion (decorative, off for reduced-motion users) ---- */
@media (prefers-reduced-motion:no-preference){
  @keyframes wl-rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
  .wl-eyebrow,.stApp h1,.wl-start{animation:wl-rise .6s var(--wl-ease) both}
  .stApp h1{animation-delay:.05s}
  [data-testid="stMetric"]{animation:wl-rise .5s var(--wl-ease) both}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ---- Small screens ---- */
@media(max-width:900px){.block-container::before{display:none}}
@media(max-width:600px){
  .block-container{padding:1rem .75rem 3rem}
  .stButton>button,.stDownloadButton>button{width:100%;min-height:44px}
  .wl-start{padding:1.2rem}
}
</style>
"""


def _compact(html_block: str) -> str:
    """Drop blank lines and indentation so Streamlit's markdown parser keeps raw HTML intact."""
    return "\n".join(line.strip() for line in html_block.splitlines() if line.strip())


def inject_theme() -> None:
    st.markdown(_compact(_CSS), unsafe_allow_html=True)


def render_header() -> None:
    """Eyebrow + real <h1> + caption. Keeps st.title for semantics and tests."""
    st.markdown('<div class="wl-eyebrow">Kitchen planning</div>', unsafe_allow_html=True)
    st.title("WasteLens")
    st.caption("Forecast demand, understand waste risk, and prepare a practical plan. "
               "Recommendations support human decisions.")


def render_empty_state() -> None:
    st.markdown(
        _compact("""
<div class="wl-start">
  <h3>Start with your kitchen's history</h3>
  <p class="wl-sub">Three steps from raw records to tomorrow's prep plan.</p>
  <div class="wl-step"><div class="n">01</div><div><b>Bring in data</b>
    <span>Upload a CSV from the sidebar, or load the labeled synthetic demo to explore.</span></div></div>
  <div class="wl-step"><div class="n">02</div><div><b>Generate a plan</b>
    <span>Choose a date, meal and menu in Plan tomorrow to forecast demand and waste risk.</span></div></div>
  <div class="wl-step"><div class="n">03</div><div><b>Review and adjust</b>
    <span>Test what-ifs, set final quantities, and keep every plan in your local history.</span></div></div>
</div>
"""),
        unsafe_allow_html=True,
    )


def style_chart(figure, axes) -> None:
    """Match matplotlib output to the page: paper background, hairline axes."""
    figure.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.yaxis.label.set_color(MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.grid(axis="y", color=LINE, alpha=.9, linewidth=.8)
        ax.set_axisbelow(True)
