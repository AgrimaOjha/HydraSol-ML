import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="HydraSol — Molecular Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

model = joblib.load("solubility_model (1).pkl")
features = joblib.load("model_features.pkl")

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Syne+Mono&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #03050f;
    color: #c8deff;
    font-family: 'Syne', sans-serif;
}

.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #03050f; }
::-webkit-scrollbar-thumb { background: rgba(74,244,255,0.2); border-radius: 4px; }

/* ── Animated starfield (CSS-only) ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        radial-gradient(circle, rgba(74,244,255,0.12) 1px, transparent 1px),
        radial-gradient(circle, rgba(192,132,252,0.08) 1px, transparent 1px),
        radial-gradient(circle, rgba(74,244,255,0.06) 1px, transparent 1px);
    background-size: 120px 120px, 180px 180px, 240px 240px;
    background-position: 0 0, 60px 60px, 30px 90px;
    animation: starDrift 80s linear infinite;
}
@keyframes starDrift {
    to { background-position: 120px 120px, 240px 240px, 270px 210px; }
}

/* ── Ambient orbs ── */
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 600px 400px at 10% 20%, rgba(74,244,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 500px 600px at 90% 80%, rgba(192,132,252,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 400px 300px at 50% 50%, rgba(4,12,40,0.6) 0%, transparent 70%);
}

/* ── Header ── */
.header-wrap {
    text-align: center;
    padding: 2.5rem 0 2rem;
    position: relative;
}
.header-eyemark {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Syne Mono', monospace;
    font-size: 10px;
    letter-spacing: 5px;
    color: rgba(74,244,255,0.5);
    margin-bottom: 12px;
}
.header-eyemark::before, .header-eyemark::after {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: rgba(74,244,255,0.3);
}
.header-title {
    font-size: clamp(36px, 5vw, 60px);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    background: linear-gradient(110deg, #a0c8ff 0%, #4af4ff 40%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
}
.header-sub {
    font-family: 'Syne Mono', monospace;
    font-size: 12px;
    letter-spacing: 4px;
    color: rgba(74,244,255,0.35);
}
.header-rift {
    width: 100%; height: 1px; margin: 2rem 0;
    background: linear-gradient(90deg,
        transparent,
        rgba(74,244,255,0.15) 20%,
        rgba(192,132,252,0.25) 50%,
        rgba(74,244,255,0.15) 80%,
        transparent
    );
    position: relative;
}
.header-rift::after {
    content: '◈';
    position: absolute; top: -8px; left: 50%;
    transform: translateX(-50%);
    font-size: 10px; color: rgba(74,244,255,0.6);
}

/* ── Panels ── */
.panel {
    background: rgba(4,12,40,0.7);
    border: 1px solid rgba(74,244,255,0.1);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    margin-bottom: 20px;
    transition: border-color 0.3s;
}
.panel:hover { border-color: rgba(74,244,255,0.2); }

.panel::before {
    content: '';
    position: absolute; inset: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 60% 40% at 0% 0%, rgba(74,244,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse 40% 60% at 100% 100%, rgba(192,132,252,0.04) 0%, transparent 50%);
}

/* corner brackets */
.panel::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 16px;
    pointer-events: none;
    background:
        linear-gradient(#4af4ff, #4af4ff) top left / 12px 1px no-repeat,
        linear-gradient(#4af4ff, #4af4ff) top left / 1px 12px no-repeat,
        linear-gradient(#4af4ff, #4af4ff) top right / 12px 1px no-repeat,
        linear-gradient(#4af4ff, #4af4ff) top right / 1px 12px no-repeat,
        linear-gradient(#4af4ff, #4af4ff) bottom left / 12px 1px no-repeat,
        linear-gradient(#4af4ff, #4af4ff) bottom left / 1px 12px no-repeat,
        linear-gradient(#4af4ff, #4af4ff) bottom right / 12px 1px no-repeat,
        linear-gradient(#4af4ff, #4af4ff) bottom right / 1px 12px no-repeat;
    opacity: 0.3;
}

.panel-label {
    font-family: 'Syne Mono', monospace;
    font-size: 10px;
    letter-spacing: 4px;
    color: rgba(74,244,255,0.45);
    margin-bottom: 16px;
}

/* ── Section headings ── */
h2, h3, .stSubheader { color: #c8deff !important; font-family: 'Syne', sans-serif !important; }

/* ── Sliders ── */
.stSlider > label {
    font-family: 'Syne Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    color: rgba(74,244,255,0.55) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #4af4ff !important;
    box-shadow: 0 0 10px rgba(74,244,255,0.6) !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #4af4ff !important;
    font-family: 'Syne Mono', monospace !important;
}

/* ── Selectbox ── */
.stSelectbox > label {
    font-family: 'Syne Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    color: rgba(74,244,255,0.55) !important;
}
.stSelectbox [data-baseweb="select"] {
    background: rgba(74,244,255,0.04) !important;
    border-color: rgba(74,244,255,0.2) !important;
    border-radius: 10px !important;
}
.stSelectbox [data-baseweb="select"] * { color: #c8deff !important; font-family: 'Syne', sans-serif !important; }

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    height: 52px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(74,244,255,0.35) !important;
    background: linear-gradient(135deg, rgba(74,244,255,0.12), rgba(192,132,252,0.18)) !important;
    color: #4af4ff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    transition: all 0.25s !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    border-color: rgba(74,244,255,0.7) !important;
    box-shadow: 0 0 30px rgba(74,244,255,0.15), 0 8px 32px rgba(0,0,0,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(74,244,255,0.04);
    border: 1px solid rgba(74,244,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-family: 'Syne Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 3px !important;
    color: rgba(74,244,255,0.45) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne Mono', monospace !important;
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #4af4ff !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4af4ff, #c084fc) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: rgba(74,244,255,0.06) !important;
    border-radius: 4px !important;
}

/* ── Result badge ── */
.result-badge {
    padding: 20px 24px;
    border-radius: 14px;
    text-align: center;
    margin: 16px 0;
    animation: fadeRise 0.5s cubic-bezier(0.16,1,0.3,1);
}
@keyframes fadeRise {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-family: 'Syne Mono', monospace;
    font-size: 10px;
    letter-spacing: 4px;
    margin-bottom: 6px;
}
.result-value {
    font-size: 20px;
    font-weight: 700;
}

/* ── Bar chart ── */
[data-testid="stVegaLiteChart"] canvas,
.stBarChart { filter: none !important; }

/* ── Caption ── */
.stCaption {
    font-family: 'Syne Mono', monospace !important;
    font-size: 11px !important;
    color: rgba(74,244,255,0.35) !important;
    letter-spacing: 1px !important;
}

/* ── Success / info ── */
.stSuccess {
    background: rgba(74,244,255,0.06) !important;
    border-color: rgba(74,244,255,0.25) !important;
    color: #4af4ff !important;
    font-family: 'Syne Mono', monospace !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(74,244,255,0.04) !important;
    border: 1px solid rgba(74,244,255,0.12) !important;
    border-radius: 10px !important;
    color: rgba(74,244,255,0.5) !important;
    font-family: 'Syne Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 3px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(74,244,255,0.08) !important;
    margin: 24px 0 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4af4ff !important; }

/* ── Columns gap fix ── */
[data-testid="column"] { padding: 0 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-eyemark">SOLUSCAN INTELLIGENCE SYSTEM</div>
    <div class="header-title">Molecular Solubility AI</div>
    <div class="header-sub">PREDICTION · EXPLAINABILITY · SIMULATION</div>
    <div class="header-rift"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([2, 1], gap="large")

# ── INPUT PANEL ──────────────────────────────
with left:
    st.markdown('<div class="panel"><div class="panel-label">// MOLECULAR DESCRIPTOR INPUT</div>', unsafe_allow_html=True)

    input_data = {}
    cols = st.columns(2)
    for i, f in enumerate(features):
        with cols[i % 2]:
            input_data[f] = st.slider(f, -2.0, 2.0, 0.0, step=0.01)

    st.markdown("<hr>", unsafe_allow_html=True)
    predict_btn = st.button("◈  TRANSMIT · PREDICT")
    st.markdown("</div>", unsafe_allow_html=True)

# ── OUTPUT PANEL ─────────────────────────────
with right:
    st.markdown('<div class="panel"><div class="panel-label">// PREDICTION READOUT</div>', unsafe_allow_html=True)

    df = pd.DataFrame([input_data])[features]
    pred = model.predict(df)[0]

    if predict_btn:
        with st.spinner("Scanning molecular topology..."):
            time.sleep(0.6)

    st.metric("Predicted LogS", f"{pred:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Signal strength bar
    confidence = min(100, int(abs(pred) * 40 + 10))
    st.caption("SIGNAL CONFIDENCE")
    st.progress(confidence)

    # Result classification
    if pred > 0:
        color, icon, label, detail = "#4af4ff", "◉", "HIGH SOLUBILITY", "Highly water-soluble compound — favorable aqueous behavior."
        bg = "rgba(74,244,255,0.08)"
        border = "rgba(74,244,255,0.3)"
    elif pred > -2:
        color, icon, label, detail = "#f0b429", "◎", "MODERATE SOLUBILITY", "Balanced solubility — structural optimization may improve profile."
        bg = "rgba(240,180,41,0.08)"
        border = "rgba(240,180,41,0.3)"
    else:
        color, icon, label, detail = "#ef4444", "◌", "LOW SOLUBILITY", "Poor aqueous solubility — formulation strategy required."
        bg = "rgba(239,68,68,0.08)"
        border = "rgba(239,68,68,0.3)"

    st.markdown(f"""
    <div class="result-badge" style="background:{bg}; border:1px solid {border};">
        <div class="result-label" style="color:{color};">{icon} STATUS</div>
        <div class="result-value" style="color:{color};">{label}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(detail)

    # LogS scale reference
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("LOGS SCALE REFERENCE")
    scale_html = """
    <div style="display:flex; gap:4px; margin-top:6px; height:6px; border-radius:4px; overflow:hidden;">
        <div style="flex:1; background:rgba(239,68,68,0.6);"></div>
        <div style="flex:1; background:rgba(240,180,41,0.6);"></div>
        <div style="flex:1; background:rgba(74,244,255,0.6);"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:4px;
                font-family:'Syne Mono',monospace; font-size:9px; color:rgba(200,222,255,0.3);">
        <span>−4 (insoluble)</span><span>−2</span><span>0 (soluble)</span>
    </div>
    """
    st.markdown(scale_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WHAT-IF ANALYSIS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="panel"><div class="panel-label">// WHAT-IF SIMULATION</div>', unsafe_allow_html=True)

w1, w2, w3 = st.columns([1, 1, 1], gap="large")

with w1:
    selected_feature = st.selectbox("SELECT FEATURE", features)

with w2:
    new_val = st.slider("ADJUSTED VALUE", -2.0, 2.0, float(input_data[selected_feature]), step=0.01)

with w3:
    temp_input = input_data.copy()
    temp_input[selected_feature] = new_val
    temp_df = pd.DataFrame([temp_input])[features]
    new_pred = model.predict(temp_df)[0]
    delta = new_pred - pred

    st.metric(
        label="SIMULATED LOGS",
        value=f"{new_pred:.4f}",
        delta=f"{delta:+.4f}"
    )

    arrow = "▲" if delta > 0 else "▼"
    direction = "INCREASE" if delta > 0 else "DECREASE"
    dir_color = "#4af4ff" if delta > 0 else "#ef4444"

    st.markdown(f"""
    <div style="margin-top:8px; font-family:'Syne Mono',monospace;
                font-size:10px; letter-spacing:2px; color:{dir_color};">
        {arrow} {direction} of {abs(delta):.4f} units
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="panel"><div class="panel-label">// FEATURE IMPORTANCE MATRIX</div>', unsafe_allow_html=True)

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fi1, fi2 = st.columns([2, 1], gap="large")

with fi1:
    st.bar_chart(
        feat_df.set_index("Feature"),
        color="#4af4ff",
        height=260
    )

with fi2:
    st.markdown('<div class="panel-label">TOP DRIVERS</div>', unsafe_allow_html=True)
    for _, row in feat_df.head(5).iterrows():
        pct = int(row["Importance"] * 100)
        bar_color = "#4af4ff" if pct == feat_df["Importance"].max() * 100 else "rgba(74,244,255,0.4)"
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-family:'Syne Mono',monospace; font-size:11px;
                             color:rgba(200,222,255,0.7); letter-spacing:1px;">{row['Feature']}</span>
                <span style="font-family:'Syne Mono',monospace; font-size:11px;
                             color:#4af4ff;">{pct}%</span>
            </div>
            <div style="height:3px; background:rgba(74,244,255,0.08); border-radius:2px;">
                <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:2px;
                            transition:width 0.6s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.caption("HIGHER IMPORTANCE → STRONGER MOLECULAR INFLUENCE ON SOLUBILITY")
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER / MODEL DETAILS
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("// SYSTEM · MODEL DETAILS"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="panel-label">MODEL ARCHITECTURE</div>
        <div style="font-family:'Syne Mono',monospace; font-size:12px;
                    color:rgba(200,222,255,0.6); line-height:2;">
            Random Forest Regressor<br>
            Ensemble · Non-linear
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="panel-label">INPUT SPACE</div>
        <div style="font-family:'Syne Mono',monospace; font-size:12px;
                    color:rgba(200,222,255,0.6); line-height:2;">
            {len(features)} Molecular Descriptors<br>
            Range: [−2.0, +2.0]
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="panel-label">OUTPUT</div>
        <div style="font-family:'Syne Mono',monospace; font-size:12px;
                    color:rgba(200,222,255,0.6); line-height:2;">
            LogS (log mol/L)<br>
            Aqueous Solubility
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:40px;
            font-family:'Syne Mono',monospace; font-size:10px;
            letter-spacing:4px; color:rgba(74,244,255,0.2);">
    SOLUSCAN · MOLECULAR INTELLIGENCE · v2.0
</div>
""", unsafe_allow_html=True)
