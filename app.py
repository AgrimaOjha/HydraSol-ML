import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ---------- CONFIG ----------
st.set_page_config(page_title="Molecular Solubility AI", layout="wide")

# ---------- LOAD ----------
model = joblib.load("solubility_model (1).pkl")
features = joblib.load("model_features.pkl")

# ---------- CSS (UNCHANGED) ----------
st.markdown("""
<style>
/* Background */
.stApp {
    background: radial-gradient(circle at top, #0b1220, #020617);
    color: white;
}

/* Header */
.title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    letter-spacing: -1px;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
    transition: 0.25s ease;
}
.card:hover {
    transform: translateY(-4px);
}

/* Result */
.result {
    padding: 26px;
    border-radius: 16px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    animation: fadeIn 0.6s ease-in-out;
}

/* Divider */
.divider {
    margin: 25px 0;
    height: 1px;
    background: rgba(255,255,255,0.08);
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 52px;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #ff7a18, #ffb347);
    color: white;
    border: none;
}

/* Animation */
@keyframes fadeIn {
    from {opacity:0; transform: translateY(10px);}
    to {opacity:1; transform: translateY(0);}
}

/* Tight top spacing */
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown('<div class="title">💧 Molecular Solubility AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prediction • Explainability • Simulation</div>', unsafe_allow_html=True)

# ---------- LAYOUT ----------
left, right = st.columns([2,1])

# ===========================
# 🔬 INPUT PANEL
# ===========================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔬 Molecular Descriptors")

    input_data = {}
    col1, col2 = st.columns(2)

    for i, f in enumerate(features):
        if i % 2 == 0:
            with col1:
                input_data[f] = st.slider(f, -2.0, 2.0, 0.0, step=0.01)
        else:
            with col2:
                input_data[f] = st.slider(f, -2.0, 2.0, 0.0, step=0.01)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Solubility")

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# 📊 OUTPUT PANEL (FIXED)
# ===========================
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction")

    # 🔥 CRITICAL FIX: enforce feature order
    df = pd.DataFrame([input_data])
    df = df[features]

    pred = model.predict(df)[0]

    # Animation only on click
    if predict_btn:
        with st.spinner("Analyzing molecular behavior..."):
            time.sleep(0.5)

    # Output
    st.metric("Predicted LogS", f"{pred:.3f}")

    # Confidence (UX only)
    confidence = min(100, int(abs(pred) * 40))
    st.progress(confidence)

    # Interpretation
    if pred > 0:
        msg = "High Solubility 💧"
        color = "#22c55e"
        insight = "Highly water-soluble compound."
    elif pred > -2:
        msg = "Moderate Solubility ⚖️"
        color = "#f59e0b"
        insight = "Balanced solubility — optimization possible."
    else:
        msg = "Low Solubility ⚠️"
        color = "#ef4444"
        insight = "Low solubility — may need improvement."

    st.markdown(
        f'<div class="result" style="background:{color}20; color:{color};">{msg}</div>',
        unsafe_allow_html=True
    )

    st.caption(insight)

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# 🧪 WHAT-IF ANALYSIS (FIXED)
# ===========================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧪 What-If Analysis")

selected_feature = st.selectbox("Select feature", features)
new_val = st.slider("Adjust value", -2.0, 2.0, 0.0)

temp_input = input_data.copy()
temp_input[selected_feature] = new_val

# 🔥 FIX AGAIN HERE
temp_df = pd.DataFrame([temp_input])[features]
new_pred = model.predict(temp_df)[0]

st.success(f"New Prediction: {new_pred:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# 🧠 MODEL EXPLAINABILITY
# ===========================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧠 Feature Importance")

importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(feat_df.set_index("Feature"))

st.caption("Higher importance → stronger impact.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
with st.expander("📊 Model Details"):
    st.write("Model: Random Forest Regressor")
    st.write("Pipeline: Input → Model → Prediction → Interpretation")