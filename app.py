"""
Flow-Aware RND — Interactive Web App
=====================================
Follows the demo.ipynb workflow:
  1. Load option chain (upload CSV or use bundled sample)
  2. Preprocess  (filter OTM → front-month → aggregate by strike)
  3. Calibrate Heston model  (parallel differential evolution)
  4. Extract FFT PDF  and  Breeden-Litzenberger PDF
  5. Apply flow-aware adjustment  (GEX / VEX / CEX / OI)
  6. Apply full-flow adjustment   (walls + OI predictors)
  7. Compute Call Wall, Put Wall, Magnet
  8. Interactive Plotly chart + summary tables
"""

import io
import os
import sys
import logging
import traceback
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from pipe import preprocess_option_chain, get_front_month_chain
from heston_calibrator import (
    HestonCalibrator,
    FlowAwarePDF,
    FlowData,
)
from validate import score_pdf_quality

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "option_chain.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flow-Aware RND",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Flow-Aware Risk-Neutral Density")
st.caption(
    "Heston-calibrated PDF with dealer-flow adjustments "
    "(GEX / VEX / CEX / OI).  Upload your option chain or use the sample data."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Data source ──────────────────────────────────────────────────────────
    st.subheader("Data")
    uploaded_file = st.file_uploader(
        "Upload option chain CSV",
        type=["csv"],
        help="Must contain the same columns as the sample option_chain.csv "
             "(strike, type, expiry, gatherdate, stk_price, lastprice, "
             "impliedvolatility, gexp, vexp, cexp, oi_chg, volume, "
             "openinterest, timevalue, fairvalue, vega).",
    )
    use_sample = st.checkbox("Use bundled sample data", value=(uploaded_file is None))

    # ── Market params ─────────────────────────────────────────────────────────
    st.subheader("Market Parameters")
    r = st.number_input("Risk-free rate", value=0.0405, min_value=0.0, max_value=0.20, step=0.001, format="%.4f")
    q = st.number_input("Dividend yield", value=0.00, min_value=0.0, max_value=0.10, step=0.001, format="%.4f")
    expiry_idx = st.number_input(
        "Expiry index (0 = front month)",
        value=0, min_value=0, max_value=10, step=1,
        help="Index into sorted expiry list (0 = nearest expiration).",
    )
    slippage = st.slider(
        "Display window (±% of spot)",
        min_value=5, max_value=30, value=15, step=1,
        help="Clip the PDF chart to spot ± this percentage.",
    )

    # ── Calibration params ────────────────────────────────────────────────────
    st.subheader("Heston Calibration")
    weight_type = st.selectbox(
        "Weighting scheme", ["atm", "tail", "uniform"],
        help="'atm' favours near-the-money; 'tail' favours far OTM; 'uniform' equal weight.",
    )
    pop_size = st.slider("DE population size", min_value=10, max_value=60, value=20, step=5)
    max_iter = st.slider("DE iterations", min_value=20, max_value=200, value=60, step=10)

    # ── Flow adjustment params ────────────────────────────────────────────────
    st.subheader("Flow Adjustment (α)")
    alpha_gex = st.slider("α GEX (gamma)", 0.0, 1.0, 0.25, 0.05)
    alpha_vex = st.slider("α VEX (vanna)", 0.0, 1.0, 0.25, 0.05)
    alpha_cex = st.slider("α CEX (charm)", 0.0, 1.0, 0.05, 0.05)
    alpha_oi  = st.slider("α OI change",   0.0, 1.0, 0.15, 0.05)

    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(raw_bytes),
        parse_dates=["expiry", "gatherdate"],
    )


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV, parse_dates=["expiry", "gatherdate"])


def get_dataframe() -> pd.DataFrame | None:
    if uploaded_file is not None:
        return load_csv(uploaded_file.getvalue())
    if use_sample:
        return load_sample()
    return None


raw_df = get_dataframe()

if raw_df is None:
    st.info("👈  Upload a CSV or enable **Use bundled sample data** in the sidebar, then click **Run Analysis**.")
    st.stop()

# Use the most-recent gather date (as demo.ipynb does)
ochain = raw_df[raw_df["gatherdate"] == raw_df["gatherdate"].max()].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔍 Raw data preview", expanded=False):
    st.write(f"Shape: {ochain.shape[0]} rows × {ochain.shape[1]} columns")
    st.dataframe(ochain.head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline – only runs when user clicks the button
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("👆  Configure parameters in the sidebar and click **Run Analysis**.")
    st.stop()


def run_pipeline(ochain, expiry_idx, r, q, weight_type, pop_size, max_iter,
                 alpha_gex, alpha_vex, alpha_cex, alpha_oi, slippage):
    """Execute full demo.ipynb workflow and return all artefacts."""

    results = {}

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    try:
        gcdf = preprocess_option_chain(ochain, expiry=expiry_idx)
    except Exception as exc:
        return None, f"Preprocessing failed: {exc}"

    if len(gcdf) < 5:
        return None, "Not enough strikes after preprocessing (need ≥ 5)."

    S0 = float(gcdf["stk_price"].iloc[-1])
    tau = float(gcdf["timevalue"].mean())
    strikes = gcdf["strike"].values.astype(float)
    market_prices = gcdf["lastprice"].values.astype(float)
    gamma_exposure = gcdf["gexp"].values.astype(float) if "gexp" in gcdf.columns else None
    charm_exposure = gcdf["cexp"].values.astype(float) if "cexp" in gcdf.columns else None
    vanna_exposure = gcdf["vexp"].values.astype(float) if "vexp" in gcdf.columns else None
    volume = gcdf["volume"].values.astype(float) if "volume" in gcdf.columns else None
    openinterest = gcdf["openinterest"].values.astype(float) if "openinterest" in gcdf.columns else None
    oi_chg = gcdf["oi_chg"].values.astype(float) if "oi_chg" in gcdf.columns else None
    market_ivs_raw = gcdf["impliedvolatility"].values.astype(float) if "impliedvolatility" in gcdf.columns else None

    results.update(dict(
        gcdf=gcdf, S0=S0, tau=tau, strikes=strikes, market_prices=market_prices,
        gamma_exposure=gamma_exposure, charm_exposure=charm_exposure,
        vanna_exposure=vanna_exposure, volume=volume, oi_chg=oi_chg,
        market_ivs_raw=market_ivs_raw,
    ))

    # ── 2. Calibrate Heston ───────────────────────────────────────────────────
    try:
        N, umax = HestonCalibrator.auto_select_params(tau, strikes, S0)
        calibrator = HestonCalibrator(N=N, umax=umax)

        flow_wts = {}
        if gamma_exposure is not None:
            flow_wts["gamma"] = gamma_exposure
        if charm_exposure is not None:
            flow_wts["charm"] = charm_exposure

        cal_result = calibrator.calibrate(
            strikes=strikes,
            market_prices=market_prices,
            S0=S0, tau=tau, r=r, q=q,
            weight_type=weight_type,
            flow_weights=flow_wts if flow_wts else None,
            pop_size=pop_size,
            max_iter=max_iter,
        )
        results["cal_result"] = cal_result
        results["calibrator"] = calibrator
    except Exception as exc:
        return None, f"Calibration failed: {exc}\n{traceback.format_exc()}"

    # ── 3. Extract FFT PDF ────────────────────────────────────────────────────
    try:
        strikes_fft, pdf_fft = calibrator.extract_pdf_fft(S0, cal_result.params, tau, r, q)
        results["strikes_fft"] = strikes_fft
        results["pdf_fft"] = pdf_fft
    except Exception as exc:
        return None, f"FFT PDF extraction failed: {exc}"

    # ── 4. Extract Breeden-Litzenberger PDF ───────────────────────────────────
    try:
        strikes_bl, pdf_bl = calibrator.extract_pdf_from_prices(
            S0, cal_result.params, tau, r, q, n_strikes=120
        )
        results["strikes_bl"] = strikes_bl
        results["pdf_bl"] = pdf_bl
    except Exception as exc:
        results["strikes_bl"] = None
        results["pdf_bl"] = None

    # ── 5. Flow-aware adjustment ──────────────────────────────────────────────
    try:
        flows = FlowData(
            strikes=strikes,
            gex=gamma_exposure,
            vex=vanna_exposure,
            cex=charm_exposure,
            oi_chg=None,   # oi_chg applied separately via full_flow_adjustment
            volume=volume,
        )
        flow_adj = FlowAwarePDF(
            alpha_gex=alpha_gex,
            alpha_vex=alpha_vex,
            alpha_cex=alpha_cex,
            alpha_oi=alpha_oi,
            mode_bandwidth=0.1,
            tail_threshold=0.20,
        )
        pdf_adjusted, components = flow_adj.adjust_pdf(
            strikes=strikes_fft,
            pdf=pdf_fft,
            S0=S0,
            flows=flows,
            tau=tau,
        )
        results["pdf_adjusted"] = pdf_adjusted
        results["flow_adj_components"] = components

        key = flow_adj.identify_key_strikes(flows.strikes, flows, S0, n_strikes=3)
        results["key_strikes"] = key
    except Exception as exc:
        results["pdf_adjusted"] = None
        results["key_strikes"] = None

    # ── 6. Full-flow adjustment (OI / walls) ──────────────────────────────────
    try:
        chain_for_walls = get_front_month_chain(ochain, expiry_idx)
        oi_pred_pdf, oi_info = flow_adj.full_flow_adjustment(
            strikes=strikes_fft,
            pdf=pdf_fft,
            S0=S0, tau=tau,
            option_chain=chain_for_walls,
            weight_col="openinterest",
        )
        results["oi_pred_pdf"] = oi_pred_pdf
        results["oi_info"] = oi_info
    except Exception as exc:
        results["oi_pred_pdf"] = None
        results["oi_info"] = None

    # ── 7. Walls & magnet ────────────────────────────────────────────────────
    try:
        walls = flow_adj.compute_walls_and_magnet(
            chain_for_walls, S0, weight_col="volume"
        )
        results["walls"] = walls
    except Exception:
        results["walls"] = None

    # ── 8. Model IVs for diagnostics ─────────────────────────────────────────
    try:
        diag = calibrator.get_diagnostics()
        results["market_ivs"] = diag.get("market_ivs")
        results["model_ivs"] = diag.get("model_ivs")
    except Exception:
        results["market_ivs"] = None
        results["model_ivs"] = None

    # ── 9. PDF quality score ─────────────────────────────────────────────────
    try:
        valid = (strikes_fft > S0 * 0.8) & (strikes_fft < S0 * 1.2) & (pdf_fft > 0)
        if np.sum(valid) > 20:
            score, diag_info = score_pdf_quality(
                strikes_fft[valid], pdf_fft[valid], S0, verbose=False
            )
            results["pdf_score"] = score
            results["pdf_diag"] = diag_info
        else:
            results["pdf_score"] = None
    except Exception:
        results["pdf_score"] = None

    return results, None


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Running pipeline… (calibration may take ~30-60 s)"):
    res, err = run_pipeline(
        ochain, expiry_idx, r, q, weight_type, pop_size, max_iter,
        alpha_gex, alpha_vex, alpha_cex, alpha_oi, slippage,
    )

if err:
    st.error(f"❌ Pipeline error:\n\n```\n{err}\n```")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics row
# ─────────────────────────────────────────────────────────────────────────────
S0    = res["S0"]
tau   = res["tau"]
cr    = res["cal_result"]
walls = res.get("walls")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Spot (S₀)", f"{S0:.2f}")
col2.metric("τ (years)", f"{tau:.4f}", f"≈ {tau*365:.1f} DTE")
col3.metric("Cal. RMSE", f"{cr.rmse:.5f}", "IV-weighted" if cr.success else "⚠️ check fit")
if res.get("pdf_score") is not None:
    col4.metric("PDF quality score", f"{res['pdf_score']:.3f}", "lower = better")
else:
    col4.metric("PDF quality score", "—")
if walls and not np.isnan(walls.get("magnet", np.nan)):
    bias_txt = "Bullish" if walls["directional_bias"] > 0 else "Bearish"
    col5.metric("Directional bias", bias_txt, f"{walls['bias_magnitude']:.2f} strength")
else:
    col5.metric("Directional bias", "—")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_pdf, tab_iv, tab_flow = st.tabs(["📊 PDF Analysis", "📉 IV Fit", "🌊 Flow Summary"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 – PDF chart
# ──────────────────────────────────────────────────────────────────────────────
with tab_pdf:
    strikes_fft = res["strikes_fft"]
    pdf_fft     = res["pdf_fft"]
    pdf_adj     = res.get("pdf_adjusted")
    oi_pred_pdf = res.get("oi_pred_pdf")
    strikes_bl  = res.get("strikes_bl")
    pdf_bl      = res.get("pdf_bl")

    upper = S0 * (1 + slippage / 100)
    lower = S0 * (1 - slippage / 100)

    def mask(arr, ref=strikes_fft):
        return (ref >= lower) & (ref <= upper) & np.isfinite(arr) & (arr > 0)

    fig = go.Figure()

    # FFT PDF
    m = mask(pdf_fft)
    if np.any(m):
        fig.add_trace(go.Scatter(
            x=strikes_fft[m], y=pdf_fft[m],
            mode="lines", name="FFT PDF",
            line=dict(color="#1f77b4", width=2),
        ))

    # BL PDF
    if strikes_bl is not None and pdf_bl is not None:
        mbl = mask(pdf_bl, strikes_bl)
        if np.any(mbl):
            fig.add_trace(go.Scatter(
                x=strikes_bl[mbl], y=pdf_bl[mbl],
                mode="lines", name="BL PDF",
                line=dict(color="#17becf", width=1.5, dash="dot"),
            ))

    # Flow-adjusted PDF
    if pdf_adj is not None:
        m_adj = mask(pdf_adj)
        if np.any(m_adj):
            fig.add_trace(go.Scatter(
                x=strikes_fft[m_adj], y=pdf_adj[m_adj],
                mode="lines", name="Flow-Adjusted PDF",
                line=dict(color="#d62728", width=2),
            ))

    # OI-predictor PDF
    if oi_pred_pdf is not None:
        m_oi = mask(oi_pred_pdf)
        if np.any(m_oi):
            fig.add_trace(go.Scatter(
                x=strikes_fft[m_oi], y=oi_pred_pdf[m_oi],
                mode="lines", name="OI Predictors PDF",
                line=dict(color="#2ca02c", width=2),
            ))

    # Vertical reference lines
    fig.add_vline(x=S0, line_width=1.5, line_dash="dash", line_color="grey",
                  annotation_text=f"Spot {S0:.2f}", annotation_position="top right")
    fig.add_vline(x=S0 * np.exp(r * tau), line_width=1, line_dash="dot", line_color="green",
                  annotation_text="Forward", annotation_position="top left")

    if walls:
        if not np.isnan(walls.get("call_wall", np.nan)):
            fig.add_vline(x=walls["call_wall"], line_width=1, line_dash="longdash",
                          line_color="orange",
                          annotation_text=f"Call Wall {walls['call_wall']:.1f}",
                          annotation_position="top right")
        if not np.isnan(walls.get("put_wall", np.nan)):
            fig.add_vline(x=walls["put_wall"], line_width=1, line_dash="longdash",
                          line_color="purple",
                          annotation_text=f"Put Wall {walls['put_wall']:.1f}",
                          annotation_position="top left")
        if not np.isnan(walls.get("magnet", np.nan)):
            fig.add_vline(x=walls["magnet"], line_width=1, line_dash="dot",
                          line_color="#ff7f0e",
                          annotation_text=f"Magnet {walls['magnet']:.1f}",
                          annotation_position="bottom right")

    # Key strikes
    key = res.get("key_strikes")
    if key is not None:
        for p in key.get("pinning", []):
            if lower <= p <= upper:
                fig.add_vline(x=p, line_width=0.8, line_color="blue", opacity=0.4,
                              annotation_text="📌", annotation_position="bottom right")
        for b in key.get("breakout", []):
            if lower <= b <= upper:
                fig.add_vline(x=b, line_width=0.8, line_color="red", opacity=0.4,
                              annotation_text="💥", annotation_position="bottom left")

    fig.update_layout(
        title=dict(text=f"Risk-Neutral PDF  (τ = {tau:.4f} yr, DTE ≈ {tau*365:.1f})",
                   font=dict(size=16)),
        xaxis_title="Strike",
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=520,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Integral checks
    st.caption("**PDF integral checks** (should be close to 1.0 over ±15 % window)")
    ic_cols = st.columns(4)
    integral_items = [
        ("FFT", strikes_fft, pdf_fft),
        ("BL", strikes_bl, pdf_bl),
        ("Flow-adj", strikes_fft, pdf_adj),
        ("OI-pred", strikes_fft, oi_pred_pdf),
    ]
    for col_idx, (label, sk, pk) in enumerate(integral_items):
        if sk is not None and pk is not None:
            v = (sk >= lower) & (sk <= upper) & np.isfinite(pk) & (pk > 0)
            intg = np.trapz(pk[v], sk[v]) if np.any(v) else 0.0
            ic_cols[col_idx].metric(label, f"{intg:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 – IV fit
# ──────────────────────────────────────────────────────────────────────────────
with tab_iv:
    market_ivs = res.get("market_ivs")
    model_ivs  = res.get("model_ivs")
    mkt_ivs_raw = res.get("market_ivs_raw")
    strikes_cal = res["strikes"]

    fig_iv = go.Figure()

    if mkt_ivs_raw is not None:
        valid_raw = np.isfinite(mkt_ivs_raw) & (mkt_ivs_raw > 0)
        if np.any(valid_raw):
            fig_iv.add_trace(go.Scatter(
                x=strikes_cal[valid_raw], y=mkt_ivs_raw[valid_raw] * 100,
                mode="markers", name="Market IV (chain)",
                marker=dict(color="blue", size=8, symbol="circle"),
            ))

    if market_ivs is not None:
        valid_mkt = np.isfinite(market_ivs) & (market_ivs > 0)
        if np.any(valid_mkt):
            fig_iv.add_trace(go.Scatter(
                x=strikes_cal[valid_mkt], y=market_ivs[valid_mkt] * 100,
                mode="markers", name="Market IV (from prices)",
                marker=dict(color="navy", size=7, symbol="x"),
            ))

    if model_ivs is not None:
        valid_mod = np.isfinite(model_ivs) & (model_ivs > 0)
        if np.any(valid_mod):
            fig_iv.add_trace(go.Scatter(
                x=strikes_cal[valid_mod], y=model_ivs[valid_mod] * 100,
                mode="lines+markers", name="Heston model IV",
                line=dict(color="red", width=2),
                marker=dict(size=5),
            ))

    fig_iv.add_vline(x=S0, line_width=1.5, line_dash="dash", line_color="grey",
                     annotation_text=f"Spot {S0:.2f}")
    fig_iv.update_layout(
        title="Implied Volatility Fit  (Market vs Heston)",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=450,
        template="plotly_white",
    )
    st.plotly_chart(fig_iv, use_container_width=True)

    # Calibrated parameters table
    p = cr.params
    st.subheader("Calibrated Heston Parameters")
    param_df = pd.DataFrame([{
        "κ (kappa)": f"{p.kappa:.4f}",
        "θ (theta)": f"{p.theta:.4f}",
        "v₀": f"{p.v0:.4f}",
        "ξ (vol-of-vol)": f"{p.xi:.4f}",
        "ρ (corr)": f"{p.rho:.4f}",
        "λⱼ (jump intensity)": f"{p.lambda_j:.4f}",
        "μⱼ (jump mean)": f"{p.mu_j:.4f}",
        "σⱼ (jump vol)": f"{p.sigma_j:.4f}",
        "Cal. RMSE": f"{cr.rmse:.6f}",
        "Iterations": cr.iterations,
    }])
    st.dataframe(param_df.T.rename(columns={0: "Value"}), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 – Flow summary
# ──────────────────────────────────────────────────────────────────────────────
with tab_flow:
    col_wall, col_key = st.columns(2)

    with col_wall:
        st.subheader("Walls & Magnet")
        if walls:
            bias_label = "🟢 Bullish" if walls.get("directional_bias", 0) > 0 else "🔴 Bearish"
            walls_df = pd.DataFrame([{
                "Call Wall": f"{walls.get('call_wall', np.nan):.2f}" if not np.isnan(walls.get('call_wall', np.nan)) else "—",
                "Put Wall": f"{walls.get('put_wall', np.nan):.2f}" if not np.isnan(walls.get('put_wall', np.nan)) else "—",
                "Magnet": f"{walls.get('magnet', np.nan):.2f}" if not np.isnan(walls.get('magnet', np.nan)) else "—",
                "Spread": f"{walls.get('spread', np.nan):.2f}" if not np.isnan(walls.get('spread', np.nan)) else "—",
                "Call Wall Strength": f"{walls.get('call_wall_strength', 0):.3f}",
                "Put Wall Strength": f"{walls.get('put_wall_strength', 0):.3f}",
                "Directional Bias": bias_label,
                "Bias Magnitude": f"{walls.get('bias_magnitude', 0):.3f}",
                "Put/Call Ratio": f"{walls.get('put_call_ratio', np.nan):.3f}" if not np.isnan(walls.get('put_call_ratio', np.nan)) else "—",
            }])
            st.dataframe(walls_df.T.rename(columns={0: "Value"}), use_container_width=True)
        else:
            st.info("Wall computation unavailable.")

    with col_key:
        st.subheader("Key Strikes")
        key = res.get("key_strikes")
        if key is not None:
            rows = []
            for k, label in [("pinning", "📌 Pinning"), ("breakout", "💥 Breakout"),
                               ("high_activity", "⚡ High Activity")]:
                for v in key.get(k, []):
                    rows.append({"Category": label, "Strike": f"{v:.1f}"})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No key strikes identified.")
        else:
            st.info("Flow data unavailable for key strike identification.")

    # Exposure bar chart
    gcdf = res["gcdf"]
    exp_cols = [c for c in ["gexp", "vexp", "cexp"] if c in gcdf.columns]
    if exp_cols:
        st.subheader("Dealer Exposure by Strike")
        exp_sel = st.selectbox("Exposure to display", exp_cols,
                               format_func=lambda x: {"gexp":"GEX (Gamma)","vexp":"VEX (Vanna)","cexp":"CEX (Charm)"}.get(x, x))
        exp_vals = gcdf[exp_sel].values
        fig_exp = go.Figure(go.Bar(
            x=gcdf["strike"].values,
            y=exp_vals,
            marker_color=np.where(exp_vals >= 0, "#2ca02c", "#d62728"),
            name=exp_sel,
        ))
        fig_exp.add_vline(x=S0, line_dash="dash", line_color="grey",
                          annotation_text=f"Spot {S0:.2f}")
        fig_exp.update_layout(
            title=f"{exp_sel.upper()} by Strike",
            xaxis_title="Strike", yaxis_title="Exposure ($)",
            height=350, template="plotly_white",
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    # OI / volume bar chart
    oi_cols = [c for c in ["openinterest", "volume"] if c in gcdf.columns]
    if oi_cols:
        oi_sel = st.selectbox("OI / Volume", oi_cols,
                              format_func=lambda x: {"openinterest": "Open Interest", "volume": "Volume"}.get(x, x))
        fig_oi = go.Figure(go.Bar(
            x=gcdf["strike"].values,
            y=gcdf[oi_sel].values,
            marker_color="#1f77b4",
            name=oi_sel,
        ))
        fig_oi.add_vline(x=S0, line_dash="dash", line_color="grey",
                         annotation_text=f"Spot {S0:.2f}")
        fig_oi.update_layout(
            title=f"{oi_sel.capitalize()} by Strike",
            xaxis_title="Strike", yaxis_title=oi_sel.capitalize(),
            height=320, template="plotly_white",
        )
        st.plotly_chart(fig_oi, use_container_width=True)
