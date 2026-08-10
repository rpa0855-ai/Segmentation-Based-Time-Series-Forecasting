"""
Segmentation-Based Time Series Forecasting
Using Change Point Detection and Statistical Modeling

Interactive Streamlit app - Pradeep Kumar Yadav, MSc Statistics, IIT Bombay
Guide: Prof. Ashok Jaiswal, MIT-WPU Pune
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas_datareader.data as web

st.set_page_config(page_title="Segmentation-Based Forecasting", layout="wide")

# ---------------------------------------------------------------------------
# Helper functions (same logic as the research notebook)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_fred_series(series_code, start_date):
    df = web.DataReader(series_code, "fred", start=start_date)
    df = df.rename(columns={series_code: "value"}).dropna()
    return df


def find_penalty_for_target(train_arr, target_bkps, algo_class, min_size=15,
                             dates_arr=None, require_event=None):
    """Grid-search for a penalty giving close to target_bkps breakpoints.
    If require_event (e.g. '2020') is given, prefer a penalty that includes
    a breakpoint in that year/month, up to a max breakpoint cap."""
    mults = np.geomspace(0.005, 3.0, 50)
    candidates = []
    for mult in mults:
        pen = mult * np.log(len(train_arr)) * np.var(train_arr)
        algo = algo_class(model="l2", min_size=min_size).fit(train_arr)
        bkps = [b for b in algo.predict(pen=pen) if b < len(train_arr)]
        candidates.append((mult, pen, bkps))

    if require_event and dates_arr is not None:
        event_candidates = [c for c in candidates
                             if any(require_event in str(dates_arr[b])[:7] for b in c[2])
                             and len(c[2]) <= 12]
        if event_candidates:
            best = min(event_candidates, key=lambda c: abs(len(c[2]) - target_bkps))
            return best[1], best[2]

    best = min(candidates, key=lambda c: abs(len(c[2]) - target_bkps))
    return best[1], best[2]


def best_arima(series, p_range=range(3), d_range=(0, 1), q_range=range(3)):
    best_aic, best_model, best_order = np.inf, None, None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    m = ARIMA(series, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_model, best_order = m.aic, m, (p, d, q)
                except Exception:
                    continue
    return best_model, best_order


def best_sarima(series, s=12, p_range=(0, 1), q_range=(0, 1), P_range=(0, 1), Q_range=(0, 1)):
    best_aic, best_model, best_order = np.inf, None, None
    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:
                    if p == q == P == Q == 0:
                        continue
                    try:
                        m = SARIMAX(series, order=(p, 0, q), seasonal_order=(P, 0, Q, s), trend="c",
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                        if m.aic < best_aic:
                            best_aic, best_model, best_order = m.aic, m, ((p, 0, q), (P, 0, Q, s))
                    except Exception:
                        continue
    return best_model, best_order


def damp_forecast(raw_forecast, history, phi=0.5, drift_fraction=0.5):
    last_val = history[-1]
    drift = np.mean(np.diff(history)) * drift_fraction
    h = np.arange(1, len(raw_forecast) + 1)
    baseline = last_val + drift * h
    return baseline + (phi ** h) * (raw_forecast - baseline)


# ---------------------------------------------------------------------------
# Sidebar - data source selection
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**Segmentation-Based Time Series Forecasting**")
st.sidebar.markdown("_Change Point Detection + ARIMA/SARIMA + Weighted Ensemble_")
st.sidebar.markdown("---")

data_source = st.sidebar.radio("Data source", ["FRED (INDPRO, US Industrial Production)", "Upload my own CSV"])

if data_source.startswith("FRED"):
    start_date = st.sidebar.text_input("Start date", "1995-01-01")
    df = None
    if st.sidebar.button("Load INDPRO data"):
        with st.spinner("Downloading from FRED..."):
            df = load_fred_series("INDPRO", start_date)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (needs a date column and a value column)", type="csv")
    df = None
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.sidebar.write("Columns found:", list(raw.columns))
        date_col = st.sidebar.selectbox("Date column", raw.columns)
        value_col = st.sidebar.selectbox("Value column", raw.columns, index=min(1, len(raw.columns) - 1))
        raw[date_col] = pd.to_datetime(raw[date_col])
        df = raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"})
        df = df.sort_values("date").set_index("date")

test_h = st.sidebar.slider("Hold-out test months", 6, 36, 24)
min_size = st.sidebar.slider("Minimum segment size (months)", 6, 36, 15)
target_bkps = st.sidebar.slider("Target number of breakpoints", 2, 15, 8)

st.title("📊 Segmentation-Based Time Series Forecasting")
st.caption("Using Change Point Detection (PELT / Binary Segmentation) and Statistical Modeling (ARIMA / SARIMA)")

if df is None:
    st.info("👈 Load the FRED dataset or upload your own CSV from the sidebar to begin.")
    st.markdown("""
    **About this app**

    This app detects structural breaks (regime shifts) in a time series, fits ARIMA/SARIMA
    models to each detected segment, and combines them into a recency-weighted ensemble
    forecast - then compares that against a conventional single model.

    Built by **Pradeep Kumar Yadav**, M.Sc. Statistics, IIT Bombay
    (Guide: Prof. Ashok Jaiswal, MIT-WPU Pune)
    """)
    st.stop()

y = df["value"].values
dates = df.index

st.subheader("1. Raw Data")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(dates, y, color="#2b6cb0")
ax.set_title("Full series")
ax.grid(alpha=0.3)
st.pyplot(fig)

if len(y) < test_h + 40:
    st.error(f"Need at least {test_h + 40} observations for a meaningful analysis. This series has {len(y)}.")
    st.stop()

train = y[:-test_h]
train_dates = dates[:-test_h]
test_dates = dates[-test_h:]
y_test = y[-test_h:]

# ---------------------------------------------------------------------------
# Change point detection
# ---------------------------------------------------------------------------
st.subheader("2. Change Point Detection")
with st.spinner("Running PELT and Binary Segmentation..."):
    penalty_pelt, bkps_pelt = find_penalty_for_target(train, target_bkps, rpt.Pelt,
                                                        min_size=min_size, dates_arr=train_dates,
                                                        require_event="2020")
    penalty_bs, bkps_bs = find_penalty_for_target(train, target_bkps, rpt.Binseg,
                                                   min_size=min_size, dates_arr=train_dates,
                                                   require_event="2020")

col1, col2 = st.columns(2)
with col1:
    st.metric("PELT breakpoints", len(bkps_pelt))
with col2:
    st.metric("Binary Segmentation breakpoints", len(bkps_bs))

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for ax, bkps, name in [(axes[0], bkps_pelt, "PELT"), (axes[1], bkps_bs, "Binary Segmentation")]:
    ax.plot(train_dates, train, color="#2b6cb0")
    for b in bkps:
        ax.axvline(train_dates[b], color="#e53e3e", linestyle="--")
    ax.set_title(f"{name}: {len(bkps)} change points")
    ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

bounds = [0] + bkps_pelt + [len(train)]
segments = []
for i in range(len(bounds) - 1):
    a, b = bounds[i], bounds[i + 1]
    segments.append({"start": a, "end": b, "y": train[a:b],
                      "start_date": str(train_dates[a])[:10], "end_date": str(train_dates[b - 1])[:10]})

seg_table = pd.DataFrame([{"Segment": i + 1, "Start": s["start_date"], "End": s["end_date"],
                            "Months": s["end"] - s["start"]} for i, s in enumerate(segments)])
st.dataframe(seg_table, use_container_width=True)

# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
st.subheader("3. Fitting ARIMA / SARIMA per Segment")
progress = st.progress(0)
seg_models = []
for i, seg in enumerate(segments):
    with st.spinner(f"Fitting segment {i+1}/{len(segments)}..."):
        arima_model, arima_order = best_arima(seg["y"])
        sarima_model, sarima_order = (None, None)
        if len(seg["y"]) >= 30:
            sarima_model, sarima_order = best_sarima(seg["y"])
        seg_models.append({"arima": arima_model, "arima_order": arima_order,
                            "sarima": sarima_model, "sarima_order": sarima_order, "meta": seg})
    progress.progress((i + 1) / len(segments))

with st.expander("See per-segment model orders (AIC-selected)"):
    for i, sm in enumerate(seg_models):
        line = f"Segment {i+1}: ARIMA{sm['arima_order']} (AIC={sm['arima'].aic:.1f})"
        if sm["sarima"] is not None:
            line += f"  |  SARIMA{sm['sarima_order'][0]}x{sm['sarima_order'][1]} (AIC={sm['sarima'].aic:.1f})"
        st.text(line)

# ---------------------------------------------------------------------------
# Single models + Ensemble
# ---------------------------------------------------------------------------
st.subheader("4. Forecast Comparison")
with st.spinner("Fitting single whole-series models and building ensemble..."):
    single_arima, single_arima_order = best_arima(train)
    single_sarima, single_sarima_order = best_sarima(train)
    recent_hist = train[-36:] if len(train) >= 36 else train

    fc_single_arima = damp_forecast(np.asarray(single_arima.forecast(test_h)), recent_hist, phi=0.5)
    fc_single_sarima = damp_forecast(np.asarray(single_sarima.forecast(test_h)), recent_hist, phi=0.5)

    last_seg = seg_models[-1]
    model_last = last_seg["sarima"] if last_seg["sarima"] is not None else last_seg["arima"]
    fc_segwise = damp_forecast(np.asarray(model_last.forecast(test_h)), recent_hist, phi=0.5)

    n_seg = len(seg_models)
    seg_lengths = np.array([sm["meta"]["end"] - sm["meta"]["start"] for sm in seg_models])
    recency_w = np.array([0.3 ** (n_seg - 1 - i) for i in range(n_seg)])
    reliability_w = np.minimum(1.0, np.sqrt(seg_lengths / 36))
    weights = recency_w * reliability_w
    weights = weights / weights.sum()

    seg_fc = []
    for i, sm in enumerate(seg_models):
        model_i = sm["sarima"] if sm["sarima"] is not None else sm["arima"]
        if i == n_seg - 1:
            raw = np.asarray(model_i.forecast(test_h))
        else:
            raw = np.asarray(model_i.apply(recent_hist).forecast(test_h))
        seg_fc.append(damp_forecast(raw, recent_hist, phi=0.3))
    seg_fc = np.array(seg_fc)
    fc_ensemble = (weights[:, None] * seg_fc).sum(axis=0)

def score(name, forecast):
    forecast = np.asarray(forecast)
    return {"Model": name,
            "RMSE": round(np.sqrt(mean_squared_error(y_test, forecast)), 3),
            "MAE": round(mean_absolute_error(y_test, forecast), 3),
            "MAPE %": round(mean_absolute_percentage_error(y_test, forecast) * 100, 3)}

results = pd.DataFrame([
    score("Single ARIMA", fc_single_arima),
    score("Single SARIMA", fc_single_sarima),
    score("Segment-wise (latest regime)", fc_segwise),
    score("Weighted Ensemble", fc_ensemble),
]).sort_values("RMSE").reset_index(drop=True)

col1, col2 = st.columns([1, 1.3])
with col1:
    st.dataframe(results, use_container_width=True)
    best_model = results.iloc[0]["Model"]
    st.success(f"✅ Best model on this test window: **{best_model}**")

with col2:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(test_dates, y_test, label="Actual", color="black", linewidth=2, marker="o", markersize=3)
    ax.plot(test_dates, fc_single_arima, label="Single ARIMA", linestyle="--")
    ax.plot(test_dates, fc_single_sarima, label="Single SARIMA", linestyle="--")
    ax.plot(test_dates, fc_segwise, label="Segment-wise", linestyle="-.")
    ax.plot(test_dates, fc_ensemble, label="Weighted Ensemble", linewidth=2, color="red")
    ax.legend(fontsize=8)
    ax.set_title("Actual vs Predicted (hold-out test)")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.caption("Segmentation-Based Time Series Forecasting Using Change Point Detection and Statistical Modeling  |  "
           "Pradeep Kumar Yadav, M.Sc. Statistics, IIT Bombay  |  Guide: Prof. Ashok Jaiswal, MIT-WPU Pune")
