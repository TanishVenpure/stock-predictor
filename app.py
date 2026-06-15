import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
import sys, os

sys.path.append("src")
from feature_engineering import build_features, get_nifty_features

st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_models():
    return {
        "stack" : joblib.load("models/stacking_ensemble.pkl"),
        "rf"    : joblib.load("models/random_forest.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "feats" : joblib.load("models/feature_cols.pkl")
    }

models = load_models()

st.sidebar.title("⚙️ Settings")

ticker_map = {
    "Reliance Industries" : "RELIANCE.NS",
    "TCS"                 : "TCS.NS",
    "Infosys"             : "INFY.NS",
    "HDFC Bank"           : "HDFCBANK.NS",
    "SBI"                 : "SBIN.NS"
}

selected_name   = st.sidebar.selectbox("Select Stock", list(ticker_map.keys()))
selected_ticker = ticker_map[selected_name]
model_choice    = st.sidebar.radio(
    "Model", ["Stacking Ensemble", "Random Forest"])

st.title("📈 Stock Movement Predictor")
st.markdown("Predicts whether a stock will be **higher 5 days from now** "
            "using a stacking ensemble of RF + XGB + SVM.")
st.divider()

if st.button("🔮 Predict", type="primary"):
    with st.spinner("Fetching latest data and computing features..."):
        try:
            raw = yf.download(selected_ticker,
                              period="200d", auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            nifty = get_nifty_features()
            df    = build_features(raw.copy(), nifty)
            for col in models["feats"]:
                if col not in df.columns:
                    df[col] = 0
            
            ticker_col = f"ticker_{selected_ticker}"
            if ticker_col not in df.columns:
                df[ticker_col] = 1
            
            latest= df[models["feats"]].iloc[[-1]]
            latest_date = df.index[-1].strftime("%d %b %Y")



            latest_scaled = models["scaler"].transform(latest)

            model = (models["stack"] if model_choice == "Stacking Ensemble"
                     else models["rf"])
            pred       = model.predict(latest_scaled)[0]
            confidence = model.predict_proba(latest_scaled)[0][pred]

            col1, col2, col3 = st.columns(3)

            with col1:
                direction = "📈 UP" if pred == 1 else "📉 DOWN"
                color     = "green" if pred == 1 else "red"
                st.markdown(f"### Prediction")
                st.markdown(
                    f"{direction}",
                    unsafe_allow_html=True
                )
                st.caption(f"As of {latest_date}")

            with col2:
                st.markdown("### Confidence")
                st.metric("Model confidence",
                          f"{confidence*100:.1f}%")
                st.progress(float(confidence))

            with col3:
                st.markdown("### Current Price")
                current_price = raw["Close"].iloc[-1]
                prev_price    = raw["Close"].iloc[-2]
                change        = ((current_price - prev_price)
                                 / prev_price * 100)
                st.metric(
                    selected_name,
                    f"₹{current_price:.2f}",
                    f"{change:+.2f}% today"
                )

            st.divider()

            st.subheader("📊 Recent Price + EMA")
            chart_df = df[["ema_20", "ema_50"]].copy()
            chart_df["Close"] = raw["Close"]
            st.line_chart(chart_df.tail(60))

            st.subheader("🔢 Key Indicators (Latest)")
            ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)

            ind_col1.metric("RSI (14)",
                f"{latest['rsi'].values[0]:.1f}")
            ind_col2.metric("MACD Hist",
                f"{latest['macd_hist'].values[0]:.4f}")
            ind_col3.metric("BB Width",
                f"{latest['bb_width'].values[0]:.4f}")
            ind_col4.metric("Rel Strength 5d",
                f"{latest['rel_strength_5d'].values[0]:.4f}")

            st.divider()

            st.subheader("🧠 Why this prediction? (SHAP)")
            with st.spinner("Computing SHAP explanation..."):
                try:
                    explainer = shap.TreeExplainer(models["rf"])

                    shap_vals = explainer.shap_values(
                        pd.DataFrame(
                            latest_scaled,
                            columns=models["feats"]
                        )
                    )

                    # Handle both old and new SHAP versions
                    if isinstance(shap_vals, list):
                        sv_class1 = np.array(
                            shap_vals[-1]
                        ).flatten()
                    else:
                        shap_vals = np.array(shap_vals)
                        if shap_vals.ndim == 3:
                            sv_class1 = shap_vals[0, :, -1]
                        else:
                            sv_class1 = shap_vals.flatten()

                    expected = explainer.expected_value
                    if isinstance(expected, (list, np.ndarray)):
                        base_value = float(
                            np.array(expected).flatten()[-1]
                        )
                    else:
                        base_value = float(expected)

                    explanation = shap.Explanation(
                        values=sv_class1,
                        base_values=base_value,
                        data = np.array(latest_scaled).flatten(),
                        feature_names = models["feats"]
                    )
                    fig = plt.figure(figsize=(8, 5))
                    shap.plots.waterfall(explanation, max_display=12, show=False)
                    plt.tight_layout()

                    col1, col2, col3 = st.columns([0.5, 4, 0.5])
                    with col2:
                        st.pyplot(fig, use_container_width=True)
                    plt.close()
                
                except Exception as e:
                    st.error(f"SHAP error: {e}")
                    st.exception(e)


        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

st.divider()
st.subheader("📋 Model Performance")

col_a, col_b = st.columns(2)

with col_a:
    try:
        metrics = pd.read_csv("reports/model_metrics.csv")
        st.dataframe(metrics, use_container_width=True)
    except:
        st.info("Run evaluate.py first to generate metrics")

with col_b:
    try:
        st.image("reports/roc_curve.png",
                 caption="ROC Curves", use_container_width=True)
    except:
        st.info("ROC curve not found — run Phase 4 first")

with st.expander("🔍 SHAP Feature Importance"):
    try:
        # use RF feature importance instead of SHAP png
        import plotly.express as px
        
        feat_imp = pd.DataFrame({
            "Feature"   : models["feats"],
            "Importance": models["rf"].feature_importances_
        }).sort_values("Importance", ascending=False).head(15)

        fig = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Feature Importances — Random Forest",
            color="Importance",
            color_continuous_scale="Blues",
            height=400        # ← controls size
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
