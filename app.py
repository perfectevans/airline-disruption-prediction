import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Airline Disruption Prediction",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/modeling_dataset.parquet")
MONTHLY_DATA_PATH = Path("data/processed/final_monthly_operational_summary_data.parquet")

RF_MODEL_PATH = Path("models/baseline_rf.pkl")
LGBM_MODEL_PATH = Path("models/baseline_lgbm.pkl")
CB_MODEL_PATH = Path("models/baseline_cb.pkl")
STACKED_META_PATH = Path("models/stacked_metadata.pkl")

TREND_PLOT_PATH = Path("reports/figures/monthly_disruption_trend.png")
MODEL_PLOT_PATH = Path("reports/figures/model_comparison_auc.png")
SHAP_PLOT_PATH = Path("reports/figures/shap_lgbm_summary.png")


# -----------------------------
# Caching
# -----------------------------
@st.cache_data
def load_data():
    model_df = pd.read_parquet(DATA_PATH)
    monthly_df = pd.read_parquet(MONTHLY_DATA_PATH)
    model_df["YearMonth"] = pd.to_datetime(model_df["YearMonth"])
    monthly_df["YearMonth"] = pd.to_datetime(monthly_df["YearMonth"])
    return model_df, monthly_df


@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_MODEL_PATH)
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    cb_model = joblib.load(CB_MODEL_PATH)
    stacked_meta = joblib.load(STACKED_META_PATH)
    return rf_model, lgbm_model, cb_model, stacked_meta


# -----------------------------
# Helper functions
# -----------------------------
def build_latest_feature_row(model_df: pd.DataFrame, airline: str, origin: str) -> pd.DataFrame | None:
    """
    Select the most recent row for a given airline-airport pair.
    This assumes feature engineering has already been done in model_data.parquet.
    """
    subset = model_df[
        (model_df["Reporting_Airline"] == airline) &
        (model_df["Origin"] == origin)
    ].sort_values("YearMonth")

    if subset.empty:
        return None

    return subset.tail(1).copy()


def predict_stacked(row_df: pd.DataFrame, rf_model, lgbm_model, cb_model, threshold: float):
    """
    Generate stacked prediction using simple average of model probabilities.
    """
    # Remove columns that should not be fed into model
    drop_cols = [col for col in ["HighDisruptionMonth", "YearMonth"] if col in row_df.columns]
    X_input = row_df.drop(columns=drop_cols)

    rf_prob = rf_model.predict_proba(X_input)[:, 1][0]
    lgbm_prob = lgbm_model.predict_proba(X_input)[:, 1][0]
    cb_prob = cb_model.predict_proba(X_input)[:, 1][0]

    stacked_prob = float(np.mean([rf_prob, lgbm_prob, cb_prob]))
    stacked_pred = int(stacked_prob >= threshold)

    return {
        "rf_prob": rf_prob,
        "lgbm_prob": lgbm_prob,
        "cb_prob": cb_prob,
        "stacked_prob": stacked_prob,
        "stacked_pred": stacked_pred
    }


# -----------------------------
# Load resources
# -----------------------------
model_df, monthly_df = load_data()
rf_model, lgbm_model, cb_model, stacked_meta = load_models()
stack_threshold = stacked_meta["threshold"]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "EDA Visuals", "Make Prediction"]
)

# -----------------------------
# Page: Overview
# -----------------------------
if page == "Project Overview":
    st.title("✈️ Airline Disruption Prediction")
    st.markdown(
        """
        This Streamlit app demonstrates a machine learning system for predicting
        **high-disruption months** for airline–airport combinations.

        **Final selected approach:** Baseline stacked ensemble model  
        **Primary metric:** ROC-AUC  
        **Goal:** Identify airline–airport periods with elevated disruption risk
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Years Covered", "2023–2025")
    col2.metric("Final Model", "Stacked Ensemble")
    col3.metric("Best ROC-AUC", "0.8047")

    st.subheader("Project Summary")
    st.write(
        """
        The model uses temporal, operational, and historical features such as:
        - month of operation
        - lagged disruption rates
        - rolling disruption patterns
        - flight volume
        - airline and airport context
        """
    )

    st.subheader("Why it matters")
    st.write(
        """
        Predicting disruption risk can support proactive planning, better resource allocation,
        and improved operational decision-making for airlines and airports.
        """
    )

# -----------------------------
# Page: EDA Visuals
# -----------------------------
elif page == "EDA Visuals":
    st.title("📊 EDA Visuals")

    if TREND_PLOT_PATH.exists():
        st.subheader("Monthly Disruption Trend")
        st.image(str(TREND_PLOT_PATH), use_container_width=True)

    if MODEL_PLOT_PATH.exists():
        st.subheader("Model Comparison")
        st.image(str(MODEL_PLOT_PATH), use_container_width=True)

    if SHAP_PLOT_PATH.exists():
        st.subheader("SHAP Summary Plot")
        st.image(str(SHAP_PLOT_PATH), use_container_width=True)

    st.subheader("Dataset Snapshot")
    st.dataframe(monthly_df.head(20), use_container_width=True)

# -----------------------------
# Page: Prediction
# -----------------------------
elif page == "Make Prediction":
    st.title("🔮 Predict High Disruption Risk")

    airlines = sorted(model_df["Reporting_Airline"].dropna().unique().tolist())
    airports = sorted(model_df["Origin"].dropna().unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        selected_airline = st.selectbox("Select Airline", airlines)

    with col2:
        selected_airport = st.selectbox("Select Origin Airport", airports)

    if st.button("Run Prediction"):
        input_row = build_latest_feature_row(model_df, selected_airline, selected_airport)

        if input_row is None:
            st.error("No feature row found for that airline-airport combination.")
        else:
            # 1. Date Logic: Identify the context
            data_date = pd.to_datetime(input_row["YearMonth"].values[0])
            prediction_date = data_date + pd.DateOffset(months=1)
            
            st.info(f"📅 **Input Data:** {data_date.strftime('%B %Y')} | 🎯 **Target Prediction:** {prediction_date.strftime('%B %Y')}")

            # 2. Run Prediction
            results = predict_stacked(
                input_row,
                rf_model=rf_model,
                lgbm_model=lgbm_model,
                cb_model=cb_model,
                threshold=stack_threshold
            )

            # 3. Risk Level Color Bar Logic
            prob = results["stacked_prob"]
            if prob < 0.4:
                color = "green"
                level = "LOW"
            elif prob < 0.6:
                color = "orange"
                level = "MEDIUM"
            else:
                color = "red"
                level = "HIGH"

            # 4. Display Result with Color Bar
            st.subheader("Prediction Result")
            
            # Create the custom HTML progress bar
            st.markdown(f"""
                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px;">
                    <div style="background-color: {color}; width: {prob*100}%; height: 25px; border-radius: 8px; text-align: center; color: white;">
                        <b>{prob*100:.1f}% Risk</b>
                    </div>
                </div>
                <p style="text-align: center;"><b>Risk Level: {level}</b></p>
            """, unsafe_allow_html=True)

            risk_label = "⚠️ High Disruption Risk" if results["stacked_pred"] == 1 else "✅ Lower Disruption Risk"
            
            c_res1, c_res2 = st.columns(2)
            c_res1.metric("Predicted Class", risk_label)
            c_res2.metric("Stacked Probability", f"{prob:.3f}")

            # 5. Base Model Probabilities
            st.subheader("Base Model Probabilities")
            c1, c2, c3 = st.columns(3)
            c1.metric("Random Forest", f"{results['rf_prob']:.3f}")
            c2.metric("LightGBM", f"{results['lgbm_prob']:.3f}")
            c3.metric("CatBoost", f"{results['cb_prob']:.3f}")

            st.subheader("Interpretation")
            if results["stacked_pred"] == 1:
                st.warning(
                    f"""
                    **High Alert:** The ensemble predicts a significant chance of operational disruption 
                    for {prediction_date.strftime('%B %Y')}. Consider resource buffer allocation.
                    """
                )
            else:
                st.success(
                    f"""
                    **Stable Outlook:** Operational patterns suggest {prediction_date.strftime('%B %Y')} 
                    will remain within normal performance boundaries.
                    """
                )

            st.subheader("Latest Feature Row Used")
            st.dataframe(input_row, width="stretch")
    