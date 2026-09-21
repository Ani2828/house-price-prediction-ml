"""
Professional Streamlit application for House Price Prediction.
"""

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_house_price
from src.logger import log_prediction

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 700;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center;
    }

    .metric-title {
        font-size: 14px;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🏠 House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning powered house price estimation using a '
    'tuned Gradient Boosting model.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# Sidebar — Model Information
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write(
        "This application uses a tuned "
        "**Gradient Boosting Regressor**."
    )

    st.divider()

    st.subheader("Validated Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MAE", "1.9301")
        st.metric("R²", "0.9039")

    with col2:
        st.metric("RMSE", "2.6550")
        st.metric("CV", "5-Fold")

    st.divider()

    st.caption(
        "The model was trained using the Boston Housing dataset."
    )

    st.caption(
        "Predicted values are expressed in thousands of "
        "US dollars (k$)."
    )


# ============================================================
# Property Information
# ============================================================

st.subheader("🏡 Property Characteristics")

st.caption(
    "Enter the characteristics of the property below."
)

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

with col1:

    crim = st.number_input(
        "Crime Rate (CRIM)",
        min_value=0.0,
        max_value=100.0,
        value=0.0063,
        step=0.01,
        format="%.4f",
        help="Per-capita crime rate by town."
    )

    zn = st.number_input(
        "Residential Land (ZN)",
        min_value=0.0,
        max_value=100.0,
        value=18.0,
        step=1.0,
        help="Proportion of residential land zoned for large lots."
    )

    indus = st.number_input(
        "Industrial Area (INDUS)",
        min_value=0.0,
        max_value=30.0,
        value=2.31,
        step=0.1,
        format="%.2f",
        help="Proportion of non-retail business acres."
    )

    chas = st.selectbox(
        "Charles River Property (CHAS)",
        options=[0, 1],
        format_func=lambda x: (
            "Yes" if x == 1 else "No"
        ),
        help="Whether the property borders the Charles River."
    )

    nox = st.number_input(
        "NOX Concentration",
        min_value=0.0,
        max_value=1.0,
        value=0.538,
        step=0.001,
        format="%.3f",
        help="Nitric oxide concentration."
    )


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    rm = st.number_input(
        "Average Number of Rooms (RM)",
        min_value=1.0,
        max_value=15.0,
        value=6.575,
        step=0.1,
        format="%.3f",
        help="Average number of rooms per dwelling."
    )

    age = st.number_input(
        "Property Age (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.2,
        step=1.0,
        format="%.1f",
        help="Proportion of owner-occupied units built before 1940."
    )

    dis = st.number_input(
        "Distance to Employment Centers",
        min_value=0.0,
        max_value=20.0,
        value=4.09,
        step=0.1,
        format="%.3f",
        help="Weighted distance to employment centers."
    )

    rad = st.number_input(
        "Highway Accessibility (RAD)",
        min_value=1,
        max_value=24,
        value=1,
        step=1,
        help="Index of accessibility to radial highways."
    )


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    tax = st.number_input(
        "Property Tax Rate",
        min_value=100,
        max_value=800,
        value=296,
        step=1,
        help="Full-value property-tax rate per $10,000."
    )

    ptratio = st.number_input(
        "Pupil-Teacher Ratio",
        min_value=10.0,
        max_value=25.0,
        value=15.3,
        step=0.1,
        format="%.1f",
        help="Pupil-teacher ratio by town."
    )

    b = st.number_input(
        "Black Population Index (B)",
        min_value=0.0,
        max_value=400.0,
        value=396.90,
        step=1.0,
        format="%.2f",
        help="Dataset's B feature."
    )

    lstat = st.number_input(
        "Lower Status Population (%)",
        min_value=0.0,
        max_value=40.0,
        value=4.98,
        step=0.1,
        format="%.2f",
        help="Percentage of lower-status population."
    )


# ============================================================
# Prediction
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
)


if predict_button:

    features = {
        "crim": crim,
        "zn": zn,
        "indus": indus,
        "chas": chas,
        "nox": nox,
        "rm": rm,
        "age": age,
        "dis": dis,
        "rad": rad,
        "tax": tax,
        "ptratio": ptratio,
        "b": b,
        "lstat": lstat
    }

    try:

        prediction = predict_house_price(features)
        log_prediction(
            model_name="Gradient Boosting",
            prediction=prediction
        )
        st.success("Prediction generated successfully!")

        st.markdown(
            f"""<div class="prediction-card">
        <div class="metric-title">ESTIMATED HOUSE VALUE</div>
        <div class="prediction-value">${prediction:.2f}K</div>
        <div class="metric-title">Estimated value in thousands of US dollars</div>
        </div>""",
            unsafe_allow_html=True
   )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# About section
# ============================================================

st.divider()

with st.expander("📊 About the Model"):

    st.write(
        """
        This project uses a **tuned Gradient Boosting Regressor**
        for house price prediction.

        The model was selected after comparing multiple regression
        algorithms using 5-fold cross-validation and performing
        hyperparameter tuning.
        """
    )

    st.write("**Final test-set performance:**")

    performance_data = {
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": ["1.9301", "2.6550", "0.9039"]
    }

    st.table(performance_data)


# ============================================================
# Footer
# ============================================================

st.caption(
    "House Price Prediction • Machine Learning Project • "
    "Python + scikit-learn + Streamlit"
)