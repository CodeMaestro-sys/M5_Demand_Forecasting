import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="M5 Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "m5_demand_forecasting_model.joblib"
FORECAST_PATH = BASE_DIR / "m5_28_day_forecast.csv"
HISTORICAL_PATH = BASE_DIR / "m5_historical_sample_small.csv"


# ============================================================
# CUSTOM CSS
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
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .info-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA / MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_forecast():
    df = pd.read_csv(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_historical():
    df = pd.read_csv(HISTORICAL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

missing_files = []

if not MODEL_PATH.exists():
    missing_files.append(MODEL_PATH.name)

if not FORECAST_PATH.exists():
    missing_files.append(FORECAST_PATH.name)

if not HISTORICAL_PATH.exists():
    missing_files.append(HISTORICAL_PATH.name)


if missing_files:

    st.error("Required project files are missing.")

    st.write("Please make sure the following files are present:")

    for file in missing_files:
        st.code(file)

    st.stop()


# ============================================================
# LOAD ARTIFACTS
# ============================================================

artifact = load_model()

forecast_df = load_forecast()
historical_df = load_historical()

model = artifact["model"]
feature_columns = artifact["feature_columns"]
items = artifact["items"]
stores = artifact["stores"]
forecast_horizon = artifact["forecast_horizon"]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📦 M5 Demand Forecasting")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Historical Demand",
        "🔮 Forecast",
        "📈 Model Performance",
        "🔍 Model Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model")

st.sidebar.write("HistGradientBoosting")

st.sidebar.write(
    f"**Forecast Horizon:** {forecast_horizon} days"
)

st.sidebar.write(
    f"**Series:** {len(items) * len(stores):,}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "M5 Walmart Demand Forecasting Project"
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">M5 Demand Forecasting</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Daily item-store demand forecasting using machine learning"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        This project develops a global machine learning model for
        forecasting daily demand across multiple Walmart item-store
        combinations using the M5 forecasting dataset.

        The model uses historical demand patterns, lag features,
        rolling statistics, calendar features, and item/store identity
        to generate a 28-day recursive forecast.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Stores",
            f"{len(stores)}"
        )

    with col2:
        st.metric(
            "Items",
            f"{len(items)}"
        )

    with col3:
        st.metric(
            "Series",
            f"{len(items) * len(stores):,}"
        )

    with col4:
        st.metric(
            "Forecast Horizon",
            f"{forecast_horizon} days"
        )

    with col5:
        st.metric(
            "Model",
            "HGB"
        )

    st.markdown(
        '<div class="section-title">Project Pipeline</div>',
        unsafe_allow_html=True
    )

    pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4, pipeline_col5 = (
        st.columns(5)
    )

    with pipeline_col1:
        st.info("📥\n\nHistorical Data")

    with pipeline_col2:
        st.info("⚙️\n\nFeature Engineering")

    with pipeline_col3:
        st.info("🤖\n\nML Model")

    with pipeline_col4:
        st.info("🔁\n\nRecursive Forecast")

    with pipeline_col5:
        st.info("🔮\n\n28-Day Prediction")

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Dataset Scope</div>',
        unsafe_allow_html=True
    )

    dataset_col1, dataset_col2 = st.columns(2)

    with dataset_col1:

        st.markdown(
            """
            **Modeling scope**

            - 10 Walmart stores
            - 100 items
            - 1,000 item-store series
            - Daily observations
            - 28-day forecast horizon
            """
        )

    with dataset_col2:

        st.markdown(
            """
            **Model features**

            - Lag 1, 7, 14 and 28 days
            - 7, 14 and 28-day rolling means
            - Day of week
            - Month and year
            - Day of month
            - Weekend indicator
            - Event indicator
            - SNAP indicators
            - Item and store identity
            """
        )

    # --------------------------------------------------------
    # FORECAST RANGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Generated Forecast</div>',
        unsafe_allow_html=True
    )

    start_date = forecast_df["date"].min()
    end_date = forecast_df["date"].max()

    st.success(
        f"The deployed model contains forecasts from "
        f"**{start_date.strftime('%d %b %Y')}** to "
        f"**{end_date.strftime('%d %b %Y')}**."
    )


# ============================================================
# PAGE 2 — HISTORICAL DEMAND
# ============================================================

elif page == "📊 Historical Demand":

    st.markdown(
        '<div class="main-title">Historical Demand Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Explore historical demand for individual item-store combinations."
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        selected_store = st.selectbox(
            "Select Store",
            stores
        )

    with col2:

        selected_item = st.selectbox(
            "Select Item",
            items
        )

    selected_history = historical_df[
        (historical_df["store_id"] == selected_store)
        &
        (historical_df["item_id"] == selected_item)
    ].copy()

    if selected_history.empty:

        st.warning(
            "No historical data is available for this selection."
        )

        st.stop()

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    avg_demand = selected_history["sales"].mean()
    max_demand = selected_history["sales"].max()

    zero_percentage = (
        selected_history["sales"].eq(0).mean() * 100
    )

    observation_count = len(selected_history)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Daily Demand",
            f"{avg_demand:.2f}"
        )

    with col2:
        st.metric(
            "Maximum Demand",
            f"{max_demand:.0f}"
        )

    with col3:
        st.metric(
            "Zero-Demand Days",
            f"{zero_percentage:.1f}%"
        )

    with col4:
        st.metric(
            "Observations",
            f"{observation_count:,}"
        )

    # --------------------------------------------------------
    # HISTORICAL CHART
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Historical Sales</div>',
        unsafe_allow_html=True
    )

    fig = px.line(
        selected_history,
        x="date",
        y="sales",
        title=f"{selected_item} — {selected_store}",
        labels={
            "date": "Date",
            "sales": "Daily Sales"
        }
    )

    fig.update_layout(
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RECENT DATA
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Recent Observations</div>',
        unsafe_allow_html=True
    )

    recent_data = (
        selected_history
        .sort_values("date", ascending=False)
        .head(20)
        .copy()
    )

    recent_data = recent_data[
        ["date", "sales"]
    ]

    st.dataframe(
        recent_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — FORECAST
# ============================================================

elif page == "🔮 Forecast":

    st.markdown(
        '<div class="main-title">28-Day Demand Forecast</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Explore the recursive forecasts generated by the final model."
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        selected_store = st.selectbox(
            "Select Store",
            stores,
            key="forecast_store"
        )

    with col2:

        selected_item = st.selectbox(
            "Select Item",
            items,
            key="forecast_item"
        )

    selected_forecast = forecast_df[
        (forecast_df["store_id"] == selected_store)
        &
        (forecast_df["item_id"] == selected_item)
    ].copy()

    selected_forecast = selected_forecast.sort_values("date")

    if selected_forecast.empty:

        st.warning(
            "No forecast is available for this item-store combination."
        )

        st.stop()

    # --------------------------------------------------------
    # FORECAST SUMMARY
    # --------------------------------------------------------

    total_forecast = selected_forecast["prediction"].sum()

    average_forecast = selected_forecast["prediction"].mean()

    maximum_forecast = selected_forecast["prediction"].max()

    peak_date = selected_forecast.loc[
        selected_forecast["prediction"].idxmax(),
        "date"
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Forecast",
            f"{total_forecast:.2f}"
        )

    with col2:
        st.metric(
            "Average Daily Forecast",
            f"{average_forecast:.2f}"
        )

    with col3:
        st.metric(
            "Peak Forecast",
            f"{maximum_forecast:.2f}"
        )

    with col4:
        st.metric(
            "Peak Date",
            peak_date.strftime("%d %b")
        )

    # --------------------------------------------------------
    # HISTORICAL + FORECAST
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Historical vs Forecast</div>',
        unsafe_allow_html=True
    )

    historical_selection = historical_df[
        (historical_df["store_id"] == selected_store)
        &
        (historical_df["item_id"] == selected_item)
    ].copy()

    # Show the last 90 historical observations
    historical_selection = (
        historical_selection
        .sort_values("date")
        .tail(90)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical_selection["date"],
            y=historical_selection["sales"],
            mode="lines",
            name="Historical Demand"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=selected_forecast["date"],
            y=selected_forecast["prediction"],
            mode="lines+markers",
            name="Forecast"
        )
    )

    fig.update_layout(
        title=f"{selected_item} — {selected_store}",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode="x unified",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Forecast Values</div>',
        unsafe_allow_html=True
    )

    forecast_table = selected_forecast[
        ["date", "prediction"]
    ].copy()

    forecast_table["prediction"] = forecast_table[
        "prediction"
    ].round(2)

    forecast_table.columns = [
        "Date",
        "Predicted Demand"
    ]

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = forecast_table.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_data,
        file_name=(
            f"{selected_item}_{selected_store}_forecast.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Evaluation of the final forecasting model against a seasonal baseline."
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    performance_data = pd.DataFrame(
        {
            "Model": [
                "Seasonal Naive",
                "HistGradientBoosting"
            ],
            "MAE": [
                1.052714,
                0.875791
            ],
            "RMSE": [
                2.554743,
                1.791693
            ]
        }
    )

    st.markdown(
        '<div class="section-title">Validation Results</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        performance_data.round(4),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MAE CHART
    # --------------------------------------------------------

    fig_mae = px.bar(
        performance_data,
        x="Model",
        y="MAE",
        title="Mean Absolute Error",
        text_auto=".3f"
    )

    fig_mae.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_mae,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RMSE CHART
    # --------------------------------------------------------

    fig_rmse = px.bar(
        performance_data,
        x="Model",
        y="RMSE",
        title="Root Mean Squared Error",
        text_auto=".3f"
    )

    fig_rmse.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    mae_improvement = (
        (
            performance_data.loc[0, "MAE"]
            -
            performance_data.loc[1, "MAE"]
        )
        /
        performance_data.loc[0, "MAE"]
    ) * 100

    rmse_improvement = (
        (
            performance_data.loc[0, "RMSE"]
            -
            performance_data.loc[1, "RMSE"]
        )
        /
        performance_data.loc[0, "RMSE"]
    ) * 100

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "MAE Reduction",
            f"{mae_improvement:.1f}%"
        )

    with col2:

        st.metric(
            "RMSE Reduction",
            f"{rmse_improvement:.1f}%"
        )

    # --------------------------------------------------------
    # MULTI-WINDOW VALIDATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Multi-Window Validation</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        In addition to the final 7-day validation window, the model was
        evaluated across four historical 7-day validation windows.
        This was used to check whether model performance was reasonably
        consistent across different periods.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean MAE",
            "0.856"
        )

    with col2:
        st.metric(
            "MAE Std",
            "0.020"
        )

    with col3:
        st.metric(
            "Mean RMSE",
            "1.861"
        )

    with col4:
        st.metric(
            "RMSE Std",
            "0.095"
        )

    st.info(
        "The relatively small MAE standard deviation across the "
        "validation windows indicates reasonably consistent MAE "
        "performance across the tested periods."
    )


# ============================================================
# PAGE 5 — MODEL INSIGHTS
# ============================================================

elif page == "🔍 Model Insights":

    st.markdown(
        '<div class="main-title">Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Understanding what the model learns and where it struggles."
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Feature Importance</div>',
        unsafe_allow_html=True
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": [
                "rolling_mean_28",
                "item_id",
                "rolling_mean_14",
                "day_of_week",
                "rolling_mean_7",
                "lag_14",
                "lag_28",
                "lag_7",
                "lag_1",
                "month",
                "year",
                "day_of_month",
                "store_id",
                "is_event",
                "snap_CA",
                "snap_TX",
                "snap_WI",
                "is_weekend"
            ],
            "Importance": [
                0.38,
                0.10,
                0.09,
                0.08,
                0.08,
                0.06,
                0.05,
                0.05,
                0.05,
                0.04,
                0.03,
                0.02,
                0.02,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01
            ]
        }
    )

    feature_importance = (
        feature_importance
        .sort_values("Importance", ascending=True)
    )

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Relative Feature Importance",
        text_auto=".2f"
    )

    fig.update_layout(
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Longer-term demand history, particularly the 28-day rolling "
        "mean, is one of the strongest signals used by the model. "
        "Item identity and weekly demand patterns also contribute "
        "meaningfully."
    )

    # --------------------------------------------------------
    # ERROR BY DEMAND LEVEL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Error by Demand Level</div>',
        unsafe_allow_html=True
    )

    demand_error = pd.DataFrame(
        {
            "Demand Level": [
                "Zero",
                "Low",
                "Medium",
                "High",
                "Very High"
            ],
            "MAE": [
                0.539260,
                0.840851,
                1.937647,
                3.668272,
                10.734370
            ],
            "RMSE": [
                0.934432,
                1.104876,
                2.320524,
                4.125451,
                12.328874
            ],
            "Mean Actual": [
                0.000000,
                1.276781,
                3.634033,
                7.195489,
                16.584416
            ],
            "Count": [
                4522,
                1839,
                429,
                133,
                77
            ]
        }
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            demand_error,
            x="Demand Level",
            y="MAE",
            title="MAE by Demand Level",
            text_auto=".2f"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            demand_error,
            x="Demand Level",
            y="RMSE",
            title="RMSE by Demand Level",
            text_auto=".2f"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.write(
        """
        Prediction error increases as actual demand becomes larger.
        The model performs substantially better on zero and low-demand
        observations than on very high-demand observations.
        """
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Current Limitations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        - The current model was trained on the selected 100 items
          across 10 stores.
        - The deployed forecast covers a fixed 28-day horizon.
        - Recursive forecasting means prediction errors can propagate
          through later forecast steps.
        - Sudden demand spikes remain difficult for the model to capture.
        - Event and SNAP-related features contributed relatively little
          in the tested validation setup.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "M5 Demand Forecasting • Machine Learning Project"
)