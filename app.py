import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Geopolitical Shocks and Commodity Markets",
    page_icon="🛢️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f8f6fb; }
    .main p { font-size: 1.05rem; line-height: 1.8; color: #2d004b; }
    </style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="#f8f6fb",
        plot_bgcolor="#f8f6fb",
        font=dict(color="#2d004b"),
    )
    fig.update_xaxes(gridcolor="#ddd8e8", linecolor="#2d004b")
    fig.update_yaxes(gridcolor="#ddd8e8", linecolor="#2d004b")
    return fig


# === DATA LOADING ===

# Streamlit reruns the entire script from top to bottom on every user interaction
# (a slider move, a button click, even resizing the window). Without caching, that
# means reading all 9 CSV files from disk on every single rerun. @st.cache_data tells
# Streamlit to run the function once, store the returned dataframes in memory, and
# hand back the cached result on every subsequent rerun — so disk reads happen only
# once per session no matter how many times the page updates.
@st.cache_data
def load_data():
    df = pd.read_csv("data/capstone_master_dataset_v2.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    predictions = pd.read_csv("data/test_predictions.csv")
    predictions["date"] = pd.to_datetime(predictions["date"])

    part1_results = pd.read_csv("data/part1_results.csv")
    part2_results = pd.read_csv("data/part2_results.csv")
    part1_wfv = pd.read_csv("data/part1_walk_forward.csv")
    part2_wfv = pd.read_csv("data/part2_walk_forward.csv")
    unified = pd.read_csv("data/unified_results.csv")

    imp1 = pd.read_csv("data/part1_feature_importance.csv")
    imp2 = pd.read_csv("data/part2_feature_importance.csv")

    return df, predictions, part1_results, part2_results, part1_wfv, part2_wfv, unified, imp1, imp2


df, predictions, part1_results, part2_results, part1_wfv, part2_wfv, unified, imp1, imp2 = load_data()


# === PAGE HEADER ===

st.title("Geopolitical Shocks and Commodity Markets")
st.subheader("Predicting WTI Crude Oil Price Movements Using Machine Learning")
st.markdown(
    "**Author:** Milo Joseph Gaida Barlafante | **Institution:** Imperial College Business School | **Supervisor:** Professor Vikesh Koul"
)
st.markdown("---")


# === EXECUTIVE SUMMARY ===

st.header("Executive Summary")
st.write("""
This capstone investigates how geopolitical shocks propagate across global commodity markets
and whether machine learning models can detect and predict their impact on WTI crude oil prices.
Using a dataset spanning 1986 to 2025 with 14,609 observations across 43 variables, the project
builds two distinct modelling frameworks that together answer the core research question.

Part 1 establishes that tree based models can predict WTI price levels with high accuracy (R2 of 0.97),
but this performance is largely driven by price momentum rather than geopolitical signal.
Part 2 reframes the target variable to measure daily price deviations from trend, forcing the model
to explain genuine price surprises. The result is a more honest finding: geopolitical features
explain approximately 10% of daily price deviations, with the signal strongest during sustained
supply side shocks such as the COVID crash and the Ukraine invasion.
""")
st.markdown("---")


# === DATASET AND METHODOLOGY ===

st.header("Dataset and Methodology")
st.write("""
The analysis draws on 10 data sources combined into a single master dataset of 14,609 daily
observations spanning January 1986 to December 2025. Key sources include FRED for WTI and
Henry Hub natural gas prices, yfinance for Brent crude, gold, copper, VIX and the US Dollar
Index, the Caldara and Iacoviello Geopolitical Risk index, and ACLED conflict event data
covering political violence by country and year.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Observations", "14,609")
with col2:
    st.metric("Date Range", "1986 to 2025")
with col3:
    st.metric("Features", "43 columns")

st.markdown("---")


# === EDA ===

st.header("Exploratory Data Analysis")
st.write(
    "WTI crude oil prices have passed through eight distinct price eras since 1986, each shaped by a combination of supply decisions, demand shocks, and geopolitical events."
)

fig1 = px.line(
    df,
    x="observation_date",
    y="wti_price",
    title="WTI Crude Oil Price 1986 to 2025",
    color_discrete_sequence=["#7b2d8b"],
)
fig1.update_layout(xaxis_title="Date", yaxis_title="Price (USD per barrel)")
fig1 = style_fig(fig1)
st.plotly_chart(fig1, use_container_width=True)
st.caption(
    "WTI prices reflect eight distinct market regimes driven by supply shocks, demand cycles and geopolitical events."
)
st.markdown("---")

# Cross-commodity performance
st.subheader("Cross Commodity Performance Since 2000")
st.write(
    "Normalising all four commodities to 100 at January 2000 reveals dramatically different long run trajectories. Gold has been the strongest performer, while WTI and copper are cyclical and mean reverting."
)

df_2000 = df[df["observation_date"] >= "2000-01-01"].copy()


def normalize_to_100(series):
    first_valid = series.dropna().iloc[0]
    return (series / first_valid) * 100


df_2000["wti_norm"] = normalize_to_100(df_2000["wti_price"])
df_2000["gold_norm"] = normalize_to_100(df_2000["gold_price"])
df_2000["natgas_norm"] = normalize_to_100(df_2000["natgas_price"])
df_2000["copper_norm"] = normalize_to_100(df_2000["copper_price"])

# cap natgas spikes so they don't distort the chart
df_2000["natgas_norm"] = df_2000["natgas_norm"].clip(upper=500)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=df_2000["observation_date"],
        y=df_2000["wti_norm"],
        name="WTI Oil",
        line=dict(color="#2d004b", width=1.5),
    )
)
fig2.add_trace(
    go.Scatter(
        x=df_2000["observation_date"],
        y=df_2000["gold_norm"],
        name="Gold",
        line=dict(color="#7b2d8b", width=1.5),
    )
)
fig2.add_trace(
    go.Scatter(
        x=df_2000["observation_date"],
        y=df_2000["natgas_norm"],
        name="Natural Gas",
        line=dict(color="#c06ec0", width=1.5),
    )
)
fig2.add_trace(
    go.Scatter(
        x=df_2000["observation_date"],
        y=df_2000["copper_norm"],
        name="Copper",
        line=dict(color="#e8b4e8", width=1.5),
    )
)
fig2.update_layout(
    title="Gold Has Been the Strongest Long Run Performer",
    yaxis_title="Indexed Price (Jan 2000 = 100)",
    xaxis_title="Date",
)
fig2 = style_fig(fig2)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "Gold has outperformed all commodities since 2000. WTI and copper are cyclical with no long run trend."
)
st.markdown("---")

# Market regime clustering
st.subheader("Market Regime Clustering")
st.write("""
KMeans clustering independently identified five distinct market regimes across the full dataset
without being told anything about historical events. The algorithm correctly flagged the Gulf War,
9/11, Iraq invasion, 2008 financial crisis, and COVID crash as crisis periods.
Use the legend to toggle individual regimes on and off.
""")

# KMeans on ~14k rows with StandardScaler is the most expensive computation in this app.
# It is also deterministic (random_state=42), so the result is identical every time.
# @st.cache_data hashes the input dataframe — if df hasn't changed, Streamlit skips the
# fit entirely and returns the cached cluster assignments and regime boundaries straight
# from memory. Without this, every user interaction would re-run the full clustering.
@st.cache_data
def run_clustering(df):
    cluster_df = df.dropna(subset=["wti_price", "vix_close", "gpr_index"]).copy()
    cluster_features = [
        "wti_price",
        "wti_daily_return",
        "wti_rolling_mean_30",
        "vix_close",
        "dxy_index",
        "gpr_index",
        "gpr_threats",
        "gpr_acts",
        "total_violence_events",
        "total_fatalities",
        "mena_violence_events",
    ]
    cluster_data = cluster_df[cluster_features].fillna(
        cluster_df[cluster_features].median()
    )
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_df["cluster"] = kmeans.fit_predict(cluster_scaled)

    cluster_names = {
        0: "Low Price High Stress",
        1: "High Price Calm",
        2: "Crisis Spike",
        3: "Stable Low Price",
        4: "High Conflict High Price",
    }
    cluster_colors = {0: "#f4a460", 1: "#2ecc71", 2: "#e74c3c", 3: "#3498db", 4: "#9b59b6"}

    regime_periods = []
    current_cluster = cluster_df["cluster"].iloc[0]
    start_date = cluster_df["observation_date"].iloc[0]
    for _, row in cluster_df.iterrows():
        if row["cluster"] != current_cluster:
            regime_periods.append(
                {
                    "start": start_date,
                    "end": row["observation_date"],
                    "cluster": current_cluster,
                }
            )
            current_cluster = row["cluster"]
            start_date = row["observation_date"]
    regime_periods.append(
        {
            "start": start_date,
            "end": cluster_df["observation_date"].iloc[-1],
            "cluster": current_cluster,
        }
    )
    regime_df = pd.DataFrame(regime_periods)

    return cluster_df, regime_df, cluster_names, cluster_colors


cluster_df, regime_df, cluster_names, cluster_colors = run_clustering(df)

y_max = cluster_df["wti_price"].max() * 1.1
fig3 = go.Figure()
for _, row in regime_df.iterrows():
    fig3.add_trace(
        go.Scatter(
            x=[str(row["start"]), str(row["start"]), str(row["end"]), str(row["end"])],
            y=[0, y_max, y_max, 0],
            fill="toself",
            fillcolor=cluster_colors[row["cluster"]],
            opacity=0.3,
            line=dict(width=0),
            mode="lines",
            name=cluster_names[row["cluster"]],
            legendgroup=str(row["cluster"]),
            showlegend=False,
        )
    )
fig3.add_trace(
    go.Scatter(
        x=cluster_df["observation_date"],
        y=cluster_df["wti_price"],
        mode="lines",
        line=dict(color="#7b2d8b", width=1.5),
        name="WTI Price",
    )
)
for cid in range(5):
    fig3.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=cluster_colors[cid], size=12, symbol="square"),
            name=cluster_names[cid],
            legendgroup=str(cid),
            showlegend=True,
        )
    )
fig3.update_layout(
    title="KMeans Market Regime Clustering: WTI Price 1990 to 2025",
    yaxis=dict(range=[-10, y_max]),
    xaxis_title="Date",
    yaxis_title="WTI Price (USD per barrel)",
    legend=dict(x=1.01, y=1),
)
fig3 = style_fig(fig3)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Five distinct market regimes identified by unsupervised clustering. Crisis Spike periods correctly flag Gulf War, 9/11, Iraq invasion and COVID."
)
st.markdown("---")


# === PART 1 ===

st.header("Part 1: Predicting WTI Price Level")
st.write("""
Part 1 builds a set of machine learning models to predict the daily WTI price level.
The dataset was split chronologically with 80% for training (1986 to 2017) and 20% for testing (2018 to 2025).
Seven models were evaluated using TimeSeriesSplit cross validation and test set metrics.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best Model", "Gradient Boosting Tuned")
with col2:
    st.metric("Test R2", "0.974")
with col3:
    st.metric("Test RMSE", "$2.77")

st.markdown("---")

# Part 1: model comparison
st.subheader("Model Comparison: Part 1")
st.write("Seven models were tested. Tree based models dramatically outperform linear approaches, confirming the relationship between features and WTI price is non linear.")

p1_plot = part1_results[part1_results["Model"] != "Linear Regression"].copy()

fig_p1_models = make_subplots(rows=1, cols=2,
    subplot_titles=["Test RMSE (lower is better)", "Test R2 (higher is better)"])

fig_p1_models.add_trace(go.Bar(
    x=p1_plot["Model"], y=p1_plot["Test RMSE"],
    marker_color="#7b2d8b", name="RMSE"
), row=1, col=1)

fig_p1_models.add_trace(go.Bar(
    x=p1_plot["Model"], y=p1_plot["Test R2"],
    marker_color="#c06ec0", name="R2"
), row=1, col=2)

fig_p1_models.update_layout(title="Part 1 Model Comparison", showlegend=False)
fig_p1_models = style_fig(fig_p1_models)
fig_p1_models.update_xaxes(tickangle=30)
st.plotly_chart(fig_p1_models, use_container_width=True)
st.markdown("---")

# Part 1: actual vs predicted
st.subheader("Part 1: Actual vs Predicted WTI Price (2018 to 2025)")
st.write("""
The tuned Gradient Boosting model tracks the actual WTI price almost perfectly on the test set.
However this performance is misleading. The dominant feature driving predictions is the 30 day
rolling average, meaning the model is following momentum rather than detecting geopolitical signal.
""")

fig_p1_pred = go.Figure()
fig_p1_pred.add_trace(go.Scatter(
    x=predictions["date"], y=predictions["actual_price"],
    name="Actual Price", line=dict(color="#2d004b", width=1.5)
))
fig_p1_pred.add_trace(go.Scatter(
    x=predictions["date"], y=predictions["predicted_price"],
    name="Predicted Price", line=dict(color="#c06ec0", width=1.5)
))
fig_p1_pred.update_layout(
    title="Part 1: Predicted vs Actual WTI Price — Model Driven by Momentum (2018 to 2025)",
    xaxis_title="Date", yaxis_title="WTI Price (USD per barrel)"
)
fig_p1_pred = style_fig(fig_p1_pred)
st.plotly_chart(fig_p1_pred, use_container_width=True)
st.caption("The model tracks prices almost perfectly but is driven by the 30 day rolling average, not geopolitical features.")
st.markdown("---")

# Part 1: walk forward validation
st.subheader("Part 1: Walk Forward Validation")
st.write("""
Walk forward validation tests the model on five distinct historical periods.
Performance varies dramatically across regimes — the model struggles most during the 2008 financial crisis
which was a demand shock rather than a geopolitical supply shock.
""")

fig_p1_wfv = make_subplots(rows=1, cols=2,
    subplot_titles=["RMSE by Fold (lower is better)", "R2 by Fold (higher is better)"])

fig_p1_wfv.add_trace(go.Bar(
    x=part1_wfv["Fold"], y=part1_wfv["RMSE"],
    marker_color="#7b2d8b", name="RMSE"
), row=1, col=1)

fig_p1_wfv.add_trace(go.Bar(
    x=part1_wfv["Fold"], y=part1_wfv["R2"],
    marker_color="#c06ec0", name="R2"
), row=1, col=2)

fig_p1_wfv.update_layout(title="Part 1 Walk Forward Validation", showlegend=False)
fig_p1_wfv = style_fig(fig_p1_wfv)
fig_p1_wfv.update_xaxes(tickangle=25)
st.plotly_chart(fig_p1_wfv, use_container_width=True)
st.markdown("---")


# === FEATURE IMPORTANCE ===

st.header("What Is Each Model Actually Learning?")
st.write("""
This comparison is the core of the project. Part 1 predicts the WTI price level
directly. The gradient boosting model achieves an R2 of 0.97 but the feature
importance tells an uncomfortable story: the 30 day rolling mean accounts for
roughly 85 percent of predictive power. The model is not reading geopolitics.
It is reading momentum.

Part 2 strips that momentum signal out by predicting the deviation from the rolling
average. With the trend removed, the feature importance landscape changes completely.
Geopolitical variables including GPR scores for Russia, Israel and Saudi Arabia, MENA
violence event counts, and lagged violence indicators all appear in the top 15. The
model R2 drops to 0.097 but that 10 percent is genuinely geopolitical signal, not
an artefact of autocorrelation.
""")

imp1_sorted = imp1.sort_values("importance", ascending=True)
imp2_sorted = imp2.sort_values("importance", ascending=True)

col_fi1, col_fi2 = st.columns(2)

with col_fi1:
    st.markdown("**Part 1: Price Level Model**")
    fig_fi1 = go.Figure()
    fig_fi1.add_trace(go.Bar(
        x=imp1_sorted["importance"],
        y=imp1_sorted["feature"],
        orientation="h",
        marker_color="#7b2d8b",
    ))
    fig_fi1.update_layout(
        title="Feature Importance: Part 1",
        xaxis_title="Importance",
        yaxis_title="",
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=40)
    )
    fig_fi1 = style_fig(fig_fi1)
    st.plotly_chart(fig_fi1, use_container_width=True)

with col_fi2:
    st.markdown("**Part 2: Deviation from Trend Model**")
    fig_fi2 = go.Figure()
    fig_fi2.add_trace(go.Bar(
        x=imp2_sorted["importance"],
        y=imp2_sorted["feature"],
        orientation="h",
        marker_color="#c06ec0",
    ))
    fig_fi2.update_layout(
        title="Feature Importance: Part 2",
        xaxis_title="Importance",
        yaxis_title="",
        height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=40)
    )
    fig_fi2 = style_fig(fig_fi2)
    st.plotly_chart(fig_fi2, use_container_width=True)

st.caption(
    "Left: wti rolling mean 30 dominates Part 1 at roughly 85 percent importance. "
    "Right: with momentum removed, GPR and conflict variables surface as genuine signals in Part 2."
)
st.markdown("---")


# === PART 2 ===

st.header("Part 2: Isolating the Geopolitical Signal")
st.write("""
Part 2 reframes the prediction target. Rather than forecasting the price level,
the model predicts how far today's WTI price sits above or below its 30 day rolling
average. This deviation variable strips out trend and momentum, leaving behind the
residual that geopolitical events might actually explain.

The best model is again gradient boosting tuned, achieving a test R2 of 0.097 and
an RMSE of 3.90 dollars. That 10 percent explained variance is modest, but it is
real. The signal is non linear and regime dependent: walk forward validation shows
positive R2 only in the COVID and Ukraine folds, the two periods defined by acute
supply side shocks. In calm markets, geopolitical noise does not move oil prices
consistently enough for a model trained on historical patterns to capture it.
""")

col_p2a, col_p2b, col_p2c = st.columns(3)
with col_p2a:
    st.metric("Best Model", "Gradient Boosting Tuned")
with col_p2b:
    st.metric("Test R2", "0.097")
with col_p2c:
    st.metric("Test RMSE", "$3.90")

st.markdown("---")

# Part 2: model comparison
st.subheader("Model Comparison: Part 2")
st.write("With momentum removed from the target variable, all models perform much more modestly. This is the honest performance floor when predicting genuine price surprises.")

p2_display = part2_results[~part2_results["Model"].isin(["Dummy", "Linear Regression"])].copy()

fig_p2_models = make_subplots(rows=1, cols=2,
    subplot_titles=["Test RMSE (lower is better)", "Test R2 (higher is better)"])

fig_p2_models.add_trace(go.Bar(
    x=p2_display["Model"], y=p2_display["Test RMSE"],
    marker_color="#7b2d8b", name="RMSE"
), row=1, col=1)

fig_p2_models.add_trace(go.Bar(
    x=p2_display["Model"], y=p2_display["Test R2"],
    marker_color="#c06ec0", name="R2"
), row=1, col=2)

fig_p2_models.update_layout(title="Part 2 Model Comparison", showlegend=False)
fig_p2_models = style_fig(fig_p2_models)
fig_p2_models.update_xaxes(tickangle=30)
st.plotly_chart(fig_p2_models, use_container_width=True)
st.markdown("---")

# Part 2: actual vs predicted deviation
st.subheader("Part 2: Actual vs Predicted Deviation from 30 Day Rolling Average")
st.write("""
The chart below plots actual and predicted deviations over the test period (2018 to 2025).
The model captures the direction and rough magnitude of spikes during major events
but undershoots the most extreme moves. This is expected behaviour for a tree based
model: it has seen crises before, but not at exactly this scale or timing.
""")

fig_p2_pred = go.Figure()
fig_p2_pred.add_trace(go.Scatter(
    x=predictions["date"],
    y=predictions["actual_deviation"],
    mode="lines",
    name="Actual Deviation",
    line=dict(color="#2d004b", width=1.5)
))
fig_p2_pred.add_trace(go.Scatter(
    x=predictions["date"],
    y=predictions["predicted_deviation"],
    mode="lines",
    name="Predicted Deviation",
    line=dict(color="#c06ec0", width=1.5, dash="dash")
))
fig_p2_pred.update_layout(
    title="WTI Price Deviation: Actual vs Predicted (Test Period 2018 to 2025)",
    xaxis_title="Date",
    yaxis_title="Deviation from 30 Day Rolling Mean (USD)",
    height=420,
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=20, r=20, t=50, b=40)
)
fig_p2_pred = style_fig(fig_p2_pred)
st.plotly_chart(fig_p2_pred, use_container_width=True)
st.markdown("---")

# Part 2: walk forward validation
st.subheader("Part 2: Walk Forward Validation")
st.write("""
Walk forward validation confirms that geopolitical signal is regime dependent. The model
performs meaningfully only during the COVID shock and the Ukraine invasion, the two
episodes where supply disruption was the dominant price driver. In demand driven or
low volatility regimes, the geopolitical channel does not produce a consistent detectable signal.
""")

p2_wf_sorted = part2_wfv.sort_values("R2", ascending=True)

fig_p2_wf = go.Figure()
fig_p2_wf.add_trace(go.Bar(
    x=p2_wf_sorted["R2"],
    y=p2_wf_sorted["Fold"],
    orientation="h",
    marker_color=[
        "#7b2d8b" if r2 > 0 else "#e8b4e8"
        for r2 in p2_wf_sorted["R2"]
    ],
    text=p2_wf_sorted["R2"].round(3),
    textposition="outside"
))
fig_p2_wf.add_vline(
    x=0,
    line_dash="dash",
    line_color="#2d004b",
    line_width=1.5
)
fig_p2_wf.update_layout(
    title="Walk Forward Validation: Test R2 by Market Regime (Part 2)",
    xaxis_title="Test R2",
    yaxis_title="",
    height=420,
    showlegend=False,
    margin=dict(l=20, r=60, t=50, b=40)
)
fig_p2_wf = style_fig(fig_p2_wf)
st.plotly_chart(fig_p2_wf, use_container_width=True)
st.caption(
    "Dark purple bars indicate positive R2. Light purple bars indicate negative R2. "
    "The zero line marks the threshold between signal and noise."
)
st.markdown("---")


# === UNIFIED MODEL COMPARISON ===

st.header("Unified Model Comparison")
st.write("""
The table below brings together results from both parts. It makes visible the
fundamental tradeoff: Part 1 models achieve high R2 because they are largely
predicting autocorrelation. Part 2 models have lower R2 but are measuring something
more meaningful: the marginal explanatory power of geopolitical variables over and
above the existing trend.
""")

unified_display = unified[~unified["Model"].isin(["Dummy", "Linear Regression"])].copy()
unified_display = unified_display.sort_values("P1 Test R2", ascending=True)

st.dataframe(
    unified_display.style.format({
        "P1 Test RMSE": "{:.2f}",
        "P1 Test R2":   "{:.3f}",
        "P2 Test RMSE": "{:.2f}",
        "P2 Test R2":   "{:.3f}"
    }).background_gradient(
        subset=["P1 Test R2", "P2 Test R2"],
        cmap="Purples"
    ),
    use_container_width=True
)

st.caption(
    "Darker purple indicates higher R2. P1 Test R2 reflects how well each model predicts the WTI price level. "
    "P2 Test R2 reflects how well it predicts deviations from trend once momentum is removed. "
    "Gradient Boosting Tuned is the best performer in both parts. "
    "Negative P2 R2 values indicate the model performs worse than simply predicting the mean deviation."
)
st.markdown("---")


# === CONCLUSION ===

st.header("Conclusion: A Proof of Concept")
st.write("""
This project set out to ask whether geopolitical signals have measurable predictive
power over short term WTI crude oil price movements. The honest answer is yes, but
not in the way a naive model would show.

The Part 1 result of R2 0.974 is seductive but it is largely a momentum model.
Strip out the rolling mean and geopolitical features account for roughly 10 percent
of daily price deviations from trend. That 10 percent is not noise. It is regime
dependent, non linear, and concentrated in precisely the periods where theory
predicts it should be strongest: acute supply side shocks like COVID and the Ukraine
invasion. In stable or demand driven markets, the signal fades.

The feature importance comparison tells the clearest story. GPR scores for Russia,
Israel and Saudi Arabia, MENA violence event counts, and lagged violence indicators
all appear as meaningful contributors once momentum is removed from the target
variable. These are not spurious correlations. They reflect the structural channel
through which geopolitical risk enters commodity pricing: supply uncertainty,
shipping route disruption, and market risk repricing.

The path to a stronger model is visible. More granular conflict data at the sub
national level, satellite imagery on pipeline and port infrastructure, options market
implied volatility as a real time risk signal, and a longer history of high frequency
event data would all sharpen the geopolitical channel. The 10 percent explained in
Part 2 is the floor of what a better specified model could achieve, not the ceiling.

This capstone is the foundation of a longer research programme at the intersection
of geopolitical risk, commodity markets, and machine learning based event forecasting.
""")
st.markdown("---")


# === FOOTER ===

st.markdown("### Project Resources")
col_link1, col_link2 = st.columns(2)
with col_link1:
    st.markdown(
        "📁 [GitHub Repository](https://github.com/milogaida/Imperial_Capstone_Project)"
    )
with col_link2:
    st.markdown(
        "📓 [Full Notebook](https://github.com/milogaida/Imperial_Capstone_Project/blob/main/notebooks/Capstone_MG.ipynb)"
    )

st.markdown("""
---
*Milo Joseph Gaida Barlafante — Imperial College Business School — MSc 2026*
*Capstone: Geopolitical Shocks and Commodity Markets*
*Supervisor: Vikesh Koul*
""")
