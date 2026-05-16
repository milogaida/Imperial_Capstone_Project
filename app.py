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
# hand back the cached result on every subsequent rerun -- so disk reads happen only
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


# KMeans on ~14k rows with StandardScaler is the most expensive computation in this app.
# It is also deterministic (random_state=42), so the result is identical every time.
# @st.cache_data hashes the input dataframe -- if df hasn't changed, Streamlit skips the
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


# === TABS ===

# st.tabs() creates the two top-level navigation tabs. Everything visible to the user
# lives inside one of these two blocks. The data loading and caching functions above stay
# outside so they run once regardless of which tab the user is viewing.
tab1, tab2, tab3 = st.tabs(["Research Report", "Investment Simulator", "Signal Analysis"])


with tab1:

    # === PAGE HEADER ===

    # Centre the logo with text-align: center on a block div. An <img> is an inline
    # element, so text-align on its parent div reliably centres it across all widths.
    st.markdown(
        '<div style="text-align: center; margin-bottom: 0.5rem;">'
        '<img src="https://nishant1529.github.io/customer_churn_dashboard/assets/imperial_logo.svg"'
        ' width="300">'
        '</div>',
        unsafe_allow_html=True,
    )

    # HTML lets us centre and style the title block precisely. unsafe_allow_html is
    # required for any inline HTML in Streamlit.
    st.markdown(
        """
        <div style="text-align: center; margin-top: 1rem;">
            <h1 style="color: #2d004b; font-weight: bold;">Geopolitical Shocks and Commodity Markets</h1>
            <p style="color: #555; font-size: 1.1rem;">Capstone Executive Summary and Report</p>
            <p style="color: #888;">by Milo Joseph Gaida Barlafante</p>
            <p style="color: #888;">May 2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


    # === EXECUTIVE SUMMARY ===

    st.header("Executive Summary")
    st.write("""
This project investigates whether geopolitical events have a measurable and predictable 
effect on WTI crude oil prices, and whether machine learning models can detect and quantify 
that effect. Oil is the world's most traded commodity and its price affects everything from 
airline tickets to heating bills, making accurate price signal detection genuinely valuable 
for investors, energy companies, and policymakers alike.

Three key findings emerge from the analysis. First, machine learning models can predict WTI 
price levels with very high accuracy, achieving an R2 of 0.974 on unseen data. However this 
performance is largely explained by price momentum rather than geopolitical insight: the model 
is essentially observing that today's oil price tends to be close to yesterday's. Second, when 
momentum is removed from the analysis by predicting daily price deviations from trend instead 
of the price level itself, geopolitical variables including conflict event counts across the 
Middle East and North Africa and country-level risk scores for Russia, Israel, and Saudi Arabia 
emerge as genuine contributors, explaining approximately 10 percent of daily price surprises. 
Third, this geopolitical signal is not constant: it appears most strongly during acute 
supply-side disruptions such as the COVID crash and the Ukraine invasion, and fades during 
demand-driven or low-volatility market periods.

The central conclusion is that geopolitical risk has a real but conditional effect on short-term 
oil prices. The 10 percent explained by Part 2 is the floor of what a better-specified model 
could achieve. With more granular conflict data, satellite monitoring of energy infrastructure, 
and options market volatility as a real-time signal, this approach could be developed into a 
practical commodity risk tool. This project is a proof of concept for that longer research 
programme.
    """)
    st.markdown("---")


    # === DATASET AND METHODOLOGY ===

    st.header("Dataset and Methodology")
    st.write("""
Oil prices are determined by a complex interaction of supply decisions, demand cycles, and 
geopolitical events. When a conflict disrupts production in a major oil-producing region, when 
sanctions close off a key supplier, or when shipping routes through the Strait of Hormuz come 
under threat, markets reprice risk within hours. Understanding and anticipating these 
geopolitical effects on commodity prices is one of the core challenges in energy economics and 
commodity investment.

This problem is a strong candidate for a machine learning approach for three reasons. First, 
the relationships involved are non-linear: a small increase in conflict intensity does not 
produce a proportional price response, and the same event can have very different effects 
depending on the broader market environment. Linear statistical models struggle to capture 
this. Second, the relevant signals are high-dimensional: no single indicator explains price 
movements, but a combination of country-level risk scores, regional conflict event counts, 
market volatility measures, and price momentum variables together contain meaningful 
information. Machine learning models are well suited to finding structure across many variables 
simultaneously. Third, there is now sufficient historical data spanning multiple geopolitical 
cycles to train and validate models robustly, something that was not possible a decade ago.

The analysis draws on 10 data sources combined into a single master dataset of 14,609 daily 
observations spanning January 1986 to December 2025. Key sources include FRED for WTI and 
Henry Hub natural gas prices, yfinance for Brent crude, gold, copper, VIX (a widely used 
measure of market anxiety, often called the fear index) and the US Dollar Index, the Caldara 
and Iacoviello Geopolitical Risk index, and ACLED conflict event data covering political 
violence by country and year. The project is structured in two parts: Part 1 predicts the WTI 
price level directly, and Part 2 isolates the geopolitical signal by predicting deviations 
from the 30-day rolling average price instead.
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
        "WTI crude oil prices have passed through eight distinct price eras since 1986, each shaped "
        "by a combination of supply decisions, demand shocks, and geopolitical events."
    )

    # fig1 uses px.line (a Plotly Express shortcut). Title removed per report style guide;
    # the st.header above is sufficient.
    fig1 = px.line(
        df,
        x="observation_date",
        y="wti_price",
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
        "Setting all four commodity prices to the same starting value of 100 in January 2000 allows "
        "a fair comparison of their growth since then. The chart shows dramatically different journeys. "
        "Gold has been the strongest performer over the period, while WTI oil and copper tend to rise "
        "and fall in cycles without a sustained upward trend."
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
    An unsupervised grouping algorithm called KMeans clustering was used to identify patterns in
    the data without being told anything about historical events. Given only price, volatility,
    and geopolitical variables, it independently identified five distinct market regimes. The
    algorithm correctly flagged the Gulf War, 9/11, the Iraq invasion, the 2008 financial crisis,
    and the COVID crash as crisis periods. Use the legend to toggle individual regimes on and off.
    """)

    # cluster_df, regime_df, cluster_names, cluster_colors were computed above by
    # run_clustering(df), which is cached and defined outside the tab block.
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
    Part 1 trains machine learning models to predict the daily WTI crude oil price. The dataset
    was split chronologically, with 80% of the data used for training (1986 to 2017) and 20%
    held back for testing (2018 to 2025). Seven models were evaluated using a time-aware
    cross-validation approach, which tests each model on successive future periods to avoid
    accidentally training on data from the future.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", "GB Tuned")
    with col2:
        st.metric("Test R2", "0.974")
    with col3:
        st.metric("Test RMSE", "$2.77")

    st.markdown("---")

    # Part 1: model comparison
    st.subheader("Model Comparison: Part 1")
    st.write(
        "Seven models were tested. Tree-based models dramatically outperform linear approaches. "
        "This confirms that the relationship between the input variables and WTI prices is non-linear, "
        "meaning it cannot be captured by a simple straight-line formula and requires more flexible "
        "methods that can detect complex patterns."
    )

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

    fig_p1_models.update_layout(showlegend=False)
    fig_p1_models = style_fig(fig_p1_models)
    fig_p1_models.update_xaxes(tickangle=30)
    st.plotly_chart(fig_p1_models, use_container_width=True)
    st.markdown("---")

    # Part 1: actual vs predicted
    st.subheader("Part 1: Actual vs Predicted WTI Price (2018 to 2025)")
    st.write("""
    The tuned Gradient Boosting model tracks the actual WTI price almost perfectly over the test
    period. However, this performance is misleading. When you examine which variables the model
    relies on most heavily, the dominant driver turns out to be the 30-day rolling average of
    the price itself. In other words, the model is saying "oil prices tend to continue in the
    same direction they have been going" rather than "this geopolitical event has shifted
    the market."
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
        xaxis_title="Date", yaxis_title="WTI Price (USD per barrel)"
    )
    fig_p1_pred = style_fig(fig_p1_pred)
    st.plotly_chart(fig_p1_pred, use_container_width=True)
    st.caption("The model tracks prices almost perfectly but is driven by the 30 day rolling average, not geopolitical features.")
    st.markdown("---")

    # Part 1: walk forward validation
    st.subheader("Part 1: Walk Forward Validation")
    st.write("""
    Walk-forward validation is a testing method that simulates real-world use: the model is
    trained on past data and then tested on the next period in sequence, repeating across five
    distinct historical windows. Performance varies considerably across these periods. The model
    struggles most during the 2008 financial crisis, which was primarily a demand shock (a
    collapse in economic activity driving down oil consumption) rather than a geopolitical supply
    disruption of the kind the model was built to detect.
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

    fig_p1_wfv.update_layout(showlegend=False)
    fig_p1_wfv = style_fig(fig_p1_wfv)
    fig_p1_wfv.update_xaxes(tickangle=25)
    st.plotly_chart(fig_p1_wfv, use_container_width=True)
    st.markdown("---")


    # === FEATURE IMPORTANCE ===

    st.header("What Is Each Model Actually Learning?")
    st.write("""
    This comparison is the core of the project. Part 1 predicts the WTI price level directly.
    The gradient boosting model achieves an R2 of 0.97, meaning it explains 97% of the variation
    in prices. But looking at which variables drove those predictions reveals an uncomfortable
    truth: the 30-day rolling average of the price itself accounts for roughly 85% of predictive
    power. The model is not reading geopolitics. It is reading momentum.

    Part 2 corrects for this by predicting the deviation from the rolling average instead. With
    momentum removed from the target, the picture changes completely. Geopolitical variables,
    including GPR scores for Russia, Israel, and Saudi Arabia, violence event counts across the
    Middle East and North Africa, and lagged conflict indicators, all appear in the top 15 most
    important features. The R2 drops to 0.097, but that 10% is genuinely geopolitical signal
    rather than an artefact of the fact that today's price tends to be close to yesterday's.
    This property, where a time series is correlated with its own recent past, is called
    autocorrelation, and it is what Part 1 is largely exploiting.
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
    Part 2 reframes what the model is asked to predict. Rather than forecasting the price level
    itself, the model predicts how far today's WTI price sits above or below its 30-day rolling
    average. This deviation measure strips out the underlying trend and momentum, leaving behind
    only the portion of price movement that might genuinely be explained by geopolitical events.

    The best model is again the tuned Gradient Boosting model, achieving a test R2 of 0.097 and
    an RMSE (average prediction error) of 3.90 dollars. That 10% of explained variance is modest,
    but it is real. The relationship between geopolitical variables and price deviations is
    non-linear, meaning it cannot be captured by simple rules, and it also varies by market
    conditions: walk-forward validation shows meaningful predictive power only during the COVID
    shock and the Ukraine invasion, the two periods defined by acute supply-side disruptions. In
    stable or demand-driven markets, geopolitical events do not move oil prices in a pattern
    consistent enough for a model to reliably detect.
    """)

    col_p2a, col_p2b, col_p2c = st.columns(3)
    with col_p2a:
        st.metric("Best Model", "GB Tuned")
    with col_p2b:
        st.metric("Test R2", "0.097")
    with col_p2c:
        st.metric("Test RMSE", "$3.90")

    st.markdown("---")

    # Part 2: model comparison
    st.subheader("Model Comparison: Part 2")
    st.write(
        "With momentum stripped out of the prediction target, all models perform far more modestly. "
        "This is the honest baseline: once a model can no longer rely on the inertia of recent "
        "prices, it must work harder to explain what actually drives day-to-day price surprises."
    )

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

    fig_p2_models.update_layout(showlegend=False)
    fig_p2_models = style_fig(fig_p2_models)
    fig_p2_models.update_xaxes(tickangle=30)
    st.plotly_chart(fig_p2_models, use_container_width=True)
    st.markdown("---")

    # Part 2: actual vs predicted deviation
    st.subheader("Part 2: Actual vs Predicted Deviation from 30 Day Rolling Average")
    st.write("""
    The chart below plots actual and predicted deviations over the test period from 2018 to 2025.
    The model captures the direction and approximate size of price spikes during major events, but
    it consistently undershoots the most extreme moves. This is expected for a tree-based model:
    it generalises from patterns it has seen in the training data, so when a crisis is larger or
    faster than anything in its history, it can only project from past experience rather than
    fully anticipating the scale of the shock.
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
    Walk-forward validation confirms that the geopolitical signal is not constant across time.
    The model performs meaningfully only during the COVID shock and the Ukraine invasion, the two
    episodes where a supply disruption was the dominant force moving prices. In periods of low
    volatility or demand-driven price changes, geopolitical events do not produce a consistent
    enough effect on prices for the model to reliably detect.
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
    The table below brings together results from both parts. It makes visible the fundamental
    trade-off between the two approaches. Part 1 models achieve high R2 scores largely because
    prices tend to stay close to recent levels, a property known as autocorrelation (the tendency
    of a series to be correlated with its own recent values). Part 2 models have lower R2 scores
    but are measuring something more meaningful: how much of the price movement that cannot be
    explained by recent trends can be attributed to geopolitical variables.
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
    This project set out to ask whether geopolitical signals have measurable predictive power over
    short-term WTI crude oil price movements. The honest answer is yes, but not in the way a first
    glance at the results would suggest.

    The Part 1 result of R2 0.974 looks impressive, but it is largely a momentum model. The model
    is not reading global events; it is observing that today's oil price tends to be close to the
    price of recent days. Strip out that rolling average and geopolitical features account for
    roughly 10% of the daily price movements that remain. That 10% is not noise. It appears most
    strongly precisely where theory predicts it should: during acute supply-side disruptions like
    the COVID crash and the Ukraine invasion. In stable or demand-driven markets, the signal fades.

    The feature importance comparison tells the clearest story. Once momentum is removed,
    Geopolitical Risk scores for Russia, Israel, and Saudi Arabia, violence event counts across
    the Middle East and North Africa, and lagged conflict indicators all emerge as meaningful
    contributors. These are not accidental correlations. They reflect the structural route through
    which geopolitical risk enters commodity pricing: supply uncertainty, shipping route disruption,
    and market risk being repriced in response to events.

    The path to a stronger model is clear. More granular conflict data at the sub-national level,
    satellite monitoring of pipeline and port infrastructure, options-market implied volatility as
    a real-time risk signal, and a longer history of high-frequency event data would all sharpen
    the geopolitical channel. The 10% explained in Part 2 is the floor of what a better-specified
    model could achieve, not the ceiling.

    This capstone is the foundation of a longer research programme at the intersection of
    geopolitical risk, commodity markets, and machine learning-based event forecasting.
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
*Milo Joseph Gaida Barlafante · Imperial College Business School · 2026*
*Capstone: Geopolitical Shocks and Commodity Markets*
""")


with tab2:

    # === INVESTMENT SIMULATOR ===

    st.header("Historical Investment Simulator")
    st.write("""
    This tool lets you explore what would have happened if you had invested a fixed amount in one
    of the four commodities studied in this research on a specific historical date and held until
    a chosen exit date. It uses the same daily price data that underpins the research report, so
    it only works within the date ranges available in the dataset. Dates that fall on weekends or
    public holidays are automatically snapped to the nearest available trading day. This simulator
    is intended as an educational demonstration of the data, not as financial advice.
    """)
    st.markdown("---")

    # Map each display label to its column name in df. This dict drives the selectbox,
    # the date range caption, and the price lookup in the calculation block below.
    COMMODITY_COLS = {
        "WTI Oil":      "wti_price",
        "Gold":         "gold_price",
        "Natural Gas":  "natgas_price",
        "Copper":       "copper_price",
    }

    # Streamlit re-runs the full script whenever any widget changes, so the date range
    # caption and the date_input bounds below will always reflect the commodity selected.
    commodity = st.selectbox("Select Commodity", list(COMMODITY_COLS.keys()))
    price_col = COMMODITY_COLS[commodity]

    # Drop rows where the price column is NaN before computing the range. Some commodities
    # start later in the dataset, so the valid window differs per commodity.
    valid_df = df.dropna(subset=[price_col])
    min_avail = valid_df["observation_date"].min().date()
    max_avail = valid_df["observation_date"].max().date()

    st.caption(
        f"Available data for {commodity}: "
        f"{min_avail.strftime('%d %b %Y')} to {max_avail.strftime('%d %b %Y')}"
    )

    investment_amount = st.number_input(
        "Investment Amount (USD)",
        min_value=1,
        value=1000,
        step=100,
    )

    # Two date pickers side by side. min_value / max_value clamp the calendar to the
    # available range for the selected commodity. The key includes the commodity name so
    # Streamlit resets the widget (and its stored value) whenever the commodity changes,
    # preventing a stale date from a different commodity's range carrying over.
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        entry_input = st.date_input(
            "Investment Date",
            value=min_avail,
            min_value=min_avail,
            max_value=max_avail,
            key=f"entry_{commodity}",
        )
    with col_d2:
        exit_input = st.date_input(
            "Exit Date",
            value=max_avail,
            min_value=min_avail,
            max_value=max_avail,
            key=f"exit_{commodity}",
        )

    # === CALCULATION ===

    if st.button("Calculate Return"):

        # --- Input validation ---
        # The widget's min_value / max_value already prevent out-of-range dates, so we
        # only need to check that the exit date is strictly after the entry date.
        if exit_input <= entry_input:
            st.error("Exit date must be after investment date.")

        else:
            # --- Snap to nearest available trading day ---
            # Convert Python date objects to Timestamps so pandas can subtract them from
            # the observation_date column (which is already pd.Timestamp). idxmin() on
            # the absolute difference returns the integer index label of the closest row.
            entry_ts = pd.Timestamp(entry_input)
            exit_ts  = pd.Timestamp(exit_input)

            entry_idx = (valid_df["observation_date"] - entry_ts).abs().idxmin()
            exit_idx  = (valid_df["observation_date"] - exit_ts).abs().idxmin()

            entry_row = valid_df.loc[entry_idx]
            exit_row  = valid_df.loc[exit_idx]

            actual_entry_date = entry_row["observation_date"].date()
            actual_exit_date  = exit_row["observation_date"].date()
            entry_price = entry_row[price_col]
            exit_price  = exit_row[price_col]

            # --- Return calculations ---
            # We notionally buy (investment / entry_price) units at entry and sell all at exit.
            end_value     = (investment_amount / entry_price) * exit_price
            dollar_return = end_value - investment_amount
            pct_return    = (dollar_return / investment_amount) * 100

            # --- Market regime at entry ---
            # cluster_df is a subset of df covering only rows where wti_price, vix_close,
            # and gpr_index are all present, so we snap to the nearest row in cluster_df
            # rather than requiring an exact match on the entry date.
            cluster_match_idx = (cluster_df["observation_date"] - entry_ts).abs().idxmin()
            cluster_id   = int(cluster_df.loc[cluster_match_idx, "cluster"])
            regime_name  = cluster_names[cluster_id]

            # --- Geopolitical context (WTI Oil only) ---
            # The GPR index measures global geopolitical risk and is the primary geopolitical
            # signal studied in the research. It is most directly relevant to oil, so this
            # contextual sentence is only shown when WTI Oil is the selected commodity.
            gpr_sentence = ""
            if commodity == "WTI Oil":
                gpr_window = df[
                    (df["observation_date"] >= entry_ts) &
                    (df["observation_date"] <= exit_ts)
                ]["gpr_index"].dropna()

                if len(gpr_window) > 0:
                    avg_gpr = gpr_window.mean()
                    if avg_gpr > 150:
                        gpr_sentence = (
                            f"Geopolitical risk was elevated over this period "
                            f"(average GPR index: {avg_gpr:.1f})."
                        )
                    elif avg_gpr < 75:
                        gpr_sentence = (
                            f"Geopolitical conditions were relatively calm over this period "
                            f"(average GPR index: {avg_gpr:.1f})."
                        )
                    else:
                        gpr_sentence = (
                            f"Geopolitical risk was moderate over this period "
                            f"(average GPR index: {avg_gpr:.1f})."
                        )
                else:
                    gpr_sentence = "GPR data is not available for this period."

            # --- Render results box ---
            # Green for a gain, red for a loss. The sign prefix is applied to both the
            # dollar and percentage return so positive numbers show a leading "+".
            return_color = "#27ae60" if dollar_return >= 0 else "#e74c3c"
            sign = "+" if dollar_return >= 0 else ""

            # Build the optional GPR row as a string so we can insert or omit it cleanly.
            gpr_row_html = (
                f'<tr style="border-bottom:1px solid #ddd8e8;">'
                f'<td style="padding:5px 8px;"><strong>Geopolitical context</strong></td>'
                f'<td style="padding:5px 8px;">{gpr_sentence}</td></tr>'
                if gpr_sentence else ""
            )

            results_html = f"""
            <div style="background-color:#f0ebf8; border-left:4px solid #7b2d8b;
                        padding:1.5rem; border-radius:6px; margin-top:1rem;">
                <h3 style="color:#2d004b; margin-top:0; margin-bottom:1rem;">Simulation Results</h3>
                <table style="width:100%; border-collapse:collapse; color:#2d004b; font-size:1rem;">
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Entry date used</strong></td>
                        <td style="padding:5px 8px;">{actual_entry_date.strftime('%d %b %Y')}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Exit date used</strong></td>
                        <td style="padding:5px 8px;">{actual_exit_date.strftime('%d %b %Y')}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Entry price</strong></td>
                        <td style="padding:5px 8px;">${entry_price:.2f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Exit price</strong></td>
                        <td style="padding:5px 8px;">${exit_price:.2f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Initial investment</strong></td>
                        <td style="padding:5px 8px;">${investment_amount:,.2f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Final value</strong></td>
                        <td style="padding:5px 8px;">${end_value:,.2f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Dollar return</strong></td>
                        <td style="padding:5px 8px; color:{return_color}; font-weight:bold;">
                            {sign}${abs(dollar_return):,.2f}
                        </td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Percentage return</strong></td>
                        <td style="padding:5px 8px; color:{return_color}; font-weight:bold;">
                            {sign}{pct_return:.2f}%
                        </td>
                    </tr>
                    <tr style="border-bottom:1px solid #ddd8e8;">
                        <td style="padding:5px 8px;"><strong>Market regime at entry</strong></td>
                        <td style="padding:5px 8px;">{regime_name}</td>
                    </tr>{gpr_row_html}
                </table>
                <p style="color:#888; font-size:0.85rem; margin-top:1rem; margin-bottom:0;">
                    For educational purposes only. Not financial advice.
                </p>
            </div>
            """
            st.markdown(results_html, unsafe_allow_html=True)


with tab3:

    # === SIGNAL ANALYSIS ===

    st.header("Signal Analysis")
    st.write("""
    This tool reads the geopolitical and market conditions recorded in the dataset on a
    chosen date and interprets what those conditions historically signal for commodity prices,
    based on the findings from the Part 2 analysis. It works within the dataset date range
    only. It is a demonstration of the geopolitical signal the research identified, not a
    live forecast.
    """)
    st.markdown("---")

    # Restrict the selectable range to rows where all five signal variables are present.
    # wti_rolling_mean_30 requires 30 days of history, and mena_violence_events only begins
    # once the ACLED dataset starts, so the effective date range is smaller than the full df.
    sig_valid_df = df.dropna(
        subset=["wti_price", "gpr_index", "vix_close", "wti_rolling_mean_30", "mena_violence_events"]
    )
    sig_min_date = sig_valid_df["observation_date"].min().date()
    sig_max_date = sig_valid_df["observation_date"].max().date()

    # Default to the Russia-Ukraine invasion date: a high-signal event where GPR, MENA conflict,
    # and price deviation all moved simultaneously, making it the clearest demonstration of the tool.
    ukraine_date = pd.Timestamp("2022-02-24").date()
    default_date = ukraine_date if sig_min_date <= ukraine_date <= sig_max_date else sig_max_date

    analysis_date = st.date_input(
        "Analysis Date",
        value=default_date,
        min_value=sig_min_date,
        max_value=sig_max_date,
        key="signal_analysis_date",
    )

    if st.button("Analyse Conditions"):

        # --- Snap to nearest available trading day ---
        # Same method as the Investment Simulator: find the row with the smallest absolute
        # time delta from the chosen date and treat it as the matched observation.
        analysis_ts = pd.Timestamp(analysis_date)
        match_idx   = (sig_valid_df["observation_date"] - analysis_ts).abs().idxmin()
        row         = sig_valid_df.loc[match_idx]

        # --- Extract values from the matched row ---
        actual_date           = row["observation_date"].date()
        wti_price             = row["wti_price"]
        gpr_index             = row["gpr_index"]
        gpr_russia            = row["gpr_russia"]
        gpr_israel            = row["gpr_israel"]
        gpr_saudi             = row["gpr_saudi"]
        mena_violence_events  = row["mena_violence_events"]
        total_violence_events = row["total_violence_events"]
        vix_close             = row["vix_close"]
        wti_rolling_mean_30   = row["wti_rolling_mean_30"]
        wti_deviation         = wti_price - wti_rolling_mean_30

        # cluster_df only covers rows where wti_price, vix_close, and gpr_index are all
        # present, so we snap to its nearest row separately to get the market regime label.
        cluster_match_idx = (cluster_df["observation_date"] - analysis_ts).abs().idxmin()
        cluster_id        = int(cluster_df.loc[cluster_match_idx, "cluster"])
        cluster_name      = cluster_names[cluster_id]

        # --- Historical context values computed from the full dataset ---
        gpr_historical_mean  = df["gpr_index"].mean()
        mena_historical_mean = df["mena_violence_events"].mean()

        # rank(pct=True) assigns each observation a fractional rank in [0, 1]; multiplying
        # by 100 gives the percentage of all historical values that fall below this one.
        # match_idx is a valid label in df because sig_valid_df shares its pandas index.
        gpr_percentile = df["gpr_index"].rank(pct=True).loc[match_idx] * 100

        vix_elevated = bool(vix_close > 20)

        # --- Overall geopolitical environment ---
        if gpr_index > gpr_historical_mean * 1.5:
            overall = "significantly elevated geopolitical risk"
        elif gpr_index > gpr_historical_mean:
            overall = "above average geopolitical risk"
        elif gpr_index < gpr_historical_mean * 0.75:
            overall = "a relatively calm geopolitical environment"
        else:
            overall = "moderate geopolitical conditions"

        # --- Dominant signal source country ---
        # Replace NaN country-level GPR values with 0.0 so the max() comparison always
        # works cleanly even if a country index is missing for early dates in the dataset.
        country_gpr = {
            "Russia":       gpr_russia if pd.notna(gpr_russia) else 0.0,
            "Israel":       gpr_israel if pd.notna(gpr_israel) else 0.0,
            "Saudi Arabia": gpr_saudi  if pd.notna(gpr_saudi)  else 0.0,
        }
        dominant_country = max(country_gpr, key=country_gpr.get)
        dominant_value   = country_gpr[dominant_country]

        # --- MENA conflict signal ---
        if mena_violence_events > mena_historical_mean * 1.5:
            mena_signal = "significantly above average conflict activity in the Middle East and North Africa"
        elif mena_violence_events > mena_historical_mean:
            mena_signal = "above average conflict activity in the Middle East and North Africa"
        else:
            mena_signal = "relatively low conflict activity in the Middle East and North Africa"

        # --- Price deviation signal ---
        if wti_deviation > 3:
            price_signal = "WTI oil was trading well above its recent trend, suggesting the market had already priced in some risk premium"
        elif wti_deviation < -3:
            price_signal = "WTI oil was trading below its recent trend, suggesting the market had not yet priced in elevated risk"
        else:
            price_signal = "WTI oil was trading close to its recent trend"

        # --- Signal confidence ---
        # Confidence is highest when GPR and MENA conflict are simultaneously elevated because
        # Part 2 found the geopolitical channel most reliably active during combined supply-side
        # stress, not when geopolitical risk rises without a corresponding conflict reading.
        if gpr_index > gpr_historical_mean * 1.5 and mena_violence_events > mena_historical_mean * 1.5:
            confidence = "moderately high"
        elif gpr_index > gpr_historical_mean * 1.5 and mena_violence_events > mena_historical_mean:
            confidence = "moderate"
        elif gpr_index < gpr_historical_mean and mena_violence_events < mena_historical_mean:
            confidence = "low"
        else:
            confidence = "low to moderate"

        # --- Commodity recommendation and price direction ---
        if confidence in ["moderate", "moderately high"] and wti_deviation < 3:
            rec_commodity = "WTI Oil"
            rec_direction = "upward price pressure"
        elif confidence in ["moderate", "moderately high"] and wti_deviation > 3:
            rec_commodity = "WTI Oil"
            rec_direction = "sustained elevated prices but with mean reversion risk"
        else:
            rec_commodity = "Gold"
            rec_direction = "safe haven demand"

        # --- Investment horizon ---
        if vix_elevated and gpr_index > gpr_historical_mean * 1.5:
            horizon = "short to medium term (2 to 8 weeks), as high volatility makes longer holds unpredictable"
        else:
            horizon = "medium term (1 to 3 months)"

        # --- Prose interpretation ---
        # Five paragraphs: overall environment, MENA context, price signal, recommendation,
        # horizon and disclaimer. No bullet points, no em dashes. Written for an informed
        # financial reader who understands markets but is not a data scientist.
        analysis_text = f"""
<p>On {actual_date.strftime('%d %b %Y')}, global conditions were characterised by {overall}.
The Geopolitical Risk index stood at {gpr_index:.1f}, placing it at the {gpr_percentile:.0f}th
percentile of its historical distribution from 1986 to 2025. Among the three major oil-producing
regions tracked in this dataset, {dominant_country} recorded the highest country-level GPR reading
at {dominant_value:.1f}, making it the dominant signal source on this date. The market was
classified as being in the {cluster_name} regime, the cluster the unsupervised algorithm identified
as characteristic of conditions combining that level of price, volatility, and geopolitical
stress.</p>

<p>The Middle East and North Africa region is the most consequential geography for global oil
supply. It accounts for a significant share of world production and controls the shipping lanes
through which a large fraction of global oil trade moves, including the Strait of Hormuz and
the Suez Canal. Conflict in this region raises the risk of supply disruption through damage to
production infrastructure, closure of transit routes, and the general repricing of supply risk
by market participants. On {actual_date.strftime('%d %b %Y')}, the dataset recorded
{mena_violence_events:.0f} MENA conflict events, representing {mena_signal}. The historical
daily average across the full dataset is {mena_historical_mean:.1f} events.</p>

<p>The deviation of WTI crude oil from its 30-day rolling average is the central metric that
Part 2 of the research was built to explain. A positive deviation means oil is trading above
its recent trend, typically indicating that some risk has already been absorbed into the price.
A negative deviation means the market may not yet have reacted to the underlying conditions. On
this date WTI was priced at ${wti_price:.2f} per barrel against a 30-day rolling average of
${wti_rolling_mean_30:.2f}, producing a deviation of {wti_deviation:+.2f} dollars.
{price_signal}.</p>

<p>Taking these readings together, the conditions on this date most directly affect {rec_commodity},
where the geopolitical signal points toward {rec_direction}. The overall signal confidence is
{confidence}. Confidence reaches moderate or higher when the GPR index exceeds one and a half
times its historical average and MENA conflict activity is simultaneously elevated, because the
Part 2 findings showed the geopolitical channel to be most reliably active during periods that
combined both signals. A lower MENA reading or a GPR index below its historical average each
reduce confidence, since the research found little consistent geopolitical effect on prices
outside genuine supply-side stress events.</p>

<p>Given these conditions, the investment horizon suggested by this signal is {horizon}. This is
a historical demonstration of the geopolitical signal identified in this research project, applied
retrospectively to conditions recorded in the dataset on the chosen date. It is not a live
forecast, does not account for events that occurred after that date, and should not be treated
as financial advice.</p>
"""

        # --- Summary table ---
        # Six rows, two columns. use_container_width=False keeps the table narrow so it reads
        # as a compact snapshot rather than stretching across the full width of the page.
        deviation_label = f"${wti_deviation:+.2f} vs 30-day avg"
        summary_df = pd.DataFrame({
            "Metric": ["Date", "WTI Price", "GPR Index", "Market Regime", "VIX", "WTI vs Trend"],
            "Value": [
                actual_date.strftime("%d %b %Y"),
                f"${wti_price:.2f}",
                f"{gpr_index:.1f} ({gpr_percentile:.0f}th percentile)",
                cluster_name,
                f"{vix_close:.1f}{'  (elevated)' if vix_elevated else ''}",
                deviation_label,
            ],
        })
        st.dataframe(summary_df, use_container_width=False, hide_index=True)

        # --- Styled interpretation box ---
        # analysis_text contains HTML paragraph tags, so unsafe_allow_html=True is required
        # for the prose to render as styled text rather than raw markup.
        # Build the styled box as a single concatenated string so there are no blank lines
        # between the opening wrapper tags, the paragraph content, and the closing tags.
        # An f-string with {analysis_text} on its own line creates a blank line (when
        # analysis_text has a leading newline) that terminates the HTML block early in
        # Streamlit's markdown renderer, leaving the closing </div> tags as literal text.
        st.markdown(
            '<div style="background-color:#f0ebf8; border-left:4px solid #7b2d8b;'
            ' padding:1.5rem; border-radius:6px; margin-top:1rem;">'
            '<h3 style="color:#2d004b; margin-top:0; margin-bottom:1rem;">Signal Interpretation</h3>'
            '<div style="color:#2d004b; font-size:1rem; line-height:1.8;">'
            + analysis_text.strip()
            + '</div></div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Signal analysis based on research findings from this project. "
            "For educational purposes only. Not financial advice."
        )
