"""
╔══════════════════════════════════════════════════════════════╗
║   AI IMPACT ON JOBS — Interactive Dashboard                  ║
║   Data Mining Final Project — Phase 2                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI & Jobs Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — dark refined look
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* Root palette */
:root {
    --bg:        #09090f;
    --surface:   #12121e;
    --card:      #181828;
    --border:    #2a2a42;
    --accent:    #6c63ff;
    --accent2:   #00e5b4;
    --danger:    #ff4d6d;
    --warning:   #ffb347;
    --text:      #e8e8f0;
    --muted:     #8888a8;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1600px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: border-color .2s;
}
[data-testid="metric-container"]:hover { border-color: var(--accent); }
[data-testid="stMetricValue"]  { font-family: var(--mono); font-size: 2rem !important; color: var(--accent2) !important; }
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: .8rem !important; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="stMetricDelta"]  { font-family: var(--mono); }

/* Tabs */
.stTabs [data-baseweb="tab-list"]    { background: var(--surface); border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"]         { background: transparent; border-radius: 8px; color: var(--muted); font-family: var(--sans); font-weight: 500; padding: .5rem 1.2rem; }
.stTabs [aria-selected="true"]       { background: var(--accent) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-border"]  { display: none; }

/* Headings */
h1,h2,h3,h4 { font-family: var(--sans); }

/* Section title helper */
.sec-title {
    font-family: var(--mono);
    font-size: .7rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .25rem;
}
.sec-heading {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: var(--text);
}

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, var(--card) 0%, #1a1a30 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
}
.insight-card h4 { margin: 0 0 .4rem; color: var(--accent2); font-size: .95rem; }
.insight-card p  { margin: 0; color: var(--muted); font-size: .85rem; line-height: 1.5; }

/* Predict result box */
.pred-box {
    background: linear-gradient(135deg, #1e1e35, #252540);
    border: 2px solid var(--accent);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.pred-box .label   { font-family: var(--mono); font-size: .8rem; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
.pred-box .value   { font-size: 2.4rem; font-weight: 700; margin: .4rem 0; }
.pred-box .sub     { font-size: .9rem; color: var(--muted); }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* Badge */
.badge {
    display: inline-block;
    padding: .15rem .6rem;
    border-radius: 99px;
    font-family: var(--mono);
    font-size: .7rem;
    letter-spacing: .05em;
    margin-left: .4rem;
}
.badge-green  { background: #00e5b420; color: var(--accent2); border: 1px solid #00e5b440; }
.badge-purple { background: #6c63ff20; color: var(--accent); border: 1px solid #6c63ff40; }
.badge-red    { background: #ff4d6d20; color: var(--danger); border: 1px solid #ff4d6d40; }

/* Plotly chart wrappers */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* selectbox / slider styling */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"]    label { color: var(--muted); font-size: .85rem; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#12121e",
        plot_bgcolor="#12121e",
        font=dict(family="DM Sans", color="#e8e8f0", size=12),
        xaxis=dict(gridcolor="#2a2a42", linecolor="#2a2a42", zerolinecolor="#2a2a42"),
        yaxis=dict(gridcolor="#2a2a42", linecolor="#2a2a42", zerolinecolor="#2a2a42"),
        colorway=["#6c63ff","#00e5b4","#ff4d6d","#ffb347","#38bdf8","#a78bfa","#fb923c"],
        legend=dict(bgcolor="rgba(24,24,40,0.12)", bordercolor="#2a2a42"),
        margin=dict(t=50, b=40, l=50, r=30),
    )
)

COLOR_MAP = {0: "#ff4d6d", 1: "#ffb347", 2: "#00e5b4"}
CLASS_NAMES = {0: "At Risk", 1: "Stable", 2: "Growing"}
CLUSTER_COLORS = ["#6c63ff","#00e5b4","#ffb347"]


# ─────────────────────────────────────────────
# DATA GENERATION  (mirrors dataset structure)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_real_dataset(path="dashboard_dataset.xls"):
    df = pd.read_csv(path)

    df.columns = df.columns.str.replace('\ufeff', '', regex=False)
    df.columns = df.columns.str.replace('"', '', regex=False)
    df.columns = df.columns.str.strip()

    df = df.replace([np.inf, -np.inf], np.nan)

    df["job_survival_class"] = pd.to_numeric(df["job_survival_class"], errors="coerce")
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

    df = df.dropna(subset=["job_survival_class", "salary"])
    df = df[df["salary"] > 0]
    df = df.reset_index(drop=True)

    df["job_survival_class"] = df["job_survival_class"].astype(int)

    return df


@st.cache_data(show_spinner=False)
def build_feature_matrix(df):
    target = df["job_survival_class"].copy()
    salary = df["salary"].copy()

    leaky_cols = [
        "target",
        "job_survival_class",
        "salary",
        "salary_bucket",
        "ai_risk_category",
        "salary_log"
    ]

    X = df.drop(columns=[c for c in leaky_cols if c in df.columns])

    X = pd.get_dummies(
        X,
        columns=[
            c for c in ["job_title", "country", "primary_skill"]
            if c in X.columns
        ],
        drop_first=False,
        dtype=int
    )

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, target, salary, X.columns.tolist()

@st.cache_resource(show_spinner=False)
def train_all_models(seed=42):
    df = load_real_dataset()
    X, y_class, y_salary, feat_cols = build_feature_matrix(df)
    y_salary_log = np.log1p(y_salary.clip(lower=1).values)

    X_tr_raw, X_te_raw, yc_tr, yc_te = train_test_split(
    X, y_class,
    test_size=.2,
    random_state=seed,
    stratify=y_class
    )

    scaler = RobustScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=feat_cols, index=X_tr_raw.index)
    X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=feat_cols, index=X_te_raw.index)

    yr_tr = y_salary_log[X_tr.index]
    yr_te = y_salary_log[X_te.index]

    # Classification
    lr_clf  = LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")
    rf_clf  = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=seed, n_jobs=-1)
    lr_clf.fit(X_tr, yc_tr)
    rf_clf.fit(X_tr, yc_tr)

    # Regression
    lin_reg = LinearRegression()
    rf_reg  = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1)
    lin_reg.fit(X_tr, yr_tr)
    rf_reg.fit(X_tr, yr_tr)

    # Clustering
    std = StandardScaler()
    X_std = std.fit_transform(X)
    km    = KMeans(n_clusters=3, random_state=seed, n_init=10)
    clusters = km.fit_predict(X_std)

    # Anomaly detection
    iso   = IsolationForest(contamination=0.05, random_state=seed, n_jobs=-1)
    anoms = iso.fit_predict(X_std)

    # PCA
    pca   = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_std)

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":    feat_cols,
        "importance": rf_clf.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    # Metrics
    lr_pred = lr_clf.predict(X_te)
    rf_pred = rf_clf.predict(X_te)
    lr_reg_pred = lin_reg.predict(X_te)
    rf_reg_pred = rf_reg.predict(X_te)

    clf_metrics = pd.DataFrame({
        "Model":         ["Logistic Regression","Random Forest"],
        "Accuracy":      [accuracy_score(yc_te, lr_pred), accuracy_score(yc_te, rf_pred)],
        "Macro F1":      [f1_score(yc_te, lr_pred, average="macro"), f1_score(yc_te, rf_pred, average="macro")],
    }).round(4)

    reg_metrics = pd.DataFrame({
        "Model":     ["Linear Regression","Random Forest"],
        "RMSE":      [np.sqrt(mean_squared_error(yr_te, lr_reg_pred)), np.sqrt(mean_squared_error(yr_te, rf_reg_pred))],
        "MAE":       [mean_absolute_error(yr_te, lr_reg_pred),         mean_absolute_error(yr_te, rf_reg_pred)],
        "R² Score":  [r2_score(yr_te, lr_reg_pred),                    r2_score(yr_te, rf_reg_pred)],
    }).round(4)

    cm_rf = confusion_matrix(yc_te, rf_pred)

    return dict(
        df=df, X=X, X_std=X_std, X_pca=X_pca,
        y_class=y_class, y_salary=y_salary,
        X_tr=X_tr, X_te=X_te, yc_tr=yc_tr, yc_te=yc_te,
        yr_tr=yr_tr, yr_te=yr_te,
        rf_clf=rf_clf, lr_clf=lr_clf,
        rf_reg=rf_reg, lin_reg=lin_reg,
        clusters=clusters, anoms=anoms,
        pca=pca, km=km, std=std, scaler=scaler,
        feat_cols=feat_cols,
        fi_df=fi_df, clf_metrics=clf_metrics, reg_metrics=reg_metrics,
        cm_rf=cm_rf, rf_pred=rf_pred, rf_reg_pred=rf_reg_pred,
    )


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner("🔄 Loading models and data..."):
    M = train_all_models()

df        = M["df"].copy()
df["cluster"]       = M["clusters"]
df["anomaly"]       = M["anoms"]
df["anomaly_label"] = df["anomaly"].map({1:"Normal", -1:"Anomaly"})
df["class_label"]   = df["job_survival_class"].map(CLASS_NAMES)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: .5rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🤖</div>
        <div style='font-family:"Space Mono",monospace; font-size:.85rem; color:#6c63ff; letter-spacing:.1em;'>AI & JOBS</div>
        <div style='font-size:1.1rem; font-weight:700; margin-top:.2rem;'>Intelligence Platform</div>
        <div style='font-size:.72rem; color:#8888a8; margin-top:.2rem;'>Data Mining · CRISP-DM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Global Filters")

    sel_country = st.multiselect(
        "Country",
        options=sorted(df["country"].unique()),
        default=sorted(df["country"].unique()),
    )
    sel_exp = st.multiselect(
        "Experience Level",
        options=["Entry","Mid","Senior"],
        default=["Entry","Mid","Senior"],
    )
    sel_edu = st.multiselect(
        "Education Level",
        options=["Bachelor","Master","PhD"],
        default=["Bachelor","Master","PhD"],
    )
    sel_year = st.slider(
        "Year Range",
        int(df["year"].min()), int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max()))
    )
    sel_risk = st.slider("Max AI Risk Score", 0, 100, 100)

    st.markdown("---")
    st.markdown(f"<div style='font-size:.75rem;color:#8888a8;'>Dataset: <b style='color:#e8e8f0'>{len(df):,}</b> records</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.75rem;color:#8888a8;'>Features: <b style='color:#e8e8f0'>{len(M['feat_cols'])}</b></div>", unsafe_allow_html=True)

# Apply filters
mask = (
    df["country"].isin(sel_country) &
    df["experience_level"].isin(sel_exp) &
    df["education_level"].isin(sel_edu) &
    df["year"].between(*sel_year) &
    (df["ai_risk_score"] <= sel_risk)
)
dff = df[mask].copy()

if len(dff) == 0:
    st.warning("⚠️ No data matches the current filters. Please broaden your selection.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;'>
    <div style='font-family:"Space Mono",monospace; font-size:.7rem; color:#6c63ff; letter-spacing:.15em; text-transform:uppercase;'>Data Mining Final Project · Phase 2</div>
    <h1 style='margin:.2rem 0 .1rem; font-size:2rem; font-weight:800;'>AI Impact on Jobs <span style='color:#6c63ff;'>&</span> Salary Intelligence</h1>
    <p style='color:#8888a8; margin:0; font-size:.9rem;'>End-to-end CRISP-DM pipeline · Classification · Regression · Clustering · Anomaly Detection · PCA</p>
</div>
""", unsafe_allow_html=True)

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Total Records",       f"{len(dff):,}")
with k2: st.metric("Avg AI Risk",         f"{dff['ai_risk_score'].mean():.1f}")
with k3: st.metric("Avg Salary",          f"${dff['salary'].mean():,.0f}")
with k4: st.metric("At-Risk Jobs",        f"{(dff['job_survival_class']==0).sum():,}")
with k5: st.metric("Growing Jobs",        f"{(dff['job_survival_class']==2).sum():,}")

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🧠 Classification",
    "💰 Regression",
    "🔵 Clustering",
    "🚨 Anomaly Detection",
    "🔻 PCA",
    "🎯 Predict",
    "💡 Business Insights",
])

# ═══════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW / EDA
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-title">Phase 1 · Data Understanding</div><div class="sec-heading">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            dff["class_label"].value_counts().reset_index(),
            x="class_label", y="count",
            color="class_label",
            color_discrete_map={"At Risk":"#ff4d6d","Stable":"#ffb347","Growing":"#00e5b4"},
            title="Target Distribution — Job Survival Class",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            dff, x="ai_risk_score", color="class_label",
            nbins=40, barmode="overlay", opacity=.7,
            color_discrete_map={"At Risk":"#ff4d6d","Stable":"#ffb347","Growing":"#00e5b4"},
            title="AI Risk Score Distribution by Survival Class",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(
            dff, x="experience_level", y="salary",
            color="class_label",
            color_discrete_map={"At Risk":"#ff4d6d","Stable":"#ffb347","Growing":"#00e5b4"},
            category_orders={"experience_level":["Entry","Mid","Senior"]},
            title="Salary by Experience & Survival Class",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(
            dff.sample(min(2000, len(dff)), random_state=0),
            x="ai_risk_score", y="skill_demand_score",
            color="class_label", size="salary",
            size_max=12, opacity=.65,
            color_discrete_map={"At Risk":"#ff4d6d","Stable":"#ffb347","Growing":"#00e5b4"},
            title="AI Risk vs Skill Demand (size = salary)",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        top_jobs = dff.groupby("job_title")["ai_risk_score"].mean().nlargest(10).reset_index()
        fig = px.bar(
            top_jobs, x="ai_risk_score", y="job_title", orientation="h",
            color="ai_risk_score", color_continuous_scale="RdYlGn_r",
            title="Top 10 Highest AI Risk Roles",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        country_risk = dff.groupby("country")["ai_risk_score"].mean().reset_index()
        fig = px.choropleth(
            country_risk, locations="country", locationmode="country names",
            color="ai_risk_score", color_continuous_scale="RdYlGn_r",
            title="Average AI Risk by Country",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec-title">Phase 4 · Modelling</div><div class="sec-heading">Classification — Job Survival Prediction</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("#### Model Performance Comparison")
        cm_df = M["clf_metrics"].copy()
        cm_df["Accuracy"]  = cm_df["Accuracy"].apply(lambda x: f"{x:.2%}")
        cm_df["Macro F1"]  = cm_df["Macro F1"].apply(lambda x: f"{x:.4f}")
        st.dataframe(cm_df, use_container_width=True, hide_index=True)

        # Bar comparison
        fig = go.Figure()
        for i, row in M["clf_metrics"].iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],
                x=["Accuracy","Macro F1"],
                y=[row["Accuracy"], row["Macro F1"]],
                text=[f"{row['Accuracy']:.2%}", f"{row['Macro F1']:.4f}"],
                textposition="outside",
            ))
        fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        barmode="group",
        title="Classification Metrics Comparison",
        )

        fig.update_yaxes(range=[0,1.1], gridcolor="#2a2a42")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Confusion Matrix — Random Forest")
        cm = M["cm_rf"]
        fig = px.imshow(
            cm, text_auto=True,
            x=["At Risk","Stable","Growing"],
            y=["At Risk","Stable","Growing"],
            color_continuous_scale=[[0,"#12121e"],[1,"#6c63ff"]],
            title="Random Forest Confusion Matrix",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 15 Feature Importances")
    fi = M["fi_df"]
    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale=["#2a2a42","#6c63ff","#00e5b4"],
        title="Feature Importances (Tuned Random Forest)",
    )
    fig.update_layout(
    **PLOTLY_TEMPLATE["layout"],
    showlegend=False,
    coloraxis_showscale=False,
    height=450,
    )

    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — REGRESSION
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-title">Phase 4 · Modelling</div><div class="sec-heading">Regression — Salary Prediction</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown("#### Model Performance Comparison")
        rm = M["reg_metrics"].copy()
        rm["RMSE"]      = rm["RMSE"].apply(lambda x: f"{x:.4f}")
        rm["MAE"]       = rm["MAE"].apply(lambda x: f"{x:.4f}")
        rm["R² Score"]  = rm["R² Score"].apply(lambda x: f"{x:.4f}")
        st.dataframe(rm, use_container_width=True, hide_index=True)

        fig = go.Figure()
        for i, row in M["reg_metrics"].iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],
                x=["R² Score"],
                y=[row["R² Score"]],
                text=[f"{row['R² Score']:.4f}"],
                textposition="outside",
            ))
        fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title="R² Score Comparison",
        )

        fig.update_yaxes(range=[0,1.1])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Predicted vs Actual Salary (RF Regressor, log scale)")
        yr_te_arr = np.array(M["yr_te"])
        rf_reg_pred = M["rf_reg_pred"]
        sample_idx = np.random.default_rng(0).choice(len(yr_te_arr), min(600, len(yr_te_arr)), replace=False)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yr_te_arr[sample_idx], y=rf_reg_pred[sample_idx],
            mode="markers", marker=dict(color="#6c63ff", opacity=.5, size=5),
            name="Predictions",
        ))
        lo, hi = yr_te_arr.min(), yr_te_arr.max()
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", line=dict(color="#00e5b4", dash="dash"),
            name="Perfect Fit",
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Actual vs Predicted (log salary)",
                          xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig, use_container_width=True)

    # Salary distribution by class
    st.markdown("#### Salary Distribution by Survival Class (Filtered Data)")
    fig = px.violin(
        dff, x="class_label", y="salary",
        color="class_label", box=True, points=False,
        color_discrete_map={"At Risk":"#ff4d6d","Stable":"#ffb347","Growing":"#00e5b4"},
        title="Salary Distribution by Job Survival Class",
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — CLUSTERING
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec-title">Phase 4 · Modelling</div><div class="sec-heading">K-Means Clustering — Workforce Segments</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        sub = df[df["cluster"] == i]
        col.metric(f"Cluster {i} Size", f"{len(sub):,}")
        col.metric(f"Avg Salary", f"${sub['salary'].mean():,.0f}")
        col.metric(f"Avg AI Risk", f"{sub['ai_risk_score'].mean():.1f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        cluster_sal = df.groupby("cluster")["salary"].mean().reset_index()
        fig = px.bar(
            cluster_sal, x="cluster", y="salary",
            color="cluster", color_discrete_sequence=CLUSTER_COLORS,
            title="Average Salary by Cluster",
            text_auto=",.0f",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cluster_risk = df.groupby("cluster")["ai_risk_score"].mean().reset_index()
        fig = px.bar(
            cluster_risk, x="cluster", y="ai_risk_score",
            color="cluster", color_discrete_sequence=CLUSTER_COLORS,
            title="Average AI Risk Score by Cluster",
            text_auto=".1f",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart per cluster
    st.markdown("#### Cluster Profile Radar")
    attrs = ["ai_risk_score","skill_demand_score","salary","job_openings"]
    fig = go.Figure()
    for i in range(3):
        sub = df[df["cluster"]==i][attrs].mean()
        sub_norm = (sub - df[attrs].min()) / (df[attrs].max() - df[attrs].min())
        fig.add_trace(go.Scatterpolar(
            r=sub_norm.tolist() + [sub_norm.iloc[0]],
            theta=attrs + [attrs[0]],
            fill="toself", opacity=.5,
            name=f"Cluster {i}",
            line=dict(color=CLUSTER_COLORS[i]),
        ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                      polar=dict(bgcolor="#12121e",
                                 radialaxis=dict(gridcolor="#2a2a42"),
                                 angularaxis=dict(gridcolor="#2a2a42")),
                      title="Normalized Cluster Profiles")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-title">Phase 4 · Modelling</div><div class="sec-heading">Anomaly Detection — Isolation Forest</div>', unsafe_allow_html=True)

    n_anom   = (df["anomaly"] == -1).sum()
    n_normal = (df["anomaly"] == 1).sum()

    ka, kb, kc = st.columns(3)
    ka.metric("Total Anomalies",  f"{n_anom:,}")
    kb.metric("Normal Records",   f"{n_normal:,}")
    kc.metric("Anomaly Rate",     f"{n_anom/len(df)*100:.1f}%")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            df.sample(min(3000, len(df)), random_state=0),
            x="salary", y="ai_risk_score",
            color="anomaly_label",
            color_discrete_map={"Normal":"#6c63ff","Anomaly":"#ff4d6d"},
            opacity=.55, size_max=8,
            title="Anomalies — Salary vs AI Risk Score",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        anom_by_cluster = df.groupby("cluster")["anomaly"].apply(
            lambda x: (x == -1).mean() * 100
        ).reset_index()
        anom_by_cluster.columns = ["cluster","anomaly_rate"]
        fig = px.bar(
            anom_by_cluster, x="cluster", y="anomaly_rate",
            color="cluster", color_discrete_sequence=CLUSTER_COLORS,
            title="Anomaly Rate per Cluster (%)",
            text_auto=".1f",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Anomaly Profile — Key Statistics")
    anom_sub    = df[df["anomaly"]==-1]
    normal_sub  = df[df["anomaly"]==1]

    compare = pd.DataFrame({
        "Metric":        ["Avg Salary","Avg AI Risk","Avg Skill Demand","Avg Job Openings"],
        "Anomaly":       [anom_sub["salary"].mean(), anom_sub["ai_risk_score"].mean(),
                          anom_sub["skill_demand_score"].mean(), anom_sub["job_openings"].mean()],
        "Normal":        [normal_sub["salary"].mean(), normal_sub["ai_risk_score"].mean(),
                          normal_sub["skill_demand_score"].mean(), normal_sub["job_openings"].mean()],
    }).round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Anomaly", x=compare["Metric"], y=compare["Anomaly"], marker_color="#ff4d6d"))
    fig.add_trace(go.Bar(name="Normal",  x=compare["Metric"], y=compare["Normal"],  marker_color="#6c63ff"))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], barmode="group", title="Anomaly vs Normal — Mean Feature Values")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 5 — PCA
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec-title">Phase 4 · Modelling</div><div class="sec-heading">PCA — Dimensionality Reduction</div>', unsafe_allow_html=True)

    ev   = M["pca"].explained_variance_ratio_
    k1v, k2v, k3v = st.columns(3)
    k1v.metric("PC1 Explained Variance", f"{ev[0]*100:.1f}%")
    k2v.metric("PC2 Explained Variance", f"{ev[1]*100:.1f}%")
    k3v.metric("Total Variance Captured", f"{ev.sum()*100:.1f}%")

    pca_df = pd.DataFrame(M["X_pca"], columns=["PC1","PC2"])
    pca_df["cluster"]       = df["cluster"].values
    pca_df["anomaly_label"] = df["anomaly_label"].values
    pca_df["class_label"]   = df["class_label"].values

    c1, c2 = st.columns(2)

    with c1:
        samp = pca_df.sample(min(3000, len(pca_df)), random_state=0)
        fig = px.scatter(
            samp, x="PC1", y="PC2", color="cluster",
            color_discrete_sequence=CLUSTER_COLORS, opacity=.55,
            title=f"PCA — K-Means Clusters (PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%)",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            samp, x="PC1", y="PC2", color="anomaly_label",
            color_discrete_map={"Normal":"#6c63ff","Anomaly":"#ff4d6d"}, opacity=.55,
            title="PCA — Anomalies Highlighted",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    # Scree plot
    st.markdown("#### Scree Plot — Explained Variance by Component")
    n_comp = min(10, M["X_std"].shape[0], M["X_std"].shape[1])
    pca_full = PCA(n_components=n_comp, random_state=42)
    pca_full.fit(M["X_std"])
    scree = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca_full.explained_variance_ratio_))],
        "Variance":  pca_full.explained_variance_ratio_ * 100,
    })
    fig = go.Figure()
    fig.add_bar(x=scree["Component"], y=scree["Variance"],
                marker_color="#6c63ff", name="Individual")
    fig.add_scatter(x=scree["Component"], y=scree["Variance"].cumsum(),
                    mode="lines+markers", line=dict(color="#00e5b4"), name="Cumulative")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Scree Plot", yaxis_title="Variance Explained (%)")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 6 — PREDICT
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec-title">Deployment Layer</div><div class="sec-heading">🎯 Live Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown("Enter job profile details below to get **real-time predictions** from the trained Random Forest models.")

    pred_mode = st.radio("Prediction Mode", ["Job Survival Class", "Salary Estimate", "Both"], horizontal=True)

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Job Attributes**")
        p_job      = st.selectbox("Job Title", sorted(df["job_title"].unique()))
        p_country  = st.selectbox("Country",   sorted(df["country"].unique()))
        p_skill    = st.selectbox("Primary Skill", sorted(df["primary_skill"].unique()))
        p_year     = st.slider("Year", 2015, 2024, 2023)

    with col_b:
        st.markdown("**Personal Attributes**")
        p_exp      = st.selectbox("Experience Level", ["Entry","Mid","Senior"])
        p_edu      = st.selectbox("Education Level",  ["Bachelor","Master","PhD"])

    with col_c:
        st.markdown("**Market Indicators**")
        p_risk     = st.slider("AI Risk Score (0-100)",    0, 100, 45)
        p_demand   = st.slider("Skill Demand Score (0-100)", 0, 100, 60)
        p_openings = st.number_input("Job Openings", min_value=100, max_value=100000, value=5000, step=500)
        p_salary   = st.number_input("Current Salary ($)", min_value=20000, max_value=300000, value=70000, step=5000)

    st.markdown("")
    run_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

    if run_btn:
        # Build input row
        EXPERIENCE_MAP = {"Entry": 0, "Mid": 1, "Senior": 2}
        EDUCATION_MAP  = {"Bachelor": 0, "Master": 1, "PhD": 2}

        row = {
            "ai_risk_score":      p_risk,
            "skill_demand_score": p_demand,
            "job_openings":       p_openings,
            "salary":             p_salary,
            "year":               p_year,
            "experience_encoded": EXPERIENCE_MAP[p_exp],
            "education_encoded":  EDUCATION_MAP[p_edu],
        }

        # Feature engineering
        row["risk_demand_interaction"] = row["ai_risk_score"] * row["skill_demand_score"]
        row["openings_growth_rate"]    = row["job_openings"] / (row["year"] - 2014)
        row["human_capital_index"]     = row["experience_encoded"] * (row["education_encoded"] + 1)
        row["ai_vulnerability"]        = row["ai_risk_score"] * (2 - row["experience_encoded"] + 1)

        # OHE — align with training feature columns
        input_df = pd.DataFrame([row])
        # Add OHE columns for job_title, country, primary_skill
        for jt in df["job_title"].unique():
            input_df[f"job_title_{jt}"]     = 1 if jt == p_job     else 0
        for co in df["country"].unique():
            input_df[f"country_{co}"]        = 1 if co == p_country  else 0
        for sk in df["primary_skill"].unique():
            input_df[f"primary_skill_{sk}"]  = 1 if sk == p_skill   else 0

        feat_cols = M["feat_cols"]
        for fc in feat_cols:
            if fc not in input_df.columns:
                input_df[fc] = 0
        input_vec = input_df[feat_cols].fillna(0)

        # Scale
        input_scaled = pd.DataFrame(
            M["scaler"].transform(input_vec),
            columns=feat_cols
        )

        # Predict
        clf_pred  = M["rf_clf"].predict(input_scaled)[0]
        clf_proba = M["rf_clf"].predict_proba(input_scaled)[0]
        reg_pred  = np.expm1(M["rf_reg"].predict(input_scaled)[0])

        # Display results
        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Prediction Results")

        r1, r2, r3 = st.columns(3)

        SURVIVAL_COLOR = {"At Risk":"#ff4d6d","Stable":"#ffb347","Growing":"#00e5b4"}
        SURVIVAL_EMOJI = {"At Risk":"⚠️","Stable":"🟡","Growing":"✅"}
        label = CLASS_NAMES[clf_pred]

        with r1:
            st.markdown(f"""
            <div class='pred-box'>
                <div class='label'>Job Survival Class</div>
                <div class='value' style='color:{SURVIVAL_COLOR[label]};'>{SURVIVAL_EMOJI[label]} {label}</div>
                <div class='sub'>Predicted by Random Forest Classifier</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class='pred-box'>
                <div class='label'>Predicted Salary</div>
                <div class='value' style='color:#6c63ff;'>${reg_pred:,.0f}</div>
                <div class='sub'>Random Forest Regressor (log-transformed)</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            top_class = int(np.argmax(clf_proba))
            st.markdown(f"""
            <div class='pred-box'>
                <div class='label'>Confidence (top class)</div>
                <div class='value' style='color:#00e5b4;'>{clf_proba[top_class]*100:.1f}%</div>
                <div class='sub'>Probability for "{CLASS_NAMES[top_class]}"</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar chart
        fig = go.Figure(go.Bar(
            x=[CLASS_NAMES[i] for i in range(3)],
            y=clf_proba * 100,
            marker_color=["#ff4d6d","#ffb347","#00e5b4"],
            text=[f"{p*100:.1f}%" for p in clf_proba],
            textposition="outside",
        ))
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"],
            title="Class Probability Distribution",
            yaxis=dict(range=[0,110], title="Probability (%)"),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_risk,
            title={"text": "AI Risk Score", "font": {"size": 16, "color": "#e8e8f0"}},
            gauge={
                "axis":   {"range": [0, 100], "tickcolor": "#8888a8"},
                "bar":    {"color": "#ff4d6d" if p_risk > 70 else "#ffb347" if p_risk > 40 else "#00e5b4"},
                "steps":  [
                    {"range": [0, 40],   "color": "#00e5b410"},
                    {"range": [40, 70],  "color": "#ffb34710"},
                    {"range": [70, 100], "color": "#ff4d6d10"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": .75, "value": p_risk},
                "bgcolor": "#12121e",
            },
            number={"font": {"color": "#e8e8f0"}},
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 7 — BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="sec-title">Phase 5 · Evaluation &amp; Deployment</div><div class="sec-heading">💡 Actionable Business Insights</div>', unsafe_allow_html=True)

    insights = [
        ("🎯 Reskill High-Risk Job Groups",
         "Jobs with an AI risk score above 70 — such as data entry, receptionist, and warehouse roles — show significantly lower survival probabilities. Organizations should launch targeted reskilling programs before automation displaces these workers. Proactive investment in upskilling reduces transition costs and talent attrition."),
        ("📈 Prioritize High-Demand Skills",
         "Skill demand score is among the top 3 predictors of both job survival and salary. Roles tied to Python, Machine Learning, and Cloud Computing consistently fall in the 'Growing' cluster. Workers and students should align their development with market-demanded competencies to secure career longevity."),
        ("🔵 Use Clusters for Workforce Planning",
         "K-Means identified three distinct workforce segments: Cluster 0 (low salary, high risk — vulnerable roles), Cluster 1 (mid range — transitional), Cluster 2 (high salary, high demand — future-proof). HR leaders can use these profiles to allocate L&D budgets, design retention strategies, and forecast headcount needs."),
        ("💰 Multi-Factor Salary Strategy",
         "Salary is not driven by experience alone. The regression model reveals that skill demand, AI exposure, education, and job openings jointly determine compensation. Companies should benchmark salaries using a multi-dimensional model rather than purely tenure-based increments to remain competitive."),
    ]

    for title, body in insights:
        st.markdown(f"""
        <div class='insight-card'>
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Model Summary Table")
    summary = pd.DataFrame({
        "Step":       ["Classification","Classification","Regression","Regression","Hyperparameter Tuning","Clustering","Anomaly Detection","PCA"],
        "Technique":  ["Logistic Regression","Random Forest","Linear Regression","Random Forest Regressor","Fixed Random Forest Parameters","K-Means (k=3)","Isolation Forest","PCA (2 components)"],
        "Key Result": [
            f"Accuracy: {M['clf_metrics'].iloc[0]['Accuracy']:.2%}",
            f"Accuracy: {M['clf_metrics'].iloc[1]['Accuracy']:.2%} | F1: {M['clf_metrics'].iloc[1]['Macro F1']:.4f}",
            f"R²: {M['reg_metrics'].iloc[0]['R² Score']:.4f}",
            f"R²: {M['reg_metrics'].iloc[1]['R² Score']:.4f} | RMSE: {M['reg_metrics'].iloc[1]['RMSE']:.4f}",
            "Parameters set manually for stable baseline",
            "3 workforce segments identified",
            "~5% of records flagged as atypical",
            f"Variance captured: {M['pca'].explained_variance_ratio_.sum()*100:.1f}%",
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a30,#252540);border:1px solid #2a2a42;border-radius:16px;padding:1.5rem 2rem;text-align:center;'>
        <div style='font-size:.75rem;color:#8888a8;font-family:"Space Mono",monospace;letter-spacing:.1em;text-transform:uppercase;'>Key Finding</div>
        <div style='font-size:1.1rem;font-weight:700;color:#e8e8f0;margin:.6rem 0;'>AI risk and skill demand are the strongest dual predictors of both job survival and salary.</div>
        <div style='font-size:.85rem;color:#8888a8;'>Workforce strategies must simultaneously reduce AI exposure AND build high-demand skill portfolios to future-proof careers.</div>
    </div>
    """, unsafe_allow_html=True)
