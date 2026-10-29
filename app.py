import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="WakeWise | Wind Turbine Wake Loss",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Visual theme ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #07111f; color: #eaf2ff; }
[data-testid="stSidebar"] { background: #0c1a2b; border-right: 1px solid #20344b; }
.hero { padding: 28px 30px; border-radius: 22px; background: linear-gradient(120deg,#10283a 0%,#123d45 52%,#1b3151 100%); border: 1px solid #2c5962; margin-bottom: 20px; }
.eyebrow { color:#72e2c3; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:38px; line-height:1.12; margin:8px 0; color:#f2f7ff; }
.hero p { color:#b6c9dc; margin:0; font-size:15px; }
.metric { background:#0d1d30; border:1px solid #20364d; padding:18px 20px; border-radius:16px; min-height:110px; }
.metric-label { color:#93a9c2; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#f2f7ff; font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; margin-top:8px; }
.metric-note { color:#72e2c3; font-size:12px; margin-top:2px; }
.section-title {font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; margin:12px 0 6px;}
.muted { color:#9fb3c9; font-size:13px; }
div[data-testid="stForm"] { border:1px solid #20364d; border-radius:16px; padding:18px; background:#0b1a2b; }
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:700; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

CANDIDATE_FEATURES = [
    "V1", "V2", "Va", "D",
    "y1 (% relative to rated power)", "out1", "out2",
    "hour", "day_of_week", "month", "day_of_year"
]
TARGET_SOURCE = "y2 (% relative to rated power)"
TARGET = "wake_power_loss_pct"

def prepare_data(df):
    data = df.copy()
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], format="mixed", dayfirst=True, errors="coerce")
    for col in data.columns:
        if col != "time":
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.drop_duplicates()
    if TARGET_SOURCE not in data.columns:
        raise ValueError(f"Dataset must contain the target-source column: '{TARGET_SOURCE}'.")
    data[TARGET] = 100.0 - data[TARGET_SOURCE]
    if "time" in data.columns:
        data["hour"] = data["time"].dt.hour
        data["day_of_week"] = data["time"].dt.dayofweek
        data["month"] = data["time"].dt.month
        data["day_of_year"] = data["time"].dt.dayofyear
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    features = [c for c in CANDIDATE_FEATURES if c in data.columns]
    if not features:
        raise ValueError("No expected input columns found. Check the dataset columns.")
    features = [c for c in features if not data[c].isna().all()]
    if not features:
        raise ValueError("All available predictor columns are empty.")
    X = data[features].copy()
    y = data[TARGET].astype(float)
    return data, X, y, features

def train_models(X, y):
    if len(X) < 10:
        raise ValueError("At least 10 usable rows are recommended to train and evaluate the models.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=2, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=1, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror", n_estimators=250, learning_rate=0.05,
            max_depth=4, subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=2, tree_method="hist", eval_metric="rmse"
        )
    }
    fitted, rows, predictions = {}, [], {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = pred
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R²": r2_score(y_test, pred),
        })
    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_name = results.iloc[0]["Model"]
    return fitted, results, best_name, X_train, X_test, y_train, y_test, predictions

def card(label, value, note=""):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("""
<div class="hero">
  <div class="eyebrow">Wind energy analytics · ML prediction lab</div>
  <h1>WakeWise</h1>
  <p>Predict wake-induced power loss in wind turbines using your operational dataset.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌬️ WakeWise")
    st.caption("Wind turbine wake-loss prediction")
    page = st.radio("NAVIGATION", ["Prediction dashboard", "Model performance", "Dataset explorer", "About the model"])
    st.divider()
    st.markdown("### 1 · Upload training data")
    uploaded = st.file_uploader("Upload your turbine CSV", type=["csv"], help=f"Required target column: {TARGET_SOURCE}")
    st.caption("The app trains models from the CSV you upload. Your notebook file alone does not include the raw training data.")
    if st.button("Clear current session", use_container_width=True):
        for key in ["trained", "data", "X", "y", "features", "models", "results", "best_name", "X_test", "y_test", "predictions", "y_train"]:
            st.session_state.pop(key, None)
        st.rerun()

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        data, X, y, features = prepare_data(raw)
        with st.spinner("Training Decision Tree, Random Forest and XGBoost…"):
            models, results, best_name, X_train, X_test, y_train, y_test, predictions = train_models(X, y)
        st.session_state.update({
            "trained": True, "data": data, "X": X, "y": y, "features": features,
            "models": models, "results": results, "best_name": best_name,
            "X_test": X_test, "y_test": y_test, "predictions": predictions, "y_train": y_train
        })
        st.sidebar.success(f"Trained successfully · {len(data):,} rows")
    except Exception as e:
        st.error(f"Could not prepare/train this dataset: {e}")
        st.stop()

if not st.session_state.get("trained"):
    st.markdown("### Start with your dataset")
    st.write("Upload the same CSV used in your Colab notebook to train the three regression models and unlock the prediction dashboard.")
    c1, c2, c3 = st.columns(3)
    with c1: card("MODELS", "03", "Decision Tree · Random Forest · XGBoost")
    with c2: card("TARGET", "Wake loss %", "Calculated as 100 − y2")
    with c3: card("EVALUATION", "80 / 20", "Train/test split · random state 42")
    st.info(f"Required column: `{TARGET_SOURCE}`. Predictor columns are selected from: {', '.join(CANDIDATE_FEATURES)}. Missing predictor columns are skipped.")
    st.stop()

data = st.session_state["data"]
X = st.session_state["X"]
y = st.session_state["y"]
features = st.session_state["features"]
models = st.session_state["models"]
results = st.session_state["results"]
best_name = st.session_state["best_name"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
predictions = st.session_state["predictions"]

if page == "Prediction dashboard":
    st.markdown('<div class="section-title">Prediction dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Enter turbine operating conditions. The selected model estimates wake-induced power loss.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("#### Operating conditions")
        st.caption("Fields shown below are the predictors available in your uploaded file. Values are prefilled using dataset medians.")
        with st.form("prediction_form"):
            values = {}
            cols = st.columns(2)
            for i, feature in enumerate(features):
                series = pd.to_numeric(X[feature], errors="coerce").dropna()
                if series.empty:
                    default, low, high = 0.0, -1.0, 1.0
                else:
                    default = float(series.median())
                    low, high = float(series.quantile(0.01)), float(series.quantile(0.99))
                    if low == high:
                        low, high = float(series.min()) - 1, float(series.max()) + 1
                with cols[i % 2]:
                    values[feature] = st.number_input(
                        feature, min_value=float(min(low, default)), max_value=float(max(high, default)),
                        value=float(default), key=f"input_{feature}", help=f"Observed range approx. {low:.3g} to {high:.3g}"
                    )
            model_choice = st.selectbox("Prediction model", list(models.keys()), index=list(models.keys()).index(best_name))
            submitted = st.form_submit_button("⚡ Predict wake loss", use_container_width=True, type="primary")
    with right:
        if submitted:
            row = pd.DataFrame([values], columns=features)
            pred = float(models[model_choice].predict(row)[0])
            st.markdown("#### Prediction result")
            card("ESTIMATED WAKE-INDUCED POWER LOSS", f"{pred:.2f}%", model_choice)
            st.progress(min(max(pred / 100, 0.0), 1.0), text=f"{pred:.1f}% of rated power (visual scale)")
            st.markdown("**Interpretation**")
            if pred < 5:
                st.success("Low estimated loss on a 0–100% scale. Validate against site measurements.")
            elif pred < 15:
                st.warning("Moderate estimated loss. Compare turbine spacing and operating conditions.")
            else:
                st.error("High estimated loss. Review the input conditions and validate with measured data.")
            st.caption("This category is a UI guide, not an industry-standard threshold. Prediction quality depends on the training data and model validation.")
            st.download_button("Download this prediction (CSV)", row.assign(predicted_wake_loss_pct=pred, model=model_choice).to_csv(index=False).encode("utf-8"), "wake_loss_prediction.csv", "text/csv")

            st.markdown("#### How each feature changes the prediction")
            st.caption("One feature varies across its observed data range while the other inputs stay fixed at your selected values. This is a model sensitivity view, not proof of causation.")
            impact_feature = st.selectbox("Choose a feature to explore", features, key="impact_feature")
            observed = pd.to_numeric(X[impact_feature], errors="coerce").dropna()
            if len(observed) > 1:
                lo, hi = float(observed.quantile(0.02)), float(observed.quantile(0.98))
                if lo == hi:
                    lo, hi = float(observed.min()), float(observed.max())
                sweep = np.linspace(lo, hi, 60)
                scenario = pd.concat([row] * len(sweep), ignore_index=True)
                scenario[impact_feature] = sweep
                sweep_pred = models[model_choice].predict(scenario)
                impact_df = pd.DataFrame({impact_feature: sweep, "Predicted wake loss (%)": sweep_pred})
                impact_fig = px.line(impact_df, x=impact_feature, y="Predicted wake loss (%)",
                                     title=f"Wake-loss sensitivity to {impact_feature}", template="plotly_dark")
                impact_fig.add_scatter(x=[values[impact_feature]], y=[pred], mode="markers",
                                       name="Your selected input", marker=dict(size=11, color="#72e2c3"))
                impact_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                         margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(impact_fig, use_container_width=True)
                st.caption(f"Observed central range for this feature: {lo:.3g} to {hi:.3g}.")
        else:
            st.markdown("#### Your result will appear here")
            st.markdown("Choose realistic operating conditions and press **Predict wake loss**.")
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number", value=0, number={"suffix":"%"},
                title={"text":"Awaiting prediction"},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":"#72e2c3"},
                       "bgcolor":"#0d1d30", "bordercolor":"#29445c",
                       "steps":[{"range":[0,5]},{"range":[5,15]},{"range":[15,100]}]}
            )).update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="#eaf2ff", margin=dict(l=20,r=20,t=55,b=10)), use_container_width=True)
    st.divider()
    st.markdown("#### Dataset snapshot")
    a,b,c,d = st.columns(4)
    with a: card("USABLE ROWS", f"{len(data):,}")
    with b: card("INPUT FEATURES", str(len(features)))
    with c: card("BEST TEST MAE", f"{results.iloc[0]['MAE']:.3f}", f"{best_name}")
    with d: card("BEST TEST R²", f"{results.iloc[0]['R²']:.4f}")

elif page == "Model performance":
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    st.caption("Metrics are computed on the held-out 20% test set. Lower MAE/RMSE is better; R² closer to 1 is generally better.")
    a,b,c = st.columns(3)
    best = results.iloc[0]
    with a: card("TOP MODEL", best["Model"])
    with b: card("MAE", f"{best['MAE']:.4f}", "Wake-loss percentage points")
    with c: card("R²", f"{best['R²']:.4f}", "Held-out test set")
    st.dataframe(results.style.format({"MAE":"{:.4f}", "RMSE":"{:.4f}", "R²":"{:.4f}"}), use_container_width=True, hide_index=True)
    st.markdown("#### Which features matter most?")
    importance_model = models[best_name].named_steps["model"]
    if hasattr(importance_model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": features, "Importance": importance_model.feature_importances_}).sort_values("Importance", ascending=True)
        imp_fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         title=f"Feature importance · {best_name}", template="plotly_dark")
        imp_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis_title="", xaxis_title="Relative importance")
        st.plotly_chart(imp_fig, use_container_width=True)
    chart = px.bar(results, x="Model", y="MAE", color="Model", title="Mean Absolute Error by model", template="plotly_dark")
    chart.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(chart, use_container_width=True)
    rows = []
    for name, pred in predictions.items():
        for actual, estimated in zip(y_test.to_numpy(), pred):
            rows.append({"Model":name, "Actual wake loss (%)":actual, "Predicted wake loss (%)":estimated})
    pred_df = pd.DataFrame(rows)
    chosen = st.selectbox("Actual vs predicted plot", list(models.keys()), index=list(models.keys()).index(best_name))
    sub = pred_df[pred_df["Model"] == chosen]
    fig = px.scatter(sub, x="Actual wake loss (%)", y="Predicted wake loss (%)", title=f"Actual vs predicted · {chosen}", template="plotly_dark")
    mn = float(min(sub.iloc[:,1].min(), sub.iloc[:,2].min()))
    mx = float(max(sub.iloc[:,1].max(), sub.iloc[:,2].max()))
    fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash", color="#72e2c3"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download model metrics", results.to_csv(index=False).encode("utf-8"), "wake_loss_model_metrics.csv", "text/csv")

elif page == "Dataset explorer":
    st.markdown('<div class="section-title">Dataset explorer</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: card("ROWS", f"{len(data):,}")
    with c2: card("COLUMNS", str(data.shape[1]))
    with c3: card("MISSING TARGETS REMOVED", str(len(st.session_state["y"]) - len(y)))
    st.dataframe(data.head(100), use_container_width=True)
    st.markdown("#### Wake-loss distribution")
    fig = px.histogram(data, x=TARGET, nbins=35, marginal="box", title="Distribution of calculated wake-induced power loss", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    missing = data.isna().sum().reset_index()
    missing.columns = ["Column", "Missing values"]
    st.dataframe(missing[missing["Missing values"] > 0], use_container_width=True, hide_index=True)
    st.download_button("Download cleaned dataset", data.to_csv(index=False).encode("utf-8"), "wake_loss_cleaned_data.csv", "text/csv")

else:
    st.markdown('<div class="section-title">About the model</div>', unsafe_allow_html=True)
    st.markdown("""
    ### What is wake-induced power loss?
    When wind passes through a turbine, the rotor extracts energy and leaves a slower, more turbulent wake. A downstream turbine can therefore generate less power than it would in undisturbed wind.

    ### Modelling workflow
    1. Upload the operational CSV used in the research notebook.
    2. The target is calculated using the notebook's definition: **wake loss (%) = 100 − y2 (% relative to rated power)**.
    3. Available predictors are selected from `V1`, `V2`, `Va`, `D`, `y1 (% relative to rated power)`, `out1`, `out2` and time-derived fields.
    4. The data is split into 80% training and 20% test data with random state 42.
    5. Decision Tree, Random Forest and XGBoost regressors are trained and compared using MAE, RMSE and R².

    ### Important limitations
    - Results depend on the quality, coverage and units of your dataset.
    - This is a research/demo tool, not a certified wind-farm engineering system.
    - Inputs should be within the operating range represented in the training data.
    - Confirm that all input features are available at prediction time and do not inadvertently encode the target.
    """)
    st.warning("Before presenting performance numbers, validate the feature definitions and check for target leakage. The app intentionally reports metrics from its own held-out split rather than hard-coded notebook figures.")

st.markdown("---")
st.markdown('<div class="muted">WakeWise · Academic ML prototype · Validate predictions against measured turbine data before operational use.</div>', unsafe_allow_html=True)
