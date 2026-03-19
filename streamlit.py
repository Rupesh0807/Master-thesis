import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =========================
# Utility: Clean features
# =========================
def clean_features(X):
    X = X.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X = X.loc[:, X.var() != 0]
    return X

# =========================
# Utility: Safe date handler
# =========================
def get_safe_time_index(df, date_col):
    try:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.isna().all():
            raise ValueError
        return dt
    except Exception:
        return pd.RangeIndex(start=0, stop=len(df), step=1)

st.title("📈 AI-Driven Portfolio Analytics & Forecasting Dashboard")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    # =========================
    # LOAD & CLEAN DATA
    # =========================
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=lambda x: x.strip())
    df = df.ffill().fillna(0)

    date_col = df.columns[0]
    columns = df.columns[1:]

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    target = st.selectbox("🎯 Select target stock", columns)

    # =========================
# =========================
# STEP 1 — SAFE FEATURE SELECTION
# =========================
X = df[columns].select_dtypes(include=[np.number]).copy()

# Remove target safely
if target in X.columns:
    X = X.drop(columns=[target])

y = pd.to_numeric(df[target], errors="coerce")

# Remove invalid rows
valid_idx = y.notna()
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# 🚨 IMPORTANT CHECKS
if X.shape[1] == 0:
    st.error("❌ No numeric features available after selecting target.")
    st.stop()

if len(X) < 5:
    st.error("❌ Not enough data to train model.")
    st.stop()

# Debug (optional but useful)
st.write("Selected Features:", X.columns.tolist())
st.write("Shape of X:", X.shape)
    # =========================
    # STEP 2 — VIF FILTER
    # =========================
    X_vif = add_constant(df[top_features])
    vif_df = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
    })

    st.subheader("🧮 Variance Inflation Factor (VIF)")
    st.write(vif_df)

    vif_cutoff = st.slider("VIF cutoff", 2.0, 20.0, 10.0)
    selected = vif_df[vif_df["VIF"] < vif_cutoff]["Feature"].tolist()
    if "const" in selected:
        selected.remove("const")

    st.success(f"✔ {len(selected)} features retained after VIF filtering")

    X_features = df[selected].select_dtypes(include=[np.number]).copy()

    if X_features.shape[1] == 0:
        st.error("❌ No features left after VIF filtering.")
        st.stop()

    # =========================
    # STEP 3 — TRAIN / TEST SPLIT
    # =========================
    split = int(len(df) * 0.9)

    X_train = clean_features(X_features.iloc[:split])
    X_test = clean_features(X_features.iloc[split:])
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # =========================
    # STEP 4 — MODEL COMPARISON
    # =========================
    st.subheader("🤖 Model Comparison (Train vs Test MAPE)")

    models = {
        "Bayesian Ridge": BayesianRidge(),
        "LassoCV": LassoCV(cv=5),
        "RidgeCV": RidgeCV(cv=5),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results.append([
            name,
            round(mean_absolute_percentage_error(y_train, train_pred) * 100, 2),
            round(mean_absolute_percentage_error(y_test, test_pred) * 100, 2)
        ])

    perf_df = pd.DataFrame(results, columns=["Model", "Train MAPE %", "Test MAPE %"])
    st.dataframe(perf_df)

    # =========================
    # STEP 5 — FINAL OLS + p-VALUE FILTER
    # =========================
    st.subheader("🧾 Final Regression (Auto-drop p > 0.05)")

    X_reg = add_constant(clean_features(X_features))
    ols = OLS(y, X_reg).fit()

    pvals = ols.pvalues.drop("const", errors="ignore")
    sig_features = pvals[pvals < 0.05].index.tolist()

    if len(sig_features) == 0:
        st.error("❌ No statistically significant variables (p < 0.05)")
    else:
        st.write("✔ Significant variables:")
        st.write(sig_features)

        X_final = add_constant(clean_features(df[sig_features]))
        final_ols = OLS(y, X_final).fit()
        st.write(final_ols.summary())

        fitted = final_ols.predict(X_final)

        # =========================
        # STEP 6 — FUTURE FORECAST (DATE SAFE)
        # =========================
        st.subheader("🔮 Future Forecast")

        future_steps = st.slider("Forecast periods ahead", 1, 24, 6)
        last_row = clean_features(df[sig_features].iloc[-1:])

        forecasts = []
        for _ in range(future_steps):
            future_in = add_constant(last_row, has_constant="add")
            pred = final_ols.predict(future_in).iloc[0]
            forecasts.append(pred)

        # Safe time index
        time_index = get_safe_time_index(df, date_col)

        if isinstance(time_index, pd.DatetimeIndex):
            last_time = time_index.iloc[-1]
            future_index = pd.date_range(
                start=last_time,
                periods=future_steps + 1,
                freq="M"
            )[1:]
        else:
            future_index = range(len(df), len(df) + future_steps)

        forecast_df = pd.DataFrame({
            "Time": future_index,
            "Forecast": forecasts
        }).set_index("Time")

        # =========================
        # STEP 7 — FINAL CHARTS
        # =========================
        st.subheader("📈 Actual vs Fitted vs Forecast")

        history_df = pd.DataFrame({
            "Time": time_index,
            "Actual": y,
            "Fitted": fitted
        }).set_index("Time")

        st.line_chart(history_df)
        st.line_chart(forecast_df)

else:
    st.info("👆 Upload a CSV file to begin")

