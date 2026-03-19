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
    # LOAD DATA
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
    # STEP 1 — SAFE FEATURE SELECTION
    # =========================
    X = df[columns].select_dtypes(include=[np.number]).copy()

    if target in X.columns:
        X = X.drop(columns=[target])

    y = pd.to_numeric(df[target], errors="coerce")

    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if X.shape[1] == 0:
        st.error("❌ No numeric features available.")
        st.stop()

    if len(X) < 5:
        st.error("❌ Not enough data.")
        st.stop()

    st.write("Selected Features:", X.columns.tolist())

    # =========================
    # STEP 2 — FEATURE IMPORTANCE (RESTORED)
    # =========================
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("🌲 Feature Importance")
    st.dataframe(importance.head(20))

    top_features = importance.head(20)["Feature"].tolist()

    # =========================
    # STEP 3 — VIF FILTER
    # =========================
    X_vif = add_constant(df[top_features])

    vif_df = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
    })

    st.subheader("🧮 VIF")
    st.write(vif_df)

    selected = vif_df[vif_df["VIF"] < 100]["Feature"].tolist()

    if "const" in selected:
        selected.remove("const")

    X_features = df[selected].select_dtypes(include=[np.number]).copy()

    if X_features.shape[1] == 0:
        st.error("❌ No features after VIF.")
        st.stop()

    # =========================
    # STEP 4 — TRAIN TEST
    # =========================
    split = int(len(df) * 0.9)

    X_train = clean_features(X_features.iloc[:split])
    X_test = clean_features(X_features.iloc[split:])

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # =========================
    # STEP 5 — MODEL COMPARISON
    # =========================
    st.subheader("🤖 Model Comparison")

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

    st.dataframe(pd.DataFrame(results, columns=["Model", "Train MAPE", "Test MAPE"]))

    # =========================
    # STEP 6 — FINAL REGRESSION
    # =========================
    st.subheader("🧾 Final Regression")

    X_reg = add_constant(clean_features(X_features))
    ols = OLS(y, X_reg).fit()

    st.write(ols.summary())

else:
    st.info("👆 Upload CSV to start")
