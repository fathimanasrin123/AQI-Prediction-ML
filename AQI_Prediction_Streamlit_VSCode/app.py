import io
import zipfile
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AQI Prediction", page_icon="🌫️", layout="wide")

POSSIBLE_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

st.title("🌫️ Air Quality Index (AQI) Prediction")
st.caption("Machine Learning Application | Random Forest Regression")
st.write(
    "Upload the `city_day.csv` file or the dataset ZIP containing it. "
    "This application estimates AQI from pollutant measurements in the same observation; "
    "it does not forecast AQI for a future day."
)

@st.cache_data(show_spinner=False)
def read_uploaded_dataset(file_name, file_bytes):
    """Read a CSV directly or locate city_day.csv inside an uploaded ZIP."""
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            candidates = [
                name for name in zf.namelist()
                if not name.endswith("/") and name.split("/")[-1].lower() == "city_day.csv"
            ]
            if not candidates:
                raise ValueError("Could not find city_day.csv inside the ZIP file.")
            # Prefer the shortest path if the ZIP contains more than one matching file.
            selected = sorted(candidates, key=len)[0]
            with zf.open(selected) as csv_file:
                return pd.read_csv(csv_file)
    raise ValueError("Please upload a .csv file or a .zip file.")

def make_plot():
    return plt.subplots(figsize=(9, 5))

def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    return r2_score(y_true, y_pred)

uploaded = st.file_uploader(
    "Upload dataset (city_day.csv or archive.zip)",
    type=["csv", "zip"],
    help="The ZIP should contain city_day.csv, as in your notebook."
)

if uploaded is None:
    st.info("Start by uploading your dataset. You can use the `archive.zip` from your assignment.")
    st.markdown("### Expected dataset")
    st.write("The file should include an `AQI` column and one or more pollutant columns such as PM2.5, PM10, NO2, CO, SO2 or O3.")
    st.stop()

try:
    raw_df = read_uploaded_dataset(uploaded.name, uploaded.getvalue())
except Exception as exc:
    st.error(f"Unable to read the uploaded file: {exc}")
    st.stop()

raw_df.columns = raw_df.columns.astype(str).str.strip()
st.success(f"Dataset loaded: {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

# Sidebar options
st.sidebar.header("Model settings")
test_size = st.sidebar.slider("Testing data (%)", min_value=10, max_value=40, value=20, step=5)
n_estimators = st.sidebar.selectbox("Number of trees", [100, 150, 200, 300], index=1)
max_depth = st.sidebar.selectbox("Maximum tree depth", [10, 15, 20, 25, 30], index=2)
min_samples_leaf = st.sidebar.selectbox("Minimum samples per leaf", [1, 2, 3, 4], index=1)
st.sidebar.caption("Defaults match your notebook: 80:20 split, 150 trees, depth 20, leaf size 2.")

tabs = st.tabs([
    "1. Dataset", "2. Preprocessing", "3. Visualizations",
    "4. Model & Evaluation", "5. Predict AQI", "6. Downloads"
])

with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(raw_df):,}")
    c2.metric("Columns", f"{len(raw_df.columns)}")
    c3.metric("Duplicate rows", f"{int(raw_df.duplicated().sum()):,}")
    with st.expander("Column names"):
        st.write(raw_df.columns.tolist())
    with st.expander("Data types and non-empty values"):
        info_df = pd.DataFrame({
            "Column": raw_df.columns,
            "Data type": [str(x) for x in raw_df.dtypes],
            "Non-null values": [int(x) for x in raw_df.notna().sum().values],
            "Missing values": [int(x) for x in raw_df.isna().sum().values],
        })
        st.dataframe(info_df, use_container_width=True)
    with st.expander("Missing values by column"):
        missing_df = raw_df.isna().sum().rename("Missing values").to_frame()
        st.dataframe(missing_df, use_container_width=True)
    with st.expander("Statistical summary"):
        st.dataframe(raw_df.describe(include="all").T, use_container_width=True)

# Preprocess in the same way as the notebook.
if "AQI" not in raw_df.columns:
    st.error("The uploaded dataset does not contain an 'AQI' column. Please check your file.")
    st.stop()

df = raw_df.drop_duplicates().copy()
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
df = df.dropna(subset=["AQI"]).copy()
df = df[df["AQI"].between(0, 500)].copy()

features = [col for col in POSSIBLE_FEATURES if col in df.columns]
if not features:
    st.error("No expected pollutant columns were found. Expected at least one of: " + ", ".join(POSSIBLE_FEATURES))
    st.stop()

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
features = [col for col in features if df[col].notna().any()]
if not features:
    st.error("All expected pollutant columns are empty.")
    st.stop()

df = df.dropna(subset=features, how="all").copy()
if len(df) < 10:
    st.error("There are too few usable rows to train and evaluate the model. Please upload a larger dataset.")
    st.stop()

X = df[features].copy()
y = df["AQI"].copy()

with tabs[1]:
    st.subheader("Data preprocessing")
    st.write("The application follows the notebook's preprocessing steps:")
    st.markdown(
        "- Remove duplicate rows.\n"
        "- Convert AQI and pollutant columns to numeric values.\n"
        "- Remove rows with missing AQI values.\n"
        "- Keep AQI values from 0 to 500.\n"
        "- Select available pollutant columns from the notebook's feature list.\n"
        "- Remove rows where every selected pollutant value is missing.\n"
        "- Fill remaining pollutant missing values with the training-set median inside the model pipeline."
    )
    p1, p2, p3 = st.columns(3)
    p1.metric("Usable records", f"{len(df):,}")
    p2.metric("Selected pollutant features", len(features))
    p3.metric("Missing AQI values removed", f"{int(raw_df['AQI'].isna().sum()) if pd.api.types.is_numeric_dtype(raw_df['AQI']) else int(pd.to_numeric(raw_df['AQI'], errors='coerce').isna().sum()):,}")
    st.write("**Selected input features:**", ", ".join(features))
    st.write("**Target column:** AQI")
    st.write("Missing values in selected input features:")
    st.dataframe(X.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    st.write("Cleaned data preview:")
    st.dataframe(df.head(20), use_container_width=True)

with tabs[2]:
    st.subheader("Exploratory data analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = make_plot()
        sns.histplot(data=df, x="AQI", bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Air Quality Index (AQI)")
        ax.set_xlabel("AQI")
        ax.set_ylabel("Number of observations")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        fig, ax = make_plot()
        corr = df[features + ["AQI"]].corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Between Pollutants and AQI")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Correlation indicates association, not causation.")

# Split and train once per app run using current settings.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    ))
])
with st.spinner("Training Random Forest Regression model..."):
    model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = safe_r2(y_test, y_pred)

results = pd.DataFrame({
    "Actual AQI": y_test.to_numpy(),
    "Predicted AQI": y_pred
})
results["Absolute Error"] = np.abs(results["Actual AQI"] - results["Predicted AQI"])
forest = model.named_steps["regressor"]
importance_df = pd.DataFrame({
    "Pollutant": features,
    "Importance": forest.feature_importances_
}).sort_values("Importance", ascending=False)
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
})

with tabs[3]:
    st.subheader("Train and evaluate the model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training records", f"{len(X_train):,}")
    c2.metric("Testing records", f"{len(X_test):,}")
    c3.metric("Total usable records", f"{len(X_train) + len(X_test):,}")
    st.markdown("### Evaluation metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("MSE", f"{mse:.2f}")
    m3.metric("RMSE", f"{rmse:.2f}")
    m4.metric("R² score", f"{r2:.4f}" if np.isfinite(r2) else "Not available")
    st.caption("MAE and RMSE are in AQI units. Lower values generally indicate smaller errors. R² closer to 1 generally indicates a better fit.")
    if r2 >= 0.80:
        st.info("On this test split, the model explains a high proportion of AQI variation.")
    elif r2 >= 0.50:
        st.info("On this test split, the model explains a moderate proportion of AQI variation.")
    else:
        st.warning("The model's predictive performance may need improvement.")
    st.markdown("### Actual vs predicted AQI")
    fig, ax = make_plot()
    ax.scatter(y_test, y_pred, alpha=0.5)
    lower = min(float(y_test.min()), float(np.min(y_pred)))
    upper = max(float(y_test.max()), float(np.max(y_pred)))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_title("Actual AQI vs Predicted AQI")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Distribution of prediction errors")
    errors = y_test.to_numpy() - y_pred
    fig, ax = make_plot()
    sns.histplot(errors, bins=30, kde=True, ax=ax)
    ax.axvline(0, linestyle="--")
    ax.set_title("Distribution of Prediction Errors")
    ax.set_xlabel("Actual AQI - Predicted AQI")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Feature importance")
    fig, ax = make_plot()
    sns.barplot(data=importance_df, x="Importance", y="Pollutant", ax=ax)
    ax.set_title("Feature Importance for AQI Prediction")
    ax.set_xlabel("Importance score")
    ax.set_ylabel("Pollutant")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(importance_df, use_container_width=True)
    st.markdown("### Test-set predictions")
    st.dataframe(results.head(50), use_container_width=True)
    st.caption("Feature importance describes how the trained model uses inputs; it does not prove causation.")

with tabs[4]:
    st.subheader("Predict AQI from pollutant values")
    st.write("Enter pollutant readings below. Values are prefilled with the median of each pollutant in your uploaded dataset.")
    with st.form("prediction_form"):
        input_values = {}
        medians = X.median(numeric_only=True)
        grid = st.columns(3)
        for i, feature in enumerate(features):
            default = float(medians[feature]) if pd.notna(medians[feature]) else 0.0
            with grid[i % 3]:
                input_values[feature] = st.number_input(
                    feature, min_value=0.0, value=max(0.0, default),
                    step=0.1, format="%.3f", key=f"input_{feature}"
                )
        submitted = st.form_submit_button("Predict AQI", type="primary")
    if submitted:
        input_df = pd.DataFrame([input_values], columns=features)
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted AQI: {prediction:.2f}")
        st.caption("This is a model estimate based on the pollutant values entered, not an official AQI measurement.")
        if prediction <= 50:
            category = "Good"
        elif prediction <= 100:
            category = "Satisfactory"
        elif prediction <= 200:
            category = "Moderate"
        elif prediction <= 300:
            category = "Poor"
        elif prediction <= 400:
            category = "Very Poor"
        else:
            category = "Severe"
        st.write(f"Approximate category using common Indian AQI bands: **{category}**")
        st.caption("Category bands are a simplified display; use official CPCB guidance for formal interpretation.")

with tabs[5]:
    st.subheader("Download outputs")
    st.write("Download the generated results from this run.")
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    st.download_button(
        "Download trained model (.pkl)",
        data=model_buffer.getvalue(),
        file_name="aqi_random_forest_model.pkl",
        mime="application/octet-stream"
    )
    st.download_button(
        "Download prediction results (.csv)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="aqi_prediction_results.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download feature importance (.csv)",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="aqi_feature_importance.csv",
        mime="text/csv"
    )
    st.download_button(
        "Download model metrics (.csv)",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="aqi_model_metrics.csv",
        mime="text/csv"
    )
