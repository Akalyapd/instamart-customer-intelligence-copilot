import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.base import clone
import joblib

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned data
orders = pd.read_csv(BASE_DIR / "data" / "clean" / "orders.csv")
customers = pd.read_csv(BASE_DIR / "data" / "clean" / "customers.csv")
products = pd.read_csv(BASE_DIR / "data" / "clean" / "products.csv")

# Load raw orders separately for leakage-free delivery-time features
raw_orders = pd.read_csv(BASE_DIR / "data" / "raw" / "orders.csv")

raw_orders["order_ts"] = pd.to_datetime(
    raw_orders["order_ts"], errors="coerce"
)

# Apply the same order deduplication rules used during data cleaning
raw_orders = raw_orders.drop_duplicates()
raw_orders = raw_orders.sort_values("order_ts")
raw_orders = raw_orders.drop_duplicates(
    subset=["order_id"], keep="last"
)

# Convert delivery times and invalidate negative values
raw_orders["delivery_minutes"] = pd.to_numeric(
    raw_orders["delivery_minutes"], errors="coerce"
)

raw_orders.loc[
    raw_orders["delivery_minutes"] < 0, "delivery_minutes"
] = float("nan")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Define cutoff and lapse window
cutoff_date = pd.Timestamp("2025-09-01")
lapse_end_date = pd.Timestamp("2025-10-01")

print("Orders shape:", orders.shape)
print("Customers shape:", customers.shape)
print("Cutoff date:", cutoff_date)
print("Lapse window:", cutoff_date, "to", lapse_end_date)

# Separate completed orders before the cutoff
pre_cutoff_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff_date)
].copy()

# Customers who purchased before the cutoff
cohort_customer_ids = pre_cutoff_orders["customer_id"].unique()

# Create the cohort
cohort = customers[
    customers["customer_id"].isin(cohort_customer_ids)
].copy()

# Completed orders during September
september_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= cutoff_date) &
    (orders["order_ts"] < lapse_end_date)
]

# Identify customers who purchased during September
september_customer_ids = september_orders["customer_id"].unique()

# Assign lapse labels
cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_customer_ids)
).astype(int)

print("\nCohort shape:", cohort.shape)
print("\nLapse counts:")
print(cohort["lapsed"].value_counts().sort_index())

print("\nLapse percentages:")
print((cohort["lapsed"].value_counts(normalize=True) * 100).round(2))

# All orders before the cutoff (completed and cancelled)
pre_cutoff_all_orders = orders[
    orders["order_ts"] < cutoff_date
].copy()

# Completed orders before the cutoff
pre_cutoff_completed = pre_cutoff_orders.copy()

# Completed-order behaviour features
completed_features = pre_cutoff_completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean")
).reset_index()

# Prepare delivery-time features using only pre-cutoff raw orders
pre_cutoff_raw_completed = raw_orders[
    (raw_orders["status"] == "completed") &
    (raw_orders["order_ts"] < cutoff_date)
].copy()

pre_cutoff_raw_completed = pre_cutoff_raw_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Calculate category medians using pre-cutoff orders only
pre_cutoff_category_medians = (
    pre_cutoff_raw_completed.groupby("category")["delivery_minutes"]
    .transform("median")
)

pre_cutoff_raw_completed["delivery_minutes"] = (
    pre_cutoff_raw_completed["delivery_minutes"]
    .fillna(pre_cutoff_category_medians)
)

# Calculate average delivery time for each customer
delivery_features = (
    pre_cutoff_raw_completed.groupby("customer_id")
    .agg(avg_delivery=("delivery_minutes", "mean"))
    .reset_index()
)

# Add delivery-time features to completed-order features
completed_features = completed_features.merge(
    delivery_features,
    on="customer_id",
    how="left"
)

# Recency: days from cutoff to last completed order
completed_features["recency"] = (
    cutoff_date - completed_features["last_order_ts"]
).dt.days

# Cancellation rate using all final-status orders before cutoff
order_status_counts = pre_cutoff_all_orders.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum())
).reset_index()

order_status_counts["cancellation_rate"] = (
    order_status_counts["cancelled_orders"] /
    order_status_counts["total_orders"]
)

# Most frequently ordered product category
category_orders = pre_cutoff_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = (
    category_orders.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
)

favourite_category = category_counts.drop_duplicates(
    subset="customer_id"
)[["customer_id", "category"]].rename(
    columns={"category": "favourite_category"}
)

# Combine all features with customer information and labels
model_data = cohort.merge(
    completed_features,
    on="customer_id",
    how="left"
)

model_data = model_data.merge(
    order_status_counts[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left"
)

model_data = model_data.merge(
    favourite_category,
    on="customer_id",
    how="left"
)

# Customer tenure at the cutoff date
model_data["tenure_days"] = (
    cutoff_date - model_data["signup_date"]
).dt.days

# Display feature information
feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days"
]

print("\nModel data shape:", model_data.shape)
print("\nFeature columns:")
print(model_data[feature_columns].head())

print("\nMissing values in features:")
print(model_data[feature_columns].isnull().sum())

# Separate features and target
X = model_data[feature_columns].copy()
y = model_data["lapsed"].copy()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Baseline: predict every customer as not lapsed
baseline_predictions = [0] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

print("\nBaseline results:")
print("Accuracy:", round(baseline_accuracy, 4))
print("Lapsed class F1:", round(baseline_f1, 4))

# Separate numerical and categorical columns
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data without fitting
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# Compare training scaler mean with full-cohort frequency mean
training_frequency_mean = (
    preprocessor
    .named_transformers_["num"]
    .named_steps["scaler"]
    .mean_[1]
)

full_frequency_mean = model_data["frequency"].mean()

print("\nTraining frequency mean:", training_frequency_mean)
print("Full cohort frequency mean:", full_frequency_mean)

# Confirm no missing values in original features
print("\nMissing values in training features:")
print(X_train.isnull().sum().sum())

print("\nMissing values in testing features:")
print(X_test.isnull().sum().sum())

# Logistic Regression pipeline
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Hyperparameter grid
lr_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

# Grid search using 5-fold cross-validation
lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train using training data only
lr_grid.fit(X_train, y_train)

print("\nLogistic Regression results:")
print("Best parameters:", lr_grid.best_params_)
print("Best CV F1:", round(lr_grid.best_score_, 4))

# Predictions on untouched test data
lr_predictions = lr_grid.predict(X_test)

print("Test accuracy:", round(
    accuracy_score(y_test, lr_predictions), 4
))
print("Test lapsed F1:", round(
    f1_score(y_test, lr_predictions, zero_division=0), 4
))

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Hyperparameter grid
rf_param_grid = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

# Grid search with 5-fold cross-validation
rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train using training data only
rf_grid.fit(X_train, y_train)

print("\nRandom Forest results:")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 4))

# Predictions on untouched test data
rf_predictions = rf_grid.predict(X_test)

print("Test accuracy:", round(
    accuracy_score(y_test, rf_predictions), 4
))
print("Test lapsed F1:", round(
    f1_score(y_test, rf_predictions, zero_division=0), 4
))

# Detailed evaluation of both models
models = {
    "Logistic Regression": lr_predictions,
    "Random Forest": rf_predictions
}

for model_name, predictions in models.items():
    print(f"\n{'=' * 45}")
    print(model_name)
    print("=" * 45)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Lapsed", "Lapsed"],
        zero_division=0
    ))

    # Oversample lapsed customers in training data only

train_data = X_train.copy()
train_data["lapsed"] = y_train

lapsed_rows = train_data[train_data["lapsed"] == 1]

lapsed_oversampled = resample(
    lapsed_rows,
    replace=True,
    n_samples=570 - 354,
    random_state=42
)

balanced_train = pd.concat(
    [train_data, lapsed_oversampled],
    ignore_index=True
)

X_train_balanced = balanced_train[feature_columns]
y_train_balanced = balanced_train["lapsed"]

print("\nOversampled training distribution:")
print(y_train_balanced.value_counts())

# Refit the best Logistic Regression model
oversampled_lr = clone(lr_grid.best_estimator_)

oversampled_lr.fit(X_train_balanced, y_train_balanced)

oversampled_predictions = oversampled_lr.predict(X_test)

print("\nLogistic Regression after oversampling:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, oversampled_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    oversampled_predictions,
    target_names=["Not Lapsed", "Lapsed"],
    zero_division=0
))

# Feature importance from the selected Random Forest model

rf_model = rf_grid.best_estimator_

rf_feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
rf_importances = rf_model.named_steps["classifier"].feature_importances_

rf_importance_df = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_importances
})

rf_importance_df = rf_importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 5 Random Forest feature importances:")
print(rf_importance_df.head(5).to_string(index=False))


# Logistic Regression coefficient interpretation

lr_model = lr_grid.best_estimator_

lr_feature_names = lr_model.named_steps["preprocessor"].get_feature_names_out()
lr_coefficients = lr_model.named_steps["classifier"].coef_[0]

lr_coefficient_df = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_coefficients
})

lr_coefficient_df["absolute_value"] = lr_coefficient_df["coefficient"].abs()

lr_coefficient_df = lr_coefficient_df.sort_values(
    by="absolute_value",
    ascending=False
)

print("\nTop 3 Logistic Regression coefficients:")
print(
    lr_coefficient_df[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)

# Save the selected Random Forest model

final_model = rf_grid.best_estimator_

model_path = Path("outputs/lapse_model.joblib")
joblib.dump(final_model, model_path)

print("\nFinal model saved:", model_path)


# Reload the saved model and verify predictions

loaded_model = joblib.load(model_path)

original_predictions = final_model.predict(X_test)
loaded_predictions = loaded_model.predict(X_test)

print(
    "\nSaved model predictions match:",
    (original_predictions == loaded_predictions).all()
)


# Create retention list for Premium customers

segments = pd.read_csv("outputs/customer_segments.csv")

# Predict lapse probabilities for test customers
lapse_probabilities = loaded_model.predict_proba(X_test)[:, 1]

retention_data = pd.DataFrame({
    "customer_id": model_data.loc[X_test.index, "customer_id"].values,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.values
})

# Identify Premium customers
premium_customers = segments[
    segments["is_premium"].astype(str).str.lower() == "true"
][["customer_id", "is_premium"]]

retention_data = retention_data.merge(
    premium_customers,
    on="customer_id",
    how="inner"
)

# Rank by predicted lapse probability
retention_data = retention_data.sort_values(
    by="lapse_probability",
    ascending=False
).head(10)

# Keep only the required columns
retention_data = retention_data[
    ["customer_id", "lapse_probability", "actually_lapsed"]
]

retention_path = Path("outputs/retention_list.csv")
retention_data.to_csv(retention_path, index=False)

print("\nTop Premium customers for retention:")
print(retention_data.to_string(index=False))

print("\nRetention list saved:", retention_path)

print(
    "\nActually lapsed customers in retention list:",
    retention_data["actually_lapsed"].sum()
)