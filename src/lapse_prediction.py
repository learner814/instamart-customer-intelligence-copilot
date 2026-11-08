
import pandas as pd

# Load clean data
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Define cutoff and September end
cutoff = pd.Timestamp("2025-09-01")
september_end = pd.Timestamp("2025-10-01")

# Orders before September: used for features
before = orders[
    (orders["order_ts"] < cutoff) &
    (orders["status"].isin(["completed", "cancelled"]))
].copy()

# Completed orders before September
completed = before[before["status"] == "completed"].copy()

# Customers with at least one completed order before cutoff
cohort_ids = completed["customer_id"].unique()

# Create labels using September completed orders
september_orders = orders[
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] < september_end) &
    (orders["status"] == "completed")
]

active_ids = set(september_orders["customer_id"])

# Create customer feature table
features = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

# Recency
features["recency"] = (
    cutoff - features["last_order"]
).dt.days

# Cancellation rate
total_orders = before.groupby("customer_id").size()
cancelled_orders = before[
    before["status"] == "cancelled"
].groupby("customer_id").size()

features["cancellation_rate"] = (
    cancelled_orders.reindex(features["customer_id"]).fillna(0).to_numpy()
    / total_orders.reindex(features["customer_id"]).to_numpy()
)

# Favourite product category
products = pd.read_csv("data/clean/products.csv")

completed = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite = (
    completed.groupby("customer_id")["category"]
    .agg(lambda x: x.mode().iloc[0])
    .rename("favourite_category")
)

features = features.merge(
    favourite,
    on="customer_id",
    how="left"
)

# Add customer information
features = features.merge(
    customers[["customer_id", "city", "age", "gender", "signup_date"]],
    on="customer_id",
    how="left"
)

# Customer tenure
features["tenure"] = (
    cutoff - features["signup_date"]
).dt.days

# Lapse label: 1 means no completed September order
features["lapsed"] = (
    ~features["customer_id"].isin(active_ids)
).astype(int)

# Remove unnecessary columns
features = features.drop(
    columns=["last_order", "signup_date"]
)

# Display checkpoints
print("LAPSE COHORT RESULTS")
print("Total customers:", len(features))
print("Lapsed customers:", features["lapsed"].sum())
print(
    "Lapse percentage:",
    round(features["lapsed"].mean() * 100, 2), "%"
)

print("\nFEATURES")
print(features.head())

print("\nMissing values:")
print(features.isnull().sum())

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Separate input features and target
X = features.drop(columns=["customer_id", "lapsed"])
y = features["lapsed"]

# Customer IDs are kept separately for later use
customer_ids = features["customer_id"]

# Define numeric and categorical columns
numeric_columns = [
    "frequency",
    "monetary",
    "avg_discount",
    "avg_delivery",
    "recency",
    "cancellation_rate",
    "age",
    "tenure"
]

categorical_columns = [
    "city",
    "gender",
    "favourite_category"
]

# Confirm there are no missing values
assert X.isnull().sum().sum() == 0
print("\nNo missing values in model features.")

# Split into training and testing data
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    customer_ids,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTRAIN TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Preprocessing: scale numbers and encode categories
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ), categorical_columns)
    ]
)

# Baseline: always predict not lapsed
baseline_predictions = [0] * len(y_test)

print("\nBASELINE RESULTS")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print(
    "Lapsed F1:",
    f1_score(y_test, baseline_predictions, zero_division=0)
)

# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Logistic Regression parameter grid
logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Logistic Regression...")
logistic_search.fit(X_train, y_train)

print("Best Logistic Parameters:", logistic_search.best_params_)
print("Best Logistic CV F1:", logistic_search.best_score_)

# Random Forest pipeline
forest_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Random Forest parameter grid
forest_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

forest_search = GridSearchCV(
    forest_pipeline,
    forest_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Random Forest...")
forest_search.fit(X_train, y_train)

print("Best Forest Parameters:", forest_search.best_params_)
print("Best Forest CV F1:", forest_search.best_score_)

# Compare models on untouched test data
models = {
    "Logistic Regression": logistic_search.best_estimator_,
    "Random Forest": forest_search.best_estimator_
}

print("\nFINAL TEST RESULTS")

for name, model in models.items():
    predictions = model.predict(X_test)

    print("\n", name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1 Score:", f1_score(y_test, predictions, zero_division=0))

# Show training-only scaler mean
fitted_scaler = (
    logistic_search.best_estimator_
    .named_steps["preprocess"]
    .named_transformers_["num"]
)

print("\nFREQUENCY MEAN COMPARISON")
print("Training scaler mean:", fitted_scaler.mean_[0])
print("Full cohort mean:", X["frequency"].mean())

print(
    "The means differ because the scaler is fitted only "
    "on training rows, not on the full dataset."
)
# CLASS BALANCE EXPERIMENT

print("\nCLASS BALANCE EXPERIMENT")

# Check original training class counts
print("\nOriginal training class counts:")
print(y_train.value_counts())

# Separate majority and minority classes
majority_class = y_train.value_counts().idxmax()
minority_class = y_train.value_counts().idxmin()

# Find how many extra minority rows are needed
majority_count = (y_train == majority_class).sum()
minority_count = (y_train == minority_class).sum()

# Select minority-class training rows
minority_X = X_train[y_train == minority_class]
minority_y = y_train[y_train == minority_class]

# Duplicate minority rows until both classes are equal
extra_X = minority_X.sample(
    n=majority_count - minority_count,
    replace=True,
    random_state=42
)

extra_y = minority_y.loc[extra_X.index]

# Combine original training data with duplicated rows
X_train_balanced = pd.concat([X_train, extra_X])
y_train_balanced = pd.concat([y_train, extra_y])

print("\nBalanced training class counts:")
print(y_train_balanced.value_counts())

# Train Logistic Regression using the best C value
balanced_logistic = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        C=logistic_search.best_params_["model__C"],
        max_iter=1000
    ))
])

balanced_logistic.fit(X_train_balanced, y_train_balanced)

# Predict using the original, unchanged test data
balanced_predictions = balanced_logistic.predict(X_test)

print("\nBALANCED MODEL TEST RESULTS")
print("Precision:", precision_score(
    y_test, balanced_predictions, zero_division=0
))
print("Recall:", recall_score(
    y_test, balanced_predictions, zero_division=0
))
print("F1 Score:", f1_score(
    y_test, balanced_predictions, zero_division=0
))

print("\nCOMPARISON")
print("Original Logistic Recall:", recall_score(
    y_test, logistic_search.best_estimator_.predict(X_test),
    zero_division=0
))
print("Balanced Logistic Recall:", recall_score(
    y_test, balanced_predictions, zero_division=0
))
# MODEL INTERPRETATION

print("\nMODEL INTERPRETATION")

# Random Forest: top 5 important features
forest_model = forest_search.best_estimator_.named_steps["model"]

feature_names = (
    forest_search.best_estimator_
    .named_steps["preprocess"]
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": forest_model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(importance.head(5).to_string(index=False))

# Logistic Regression: three strongest coefficients
logistic_model = logistic_search.best_estimator_.named_steps["model"]

coefficients = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logistic_model.coef_[0]
})

coefficients["direction"] = coefficients["coefficient"].apply(
    lambda x: "Positive" if x > 0 else "Negative"
)

coefficients["absolute_value"] = coefficients["coefficient"].abs()

top_coefficients = coefficients.sort_values(
    by="absolute_value",
    ascending=False
).head(3)

print("\nTOP 3 LOGISTIC REGRESSION COEFFICIENTS")
print(
    top_coefficients[
        ["feature", "coefficient", "direction"]
    ].to_string(index=False)
)
# SAVE MODEL AND CREATE RETENTION LIST

import joblib

print("\nSAVING FINAL MODEL")

# Select the best model: Random Forest
best_model = forest_search.best_estimator_

# Save model
joblib.dump(best_model, "outputs/lapse_model.joblib")
print("Model saved successfully!")

# Reload saved model
loaded_model = joblib.load("outputs/lapse_model.joblib")

# Verify predictions
original_predictions = best_model.predict(X_test)
reloaded_predictions = loaded_model.predict(X_test)

print(
    "Reloaded model predictions match:",
    (original_predictions == reloaded_predictions).all()
)

# Get lapse probabilities for test customers
lapse_probabilities = loaded_model.predict_proba(X_test)[:, 1]

# Create test customer table
retention_data = pd.DataFrame({
    "customer_id": id_test.to_numpy(),
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.to_numpy()
})

# Load premium customer information
segments = pd.read_csv("outputs/customer_segments.csv")

# Select premium customers
premium_customers = segments[
    segments["is_premium"] == True
][["customer_id"]]

# Keep only premium customers from the test set
retention_data = retention_data.merge(
    premium_customers,
    on="customer_id",
    how="inner"
)

# Rank by highest lapse probability
retention_data = retention_data.sort_values(
    by="lapse_probability",
    ascending=False
)

# Select top 10 customers
retention_list = retention_data[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
].head(10)

# Save retention list
retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\nTOP PREMIUM CUSTOMER RETENTION LIST")
print(retention_list.to_string(index=False))

print("\nRetention list saved successfully!")
print("Number of premium customers selected:", len(retention_list))
print(
    "Actually lapsed customers in retention list:",
    retention_list["actually_lapsed"].sum()
)