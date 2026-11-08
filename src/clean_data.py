
import os
import sqlite3
import pandas as pd

# Project folders
RAW = "data/raw"
CLEAN = "data/clean"
os.makedirs(CLEAN, exist_ok=True)

# -----------------------------
# 1. LOAD RAW DATA
# -----------------------------
customers = pd.read_csv(f"{RAW}/customers.csv")
products = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")

print("\nRAW DATA AUDIT")
for name, df in [
    ("Customers", customers),
    ("Products", products),
    ("Orders", orders)
]:
    print(f"\n{name}: {df.shape}")
    print(df.dtypes)
    print("Missing values:")
    print(pd.DataFrame({
        "count": df.isna().sum(),
        "percentage": (df.isna().mean() * 100).round(2)
    }))

print("\nORDER AUDIT")
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", (orders["status"] == "pending").sum())
print("Distinct order IDs:", orders["order_id"].nunique())
print("Non-numeric amounts:",
      pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum())
print("Negative delivery times:",
      (pd.to_numeric(orders["delivery_minutes"], errors="coerce") < 0).sum())
print("Blank delivery times:",
      orders["delivery_minutes"].isna().sum()
      + orders["delivery_minutes"].astype(str).str.strip().eq("").sum())

print("\nCUSTOMER AUDIT")
print("Blank ages:", customers["age"].isna().sum()
      + customers["age"].astype(str).str.strip().eq("").sum())
ages = pd.to_numeric(customers["age"], errors="coerce")
print("Ages outside 10-100:", ((ages < 10) | (ages > 100)).sum())
print("Raw city spellings:", customers["city"].nunique())
print("Raw gender spellings:",
      customers.loc[customers["gender"].astype(str).str.strip() != "",
                    "gender"].nunique())
print("Blank genders:",
      customers["gender"].isna().sum()
      + customers["gender"].astype(str).str.strip().eq("").sum())

# -----------------------------
# 2. CLEAN CUSTOMERS
# -----------------------------
customers["city"] = customers["city"].astype(str).str.strip().str.title()

customers["gender"] = customers["gender"].fillna("").astype(str).str.strip().str.lower()

gender_map = {
    "male": "male", "m": "male",
    "female": "female", "f": "female"
}
customers["gender"] = customers["gender"].map(gender_map).fillna("unknown")

customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    (customers["age"] < 10) | (customers["age"] > 100), "age"
] = pd.NA

customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# -----------------------------
# 3. CLEAN PRODUCTS
# -----------------------------
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# -----------------------------
# 4. CLEAN ORDERS
# -----------------------------
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# Keep the latest record for each order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset="order_id", keep="last").copy()

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["quantity"] = pd.to_numeric(orders["quantity"])
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"])

# Add product price and category
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

# Fill missing order amounts using product price
missing_amount = orders["order_amount"].isna()
orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
).round(2)

# Remove impossible negative delivery times
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

# Fill missing delivery times only for completed orders
completed = orders["status"] == "completed"
category_median = orders.loc[completed].groupby("category")[
    "delivery_minutes"
].transform("median")

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"].fillna(category_median)
)

# Keep the required order columns
orders = orders[
    [
        "order_id", "customer_id", "product_id", "quantity",
        "order_ts", "status", "discount_pct", "order_amount",
        "delivery_minutes"
    ]
]

# -----------------------------
# 5. SAVE CLEAN CSV FILES
# -----------------------------
customers.to_csv(f"{CLEAN}/customers.csv", index=False)
products.to_csv(f"{CLEAN}/products.csv", index=False)
orders.to_csv(f"{CLEAN}/orders.csv", index=False)

# -----------------------------
# 6. CREATE SQLITE DATABASE
# -----------------------------
db = sqlite3.connect("data/instamart.db")

customers.to_sql("customers", db, if_exists="replace", index=False)
products.to_sql("products", db, if_exists="replace", index=False)
orders.to_sql("orders", db, if_exists="replace", index=False)

db.close()

# -----------------------------
# 7. VALIDATE REFERENCE CHECKPOINTS
# -----------------------------
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309
assert orders["order_id"].nunique() == 12309
assert (orders["status"] == "pending").sum() == 0
assert orders["order_amount"].isna().sum() == 0
assert orders.loc[
    orders["status"] == "completed", "delivery_minutes"
].isna().sum() == 0
assert orders.loc[
    orders["status"] == "cancelled", "delivery_minutes"
].isna().sum() == 1199

completed_revenue = orders.loc[
    orders["status"] == "completed", "order_amount"
].sum()

assert round(completed_revenue, 2) == 2998668.80
assert customers["age"].isna().sum() == 0
assert customers["city"].nunique() == 6
assert set(customers["gender"].unique()) == {
    "male", "female", "unknown"
}

print("\nCompleted orders:", (orders["status"] == "completed").sum())
print("Cancelled orders:", (orders["status"] == "cancelled").sum())
print("Completed revenue: INR", round(completed_revenue, 2))

print("\nALL CHECKPOINTS PASSED")
