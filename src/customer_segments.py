
import pandas as pd

# Load cleaned orders
orders = pd.read_csv("data/clean/orders.csv")

# Keep only completed orders
orders = orders[orders["status"] == "completed"].copy()

# Convert order timestamp into date format
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Snapshot date
snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# Calculate RFM values for each customer
rfm = orders.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in whole days
rfm["recency"] = (snapshot_date - rfm["last_order"]).dt.days

# Keep required columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Display RFM results
print("RFM ANALYSIS")
print("Total customers:", len(rfm))
print("\nRFM Summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

# Save RFM data
rfm.to_csv("outputs/rfm_analysis.csv", index=False)

print("\nRFM analysis saved successfully!")
