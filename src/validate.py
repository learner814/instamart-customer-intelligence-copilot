
import pandas as pd
import sqlite3

# Load cleaned CSV files
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Check 1: Customer count
assert len(customers) == 1200

# Check 2: Product count
assert len(products) == 40

# Check 3: Order count
assert len(orders) == 12309

# Check 4: No duplicate order IDs
assert orders["order_id"].nunique() == 12309

# Check 5: No pending orders
assert (orders["status"] == "pending").sum() == 0

# Check 6: No missing order amounts
assert orders["order_amount"].isna().sum() == 0

# Check 7: Completed orders have delivery times
assert orders.loc[
    orders["status"] == "completed", "delivery_minutes"
].isna().sum() == 0

# Check 8: Cancelled orders have missing delivery times
assert orders.loc[
    orders["status"] == "cancelled", "delivery_minutes"
].isna().sum() == 1199

# Check 9: Completed revenue
revenue = orders.loc[
    orders["status"] == "completed", "order_amount"
].sum()

assert round(revenue, 2) == 2998668.80

# Check 10: Customer ages are complete
assert customers["age"].isna().sum() == 0

# Check 11: Six unique cities
assert customers["city"].nunique() == 6

# Check 12: SQLite database tables
connection = sqlite3.connect("data/instamart.db")

tables = pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table'",
    connection
)

assert {"customers", "products", "orders"}.issubset(
    set(tables["name"])
)

connection.close()

print("All 12 validation checks passed successfully!")
