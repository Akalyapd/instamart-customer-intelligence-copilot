import pandas as pd
import sqlite3

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

assert len(customers) == 1200

assert customers["age"].isna().sum() == 0
assert customers["age"].between(18, 60).all()

assert customers["gender"].value_counts().to_dict() == {
    "female" : 610,
    "male" : 560,
    "unknown" : 30
}
assert len(products) == 40

assert len(orders) == 12309

assert orders["order_id"].nunique() == 12309
assert orders["order_amount"].isna().sum() == 0

completed = orders[orders["status"] == "completed"] 
assert len(completed) == 11110
assert completed["delivery_minutes"].isna().sum() == 0

cancelled = orders[orders["status"] == "cancelled"]
assert len(cancelled) == 1199
assert cancelled["delivery_minutes"].isna().sum() == 1199

revenue = round(completed["order_amount"].sum(), 2)
assert revenue == 2998668.80

connection = sqlite3.connect("data/instamart.db")

tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
table_names = {table[0] for table in tables}
assert {"customers", "products", "orders"}.issubset(table_names)

assert connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 1200
assert connection.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 40
assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309

connection.close()

print("ALL CHECK POINTS PASSED")