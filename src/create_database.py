import pandas as pd
import sqlite3

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

connection = sqlite3.connect("data/instamart.db")
customers.to_sql("customers", connection, if_exists = "replace", index = False)
products.to_sql("products", connection, if_exists = "replace", index = False)
orders.to_sql("orders", connection, if_exists = "replace", index = False)

connection.close()

print("SQLite database created successfully!")
