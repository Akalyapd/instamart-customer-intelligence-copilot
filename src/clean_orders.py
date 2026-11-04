import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")

print(orders.head())
print(orders.shape)
orders.info()
print(orders.isnull().sum())
print(orders.duplicated().sum())

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors = "coerce")
print(orders["order_ts"].dtype)

orders = orders.drop_duplicates()
print("Shape of removing duplicates :", orders.shape)
print("Remaining duplicates :", orders.duplicated().sum())

print("Unique order ids :", orders["order_id"].nunique())
print("Rows with repeated order ids :", orders.duplicated(subset = ["order_id"]).sum())

repeated_orders = orders[orders.duplicated(subset = ["order_id"], keep = False)]
print(repeated_orders.sort_values("order_id").head(10))

orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset = ["order_id"], keep = "last")
print("Shape after keeping latest orders :", orders.shape)
print("Unique order IDs :", orders["order_id"].nunique())
print(orders["status"].value_counts())

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors = "coerce")
print(orders["order_amount"].dtype)
print("Missing order amounts :", orders["order_amount"].isnull().sum())


products = pd.read_csv("data/raw/products.csv")
print(products.head())
print(products.columns)

orders = orders.merge(products[["product_id", "price_inr", "category"]], on = "product_id", how = "left")

missing_amounts = orders["order_amount"].isnull()

orders.loc[missing_amounts, "order_amount"] = (
    orders.loc[missing_amounts, "price_inr"] 
    * orders.loc[missing_amounts, "quantity"] 
    * (1 - orders.loc[missing_amounts, "discount_pct"] / 100)
).round(2)

print("Missing order amounts after filling : ", orders["order_amount"].isnull().sum())

orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")

completed = orders["status"] == "completed"

category_medians = orders.loc[completed].groupby("category")["delivery_minutes"].transform("median")

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"].fillna(category_medians)
)
print("Negative delivery times : ",(orders["delivery_minutes"] < 0).sum())
print("Missing delivery times : ", orders.loc[completed, "delivery_minutes"].isnull().sum())
print("Missing delivery times in cancelled orders : ", orders.loc[~completed, "delivery_minutes"].isnull().sum())

orders = orders[
    [
        "order_id",
        "customer_id",
        "product_id",
        "quantity",
        "order_ts",
        "status",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    ]
]
orders.to_csv("data/clean/orders.csv", index = False)

print("Cleaned orders saved successfully!")
print("Final Shape : ", orders.shape)
print("Completed revenue : ", orders.loc[orders["status"] == "completed", "order_amount"].sum())