import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

completed_orders = orders[orders["status"] == "completed"]

print("Customers :", customers.shape)
print("products :", products.shape)
print("Completed orders :", completed_orders.shape)

orders_customers = completed_orders.merge(customers, on = "customer_id", how = "left")

# Merge the result with products
sales_data = orders_customers.merge(products, on = "product_id", how = "left")

print("\n Merged dataset shape : ", sales_data.shape)
print("\nSelected columns:")
print(
    sales_data[
        ["order_id", "customer_id", "city",
         "product_id", "category", "order_amount"]
    ].head()
)

# Pandas cross-check: Revenue by city and category
revenue_pivot = pd.pivot_table(
    sales_data,
    values = "order_amount",
    index = "city",
    columns = "category",
    aggfunc = "sum",
    fill_value = 0,
    margins = True,
    margins_name = "Grand Total"
)
print("\n PANDAS : Revenue by city and category")
print(revenue_pivot)

print("\n Pandas Grand Total Revenue : ₹{:,.2f}".format(revenue_pivot.loc["Grand Total", "Grand Total"]))

# City-wise revenue statistics using groupby()
city_summary = sales_data.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_amount", "count")
).reset_index()

print("\nCITY-WISE REVENUE SUMMARY")
print(city_summary)

# Statistical analysis of completed order amounts

order_amount = completed_orders["order_amount"]

mean_amount = order_amount.mean()
median_amount = order_amount.median()
skewness = order_amount.skew()

print("\nORDER AMOUNT STATISTICS")
print("Mean: ₹{:,.2f}".format(mean_amount))
print("Median: ₹{:,.2f}".format(median_amount))
print("Skewness:", round(skewness, 3))

# Calculate quartiles and IQR

q1 = order_amount.quantile(0.25)
q2 = order_amount.quantile(0.50)
q3 = order_amount.quantile(0.75)

iqr = q3 - q1

lower_fence = q1 - (1.5 * iqr)
upper_fence = q3 + (1.5 * iqr)

# Identify outliers
outliers = order_amount[
    (order_amount < lower_fence) |
    (order_amount > upper_fence)
]

print("\nQUARTILE AND OUTLIER ANALYSIS")
print("Q1:", round(q1, 2))
print("Q2 (Median):", round(q2, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Lower fence: ₹{:,.2f}".format(lower_fence))
print("Upper fence: ₹{:,.2f}".format(upper_fence))
print("Number of outliers:", len(outliers))
print("Outliers below lower fence:", (order_amount < lower_fence).sum())
print("Outliers above upper fence:", (order_amount > upper_fence).sum())

# Correlation analysis

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_matrix = completed_orders[
    correlation_columns
].corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix)

# Find the strongest absolute correlation
correlation_pairs = correlation_matrix.unstack()

# Remove self-correlations and duplicate pairs
correlation_pairs = correlation_pairs[
    correlation_pairs.index.get_level_values(0) <
    correlation_pairs.index.get_level_values(1)
]

strongest_pair = correlation_pairs.abs().idxmax()

print("\nStrongest correlation pair:", strongest_pair)
print(
    "Correlation value:",
    round(correlation_matrix.loc[strongest_pair[0], strongest_pair[1]], 3)
)

from pathlib import Path

# Create outputs folder if it does not exist
Path("outputs").mkdir(parents=True, exist_ok=True)

# Save Pandas analysis results
revenue_pivot.to_csv("outputs/pandas_revenue_pivot.csv")
city_summary.to_csv("outputs/city_revenue_summary.csv", index=False)
correlation_matrix.to_csv("outputs/order_amount_correlations.csv")

print("\nPandas analysis outputs saved successfully!")