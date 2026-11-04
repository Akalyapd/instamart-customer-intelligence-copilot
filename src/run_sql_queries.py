import sqlite3
import pandas as pd

connection = sqlite3.connect("data/instamart.db")

q1 = """
SELECT
customer_id,
SUM(order_amount) AS total_spending
FROM orders
WHERE status = 'completed'
AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spending DESC;
"""
result = pd.read_sql_query(q1, connection)

print(result)
print("Number of customers: ", len(result))


q2 = """
SELECT
    c.city,
    p.category,
    SUM(o.order_amount) AS total_revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;
"""

result2 = pd.read_sql_query(q2, connection)

print("\nQUERY 2: Revenue by city and category")
print(result2)
print("Number of rows:", len(result2))
print("Total revenue: ₹{:,.2f}".format(result2["total_revenue"].sum()))

q3 = """
SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;
"""

result3 = pd.read_sql_query(q3, connection)

print("\nQUERY 3: Products above average price")
print(result3)
print("Number of products:", len(result3))

q4 = """
SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;
"""

result4 = pd.read_sql_query(q4, connection)

print("\nQUERY 4: Customers without completed orders")
print(result4)
print("Number of customers:", len(result4))

q5 = """
WITH customer_spending AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spending
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
)
SELECT
    customer_id,
    total_spending
FROM customer_spending
WHERE total_spending > (
    SELECT AVG(total_spending)
    FROM customer_spending
)
ORDER BY total_spending DESC;
"""

result5 = pd.read_sql_query(q5, connection)

print("\nQUERY 5: Customers spending above average")
print(result5)
print("Number of customers:", len(result5))
average_spending = pd.read_sql_query(
    """
    SELECT AVG(total_spending)
    FROM (
        SELECT customer_id, SUM(order_amount) AS total_spending
        FROM orders
        WHERE status = 'completed'
        GROUP BY customer_id
    )
    """,
    connection
).iloc[0, 0]

print("Average spending across all customers: ₹{:,.2f}".format(average_spending))


# QUERY 6: Rank customers by spending within each city

result6 = pd.read_sql_query(
    """
    WITH customer_spending AS (
        SELECT
            c.city,
            o.customer_id,
            SUM(o.order_amount) AS total_spending
        FROM orders o
        JOIN customers c
            ON o.customer_id = c.customer_id
        WHERE o.status = 'completed'
        GROUP BY c.city, o.customer_id
    )
    SELECT
        city,
        customer_id,
        total_spending,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC
        ) AS spending_rank,
        DENSE_RANK() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC
        ) AS dense_spending_rank,
        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC
        ) AS row_num
    FROM customer_spending
    ORDER BY city, spending_rank, customer_id;
    """,
    connection
)

print("\nQUERY 6: Customer spending ranks within each city")
print(result6.groupby("city").head(1))

print(
    "Rows where RANK differs from ROW_NUMBER:",
    (result6["spending_rank"] != result6["row_num"]).sum()
)


# QUERY 7: Daily revenue with running total

q7 = """
WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_date,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM daily_revenue
ORDER BY order_date;
"""

result7 = pd.read_sql_query(q7, connection)

print("\nQUERY 7: Daily revenue with running total")
print(result7.head())
print("\nLast 5 days:")
print(result7.tail())

print("\nNumber of rows:", len(result7))
print("Final running total: ₹{:,.2f}".format(
    result7["running_total"].iloc[-1]
))

connection.close()