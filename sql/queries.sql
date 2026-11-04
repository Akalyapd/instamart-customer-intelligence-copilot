-- Q1: Customers with total spending above ₹8000
SELECT 
customer_id,
SUM(order_amount) AS total_spending
FROM orders
WHERE status = 'completed'
AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spending DESC;

-- Q2: Revenue by city and product category
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

-- Q3: Products priced above average catalogue price
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

-- Q4: Customers with no completed orders
SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Customer spending using CTE
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


-- Q6: Rank customers by spending within each city
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


-- Q7: Daily completed revenue with running total
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