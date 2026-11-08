
-- 1. Total number of customers
SELECT COUNT(*) AS total_customers
FROM customers;

-- 2. Total number of products
SELECT COUNT(*) AS total_products
FROM products;

-- 3. Number of orders by status
SELECT status, COUNT(*) AS total_orders
FROM orders
GROUP BY status;

-- 4. Total revenue from completed orders
SELECT ROUND(SUM(order_amount), 2) AS completed_revenue
FROM orders
WHERE status = 'completed';

-- 5. Top 10 customers by completed order revenue
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_spent
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 10;

-- 6. Average delivery time by product category
SELECT p.category, ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category;

-- 7. Number of orders by city
SELECT c.city, COUNT(*) AS total_orders
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.city
ORDER BY total_orders DESC;
