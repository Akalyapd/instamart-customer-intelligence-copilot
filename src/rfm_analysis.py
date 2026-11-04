from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned data
orders = pd.read_csv(BASE_DIR / "data" / "clean" / "orders.csv")
customers = pd.read_csv(BASE_DIR / "data" / "clean" / "customers.csv")

# Convert order timestamp to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

print("Orders shape:", orders.shape)
print("Customers shape:", customers.shape)
print("Order statuses:")
print(orders["status"].value_counts())

# Assignment snapshot date
snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# Keep completed orders only
completed_orders = orders[
    orders["status"] == "completed"
].copy()

# Calculate RFM metrics for each customer
rfm = completed_orders.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in whole days
rfm["recency"] = (
    snapshot_date - rfm["last_order_ts"]
).dt.days

# Keep the required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Display summary
print("\nRFM shape:", rfm.shape)
print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

# Select RFM features
rfm_features = rfm[["recency", "frequency", "monetary"]]

# Standardize the features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Evaluate K values from 2 to 8
kmeans_results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(rfm_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(rfm_scaled, labels)

    kmeans_results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette": silhouette
    })

# Display results
kmeans_results_df = pd.DataFrame(kmeans_results)

print("\nK-means evaluation:")
print(kmeans_results_df.to_string(index=False))

# Fit final K-means model with 4 clusters
final_kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(rfm_scaled)

# Segment sizes and average RFM values
segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print("\nSegment profiles:")
print(segment_profile)

# PCA with two components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_)

print("Total explained variance:")
print(pca.explained_variance_ratio_.sum())

# Assign business names to segments
segment_names = {
    0: "Occasional Customers",
    1: "Premium Customers",
    2: "Loyal Regulars",
    3: "At-Risk Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Identify Premium customers
rfm["is_premium"] = (
    rfm["segment_name"] == "Premium Customers"
)

# Save required columns
customer_segments = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

# Create outputs folder if needed
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

# Save CSV
customer_segments.to_csv(
    output_dir / "customer_segments.csv",
    index=False
)

print("\nCustomer segments saved successfully!")
print("Shape:", customer_segments.shape)
print("\nSegment name counts:")
print(customer_segments["segment_name"].value_counts())
print("\nPremium customers:", customer_segments["is_premium"].sum())