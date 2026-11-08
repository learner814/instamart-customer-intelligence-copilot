
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load RFM data
rfm = pd.read_csv("outputs/rfm_analysis.csv")

features = rfm[["recency", "frequency", "monetary"]]

# Standardize RFM data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Test K values from 2 to 8
print("K-MEANS RESULTS")

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    labels = model.fit_predict(scaled_data)

    print(
        f"K={k}, Inertia={model.inertia_:.1f}, "
        f"Silhouette={silhouette_score(scaled_data, labels):.3f}"
    )

# Final model with K=4
final_model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_model.fit_predict(scaled_data)

# Create segment profile
profile = rfm.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
)

print("\nSEGMENT PROFILE")
print(profile)

# Assign segment names
premium_id = profile["avg_monetary"].idxmax()
loyal_id = profile["avg_frequency"].idxmax()
risk_id = profile["avg_recency"].idxmax()

segment_names = {}

segment_names[premium_id] = "Premium"
segment_names[loyal_id] = "Loyal"
segment_names[risk_id] = "At Risk"

for cluster_id in profile.index:
    if cluster_id not in segment_names:
        segment_names[cluster_id] = "Regular"

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Premium flag
rfm["is_premium"] = (
    rfm["segment_name"] == "Premium"
).astype(int)

# PCA analysis
pca = PCA(n_components=2)
pca.fit(scaled_data)

print("\nPCA EXPLAINED VARIANCE")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

# Save required columns
rfm = rfm[
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

rfm.to_csv("outputs/customer_segments.csv", index=False)

print("\nCustomer segmentation and PCA completed!")
print("Saved outputs/customer_segments.csv")

