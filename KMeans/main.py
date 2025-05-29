import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sample_submission.csv", sep=",")

le = LabelEncoder()
df['cuisine_encoded'] = le.fit_transform(df['cuisine'])

X = df[['id', 'cuisine_encoded']]

inertia = []
sil_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, kmeans.labels_))

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(k_range, sil_scores, marker='o', color='orange')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

optimal_k = sil_scores.index(max(sil_scores)) + 2
print(f"Optimal number of clusters: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

sil_score = silhouette_score(X, df['cluster'])
calinski_score = calinski_harabasz_score(X, df['cluster'])
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Calinski-Harabasz Score: {calinski_score:.4f}")

sns.scatterplot(data=df, x='id', y='cuisine_encoded', hue='cluster', palette='tab10')
plt.title("Clusters")
plt.show()
