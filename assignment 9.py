#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')
import mlxtend
print(mlxtend.__version__)

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('wine.csv')

# Basic data exploration
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Histograms for distribution of features
df.hist(bins=15, figsize=(15, 10))
plt.show()

# Box plots for distribution of features
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# Correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Apply PCA
pca = PCA()
pca_df = pca.fit_transform(scaled_df)

# Scree plot
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# Select number of components
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f'Number of components explaining 95% of the variance: {n_components}')

# Transform the data
pca = PCA(n_components=n_components)
pca_df = pca.fit_transform(scaled_df)

# PCA DataFrame
pca_df = pd.DataFrame(pca_df, columns=[f'PC{i+1}' for i in range(n_components)])
print(pca_df.head())


# In[6]:


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_df)
print(f'K-Means Silhouette Score: {silhouette_score(scaled_df, kmeans_labels)}')
print(f'K-Means Davies-Bouldin Index: {davies_bouldin_score(scaled_df, kmeans_labels)}')

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(scaled_df)
print(f'Hierarchical Silhouette Score: {silhouette_score(scaled_df, hc_labels)}')
print(f'Hierarchical Davies-Bouldin Index: {davies_bouldin_score(scaled_df, hc_labels)}')

# DBSCAN Clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# DBSCAN Clustering with modified parameters
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Try adjusting these parameters
dbscan_labels = dbscan.fit_predict(scaled_df)

# Check the unique labels
unique_labels = np.unique(dbscan_labels)
print(f"Unique labels from DBSCAN: {unique_labels}")

# Calculate the silhouette score only if there is more than one cluster
if len(unique_labels) > 1:
    print(f'DBSCAN Silhouette Score: {silhouette_score(scaled_df, dbscan_labels)}')
    print(f'DBSCAN Davies-Bouldin Index: {davies_bouldin_score(scaled_df, dbscan_labels)}')
else:
    print("DBSCAN did not find multiple clusters; silhouette score is not applicable.")


# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('K-Means Clustering')

plt.subplot(1, 3, 2)
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=hc_labels, palette='viridis')
plt.title('Hierarchical Clustering')

plt.subplot(1, 3, 3)
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=dbscan_labels, palette='viridis')
plt.title('DBSCAN Clustering')

plt.show()


# In[7]:


# K-Means Clustering on PCA data
kmeans_pca = KMeans(n_clusters=3, random_state=42)
kmeans_pca_labels = kmeans_pca.fit_predict(pca_df)
print(f'PCA K-Means Silhouette Score: {silhouette_score(pca_df, kmeans_pca_labels)}')
print(f'PCA K-Means Davies-Bouldin Index: {davies_bouldin_score(pca_df, kmeans_pca_labels)}')

# Hierarchical Clustering on PCA data
hc_pca = AgglomerativeClustering(n_clusters=3)
hc_pca_labels = hc_pca.fit_predict(pca_df)
print(f'PCA Hierarchical Silhouette Score: {silhouette_score(pca_df, hc_pca_labels)}')
print(f'PCA Hierarchical Davies-Bouldin Index: {davies_bouldin_score(pca_df, hc_pca_labels)}')

# DBSCAN Clustering on PCA data
dbscan_pca = DBSCAN(eps=1.5, min_samples=5)
dbscan_pca_labels = dbscan_pca.fit_predict(pca_df)
print(f'PCA DBSCAN Silhouette Score: {silhouette_score(pca_df, dbscan_pca_labels)}')
print(f'PCA DBSCAN Davies-Bouldin Index: {davies_bouldin_score(pca_df, dbscan_pca_labels)}')

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.scatterplot(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], hue=kmeans_pca_labels, palette='viridis')
plt.title('PCA K-Means Clustering')

plt.subplot(1, 3, 2)
sns.scatterplot(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], hue=hc_pca_labels, palette='viridis')
plt.title('PCA Hierarchical Clustering')

plt.subplot(1, 3, 3)
sns.scatterplot(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], hue=dbscan_pca_labels, palette='viridis')
plt.title('PCA DBSCAN Clustering')

plt.show()


# In[11]:


# Compare Silhouette Scores
print(f'Original Data K-Means Silhouette Score: {silhouette_score(scaled_df, kmeans_labels)}')
print(f'PCA Data K-Means Silhouette Score: {silhouette_score(pca_df, kmeans_pca_labels)}')

print(f'Original Data Hierarchical Silhouette Score: {silhouette_score(scaled_df, hc_labels)}')
print(f'PCA Data Hierarchical Silhouette Score: {silhouette_score(pca_df, hc_pca_labels)}')


print(f'PCA Data DBSCAN Silhouette Score: {silhouette_score(pca_df, dbscan_pca_labels)}')

# Comparison Analysis
# You can write the analysis based on the scores and visualization.

