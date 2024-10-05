#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('EastWestAirlines.csv')
df


# In[2]:


df.head()


# In[3]:


from sklearn.preprocessing import StandardScaler


print(df.isnull().sum())



# Standardize the features
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)



# In[4]:


# Convert back to a DataFrame
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
scaled_df.head()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.pairplot(scaled_df)
plt.show()


# In[6]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method to determine the optimal number of clusters
inertia = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_df, kmeans.labels_))


# In[7]:


# Plot the Elbow curve
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


# In[8]:


# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different K')
plt.show()


# In[19]:


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

methods = ['single', 'complete', 'average', 'ward']

for method in methods:
    linked = linkage(scaled_df, method=method)
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title(f'Dendrogram using {method} linkage')
    plt.show()


# In[10]:


# Clustering using 'ward' linkage and 3 clusters
cluster_labels = fcluster(linked, 3, criterion='maxclust')


# In[11]:


# Adding cluster labels to the dataframe
scaled_df['Cluster_Labels'] = cluster_labels


# In[21]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_df)

# Adding DBSCAN cluster labels to the dataframe
scaled_df['DBSCAN_Labels'] = dbscan_labels


# In[13]:


plt.figure(figsize=(10, 7))
plt.scatter(scaled_df.iloc[:, 0], scaled_df.iloc[:, 1], c=dbscan_labels, cmap='rainbow')
plt.title('DBSCAN Clustering')
plt.show()


# In[20]:


# Analyzing K-Means clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_df)

scaled_df['KMeans_Labels'] = kmeans_labels


# In[15]:


plt.figure(figsize=(10, 7))
plt.scatter(scaled_df.iloc[:, 0], scaled_df.iloc[:, 1], c=kmeans_labels, cmap='rainbow')
plt.title('K-Means Clustering')
plt.show()


# In[16]:


for cluster in range(3):
    print(f"Cluster {cluster}:")
    print(scaled_df[scaled_df['KMeans_Labels'] == cluster].describe())


# In[17]:


# Silhouette score for K-Means
silhouette_kmeans = silhouette_score(scaled_df, kmeans_labels)
print(f"Silhouette Score for K-Means: {silhouette_kmeans}")

# Silhouette score for DBSCAN
silhouette_dbscan = silhouette_score(scaled_df, dbscan_labels)
print(f"Silhouette Score for DBSCAN: {silhouette_dbscan}")


# # ### Interpretation

# The clustering models (K-Means, Hierarchical, DBSCAN) work by grouping similar data points into clusters.
# The effectiveness of these models depends on selecting appropriate parameters, which directly impacts the accuracy 
# and clarity of the identified patterns in the data. 
# Proper tuning ensures meaningful and well-separated clusters.
