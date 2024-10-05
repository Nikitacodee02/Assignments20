#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Zoo.csv')

df


# In[2]:


sns.countplot(df['type'])
plt.title('Distribution of Animal Types')
plt.show()

# Pairplot to see the relationship between features
sns.pairplot(df, hue='type')
plt.show()


# In[3]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[5]:


# Check for missing values
missing_values = df.isnull().sum()

# Check for duplicate values
duplicate_values = df.duplicated().sum()

# If missing values exist, handle them (you can drop or fill them)
df = df.dropna()  # or you can use df.fillna(method='ffill') or any other method

# Drop duplicates
df = df.drop_duplicates()



# In[6]:


# Detecting outliers using Z-score or IQR method
from scipy import stats
import numpy as np

# Z-score method
z = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(z < 3).all(axis=1)]


# In[7]:


from sklearn.model_selection import train_test_split

# Assuming the target variable is 'class_type'
X = df.drop('type', axis=1)
y = df['type']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder on each column with categorical data
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])




# In[10]:


#  re-split the data after encoding
X = df.drop('type', axis=1)
y = df['type']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune this value

# Train the model
knn.fit(X_train, y_train)


# In[11]:


# You can use GridSearchCV to find the best parameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Best parameters
best_params = grid.best_params_
print(best_params)



# In[12]:


# Using the best parameters
knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
knn.fit(X_train, y_train)


# In[13]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predictions
y_pred = knn.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


# In[14]:


print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(classification_report(y_test, y_pred))


# In[15]:


from matplotlib.colors import ListedColormap
import numpy as np

# Assuming only two features for visualization purposes
X_vis = X_train.iloc[:, :2].values
y_vis = y_train.values

h = .02  # step size in the mesh
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Train KNN on two features
knn_vis = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
knn_vis.fit(X_vis, y_vis)


# In[16]:


# Select only the first two features for visualization
X_vis = X_train.iloc[:, :2].values
y_vis = y_train.values

# Recalculate the min and max for the selected features
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# In[17]:


# Continue with the decision boundary visualization
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.title("3-Class classification (k = %i, metric = '%s')"
          % (best_params['n_neighbors'], best_params['metric']))
plt.show()


# Interview Questions Answers:
# 
# 1. What are the Key Hyperparameters in KNN?
# The key hyperparameters in K-Nearest Neighbors (KNN) are:
# 
# n_neighbors (K value):
# 
# What it is: This is the number of nearest neighbors the algorithm will consider when making a prediction.
# Why it’s important: Choosing the right K value is crucial. A small K (like 1) can make the model too sensitive to noise, leading to overfitting. A large K can make the model too generalized, potentially missing important patterns.
# metric (Distance Metric):
# 
# What it is: This determines how the distance between data points is calculated. Common choices include:
# Euclidean distance: Measures the straight-line distance between two points.
# Manhattan distance: Measures the distance between two points along the axes (like walking through a city grid).
# Minkowski distance: A generalization of both Euclidean and Manhattan distances, allowing you to adjust how distances are calculated by changing a parameter.
# Why it’s important: The choice of metric affects which neighbors are considered "close" and can impact the model’s performance.
# weights:
# 
# What it is: This determines whether all neighbors are weighted equally or if closer neighbors have more influence.
# Why it’s important: If you believe closer neighbors should have more influence, you can use distance weighting, which might improve accuracy in some cases.
# 
# 
# 2. What Distance Metrics Can Be Used in KNN?
# In KNN, the distance metric you choose determines how you measure the similarity between data points. Common distance metrics include:
# 
# Euclidean Distance:
# 
# What it is: The most common metric, it calculates the straight-line distance between two points.
# When to use it: It’s best when the features are continuous and on a similar scale.
# Manhattan Distance:
# 
# What it is: Also known as city block distance, it measures the distance between two points along the axes at right angles.
# When to use it: Useful when dealing with features that are independent or when the data has high dimensions.
# Minkowski Distance:
# 
# What it is: A generalization of both Euclidean and Manhattan distances, controlled by a parameter p.
# When to use it: When you want flexibility in choosing between Euclidean and Manhattan distances. For example, p=1 gives you Manhattan distance, and p=2 gives you Euclidean distance.
# Chebyshev Distance:
# 
# What it is: Measures the maximum absolute difference between coordinates of the points.
# When to use it: It’s like the distance a king would move in chess—useful in grid-like environments or when the largest difference matters the most.

# In[ ]:




