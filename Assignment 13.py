#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv('heart_disease.csv')
print(data.head())



# In[2]:


# Check for missing values
print(data.isnull().sum())

# Summary statistics
print(data.describe())

# Visualizing the data
import seaborn as sns
import matplotlib.pyplot as plt

# Histograms of all numeric columns
data.hist(figsize=(10, 10))
plt.show()

# Box plots to check for outliers
data.boxplot(figsize=(10, 10))
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[3]:


# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal', 'exang', 'fbs']
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# Check the encoded data
print(data.head())


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Features and target variable
X = data.drop('num', axis=1)
y = data['num']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[5]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

# Using the best parameters
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)

print("Accuracy with best parameters:", accuracy_score(y_test, y_pred_best))
print("Classification Report with best parameters:\n", classification_report(y_test, y_pred_best))


# In[9]:


from sklearn.tree import plot_tree


# Identify unique classes in the target variable
unique_classes = list(map(str, y.unique()))

# Visualizing the decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=unique_classes)
plt.show()



# Feature importance
importance = best_clf.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
print(feature_importance.sort_values(by='importance', ascending=False))


# 1. What are some common hyperparameters of decision tree models, and how do they affect the model's performance?
# 
# Some common hyperparameters of Decision Tree models include:
# 
# Max Depth (max_depth): This controls the maximum depth of the tree. If the tree is too deep, it might lead to overfitting, where the model performs well on the training data but poorly on unseen data. Setting this parameter too low might result in underfitting, where the model is too simple to capture the underlying patterns in the data.
# 
# Min Samples Split (min_samples_split): This parameter specifies the minimum number of samples required to split an internal node. Increasing this value can reduce the complexity of the model and help in preventing overfitting.
# 
# Min Samples Leaf (min_samples_leaf): This defines the minimum number of samples that must be present in a leaf node. Setting this to a higher value can create smoother decision boundaries and reduce overfitting.
# 
# Criterion: This hyperparameter defines the function used to measure the quality of a split. Common options are "gini" for the Gini impurity and "entropy" for information gain. The choice of criterion can affect the structure of the tree and its performance.
# 
# Max Features (max_features): This controls the number of features to consider when looking for the best split. Limiting this can introduce randomness into the model, which may improve generalization.
# 
# 2. What is the difference between Label encoding and One-hot encoding?
# 
# 
# Label encoding and One-hot encoding are two techniques used to convert categorical variables into numerical form so that they can be used in machine learning models.
# 
# Label Encoding: In Label encoding, each category is assigned a unique integer value. For example, if a column has categories "Red," "Green," and "Blue," they might be encoded as 0, 1, and 2, respectively. While this approach is simple, it can introduce unintended ordinal relationships between categories (e.g., 2 > 1 > 0), which might not be appropriate for all datasets.
# 
# One-hot Encoding: One-hot encoding creates a new binary column for each category, with a 1 indicating the presence of the category and 0 otherwise. For instance, "Red," "Green," and "Blue" would be represented as [1, 0, 0], [0, 1, 0], and [0, 0, 1]. This approach avoids introducing ordinal relationships but increases the dimensionality of the dataset, which can be a drawback if there are many unique categories

# In[ ]:




