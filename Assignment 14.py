#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install scikit-learn==0.24.2


# In[2]:


pip install --upgrade imbalanced-learn


# In[3]:


pip install scikit-learn==0.24.2 imbalanced-learn==0.7.0


# In[4]:


# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the Glass dataset
glass_data = pd.read_csv('glass.csv', header=None)
glass_data



# In[5]:


import pandas as pd
import matplotlib.pyplot as plt



# In[6]:


glass_data = pd.read_csv('glass.csv', header=None)
print(glass_data.shape)  # This will print the number of rows and columns in the dataset


# In[7]:


glass_data.columns = [f'Feature_{i}' for i in range(glass_data.shape[1] - 1)] + ['Type']


# In[8]:


glass_data.columns = ['Feature']


# In[9]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

# Load the Glass dataset
glass_data = pd.read_csv('glass.csv', header=None)
glass_data.columns = ['Feature']

# Task 1: Exploratory Data Analysis (EDA)
print("Summary Statistics:")
print(glass_data.describe())
print("\nMissing Values:")
print(glass_data.isnull().sum())

# Task 2: Data Visualization
# Task 2: Data Visualization
print(glass_data.dtypes)  # Check the data type of the column

if glass_data['Feature'].dtype.kind in 'bifc':  # Check if the column is numerical
    plt.figure(figsize=(10, 6))
    glass_data['Feature'].hist()
    plt.show()
else:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Feature', data=glass_data)
    plt.show()




# In[29]:


# Task 3: Data Preprocessing


# Check if the dataframe contains numerical columns
if glass_data.select_dtypes(include=['number']).shape[1] > 0:
    # Select only numerical columns
    glass_data_num = glass_data.select_dtypes(include=['number'])

    # Task 3: Data Preprocessing
    scaler = StandardScaler()
    glass_data_scaled = scaler.fit_transform(glass_data_num)
else:
    print("The dataframe does not contain any numerical columns.")



# In[10]:


# Load the Glass dataset
glass_data = pd.read_csv('glass.csv', header=None)

# Check the shape and head of the dataframe
print(glass_data.shape)
print(glass_data.head())

# Check the data types of each column
print(glass_data.dtypes)

# Check for missing values in each column
print(glass_data.isnull().sum())

# Try converting columns to numerical data types
glass_data_num = glass_data.apply(pd.to_numeric, errors='coerce')

# Check if the conversion was successful
print(glass_data_num.dtypes)


# In[30]:


# Check if the dataframe contains numerical columns
if glass_data.select_dtypes(include=['number']).shape[1] > 0:
    # Select only numerical columns
    glass_data_num = glass_data.select_dtypes(include=['number'])

    # Task 3: Data Preprocessing
    scaler = StandardScaler()
    glass_data_scaled = scaler.fit_transform(glass_data_num)
else:
    glass_data_scaled = glass_data  # Assign the original dataframe if it doesn't contain numerical columns

X = glass_data_scaled  # Assuming the target variable is the same as the feature


# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the Glass dataset
glass_data = pd.read_csv('glass.csv', header=None)

# Identify categorical columns
categorical_cols = glass_data.select_dtypes(include=['object']).columns

# Perform one-hot encoding on categorical columns
glass_data = pd.get_dummies(glass_data, columns=categorical_cols)

# Split the data into features (X) and target (y)
X = glass_data.iloc[:, :-1]
y = glass_data.iloc[:, -1]

# Encode the target variable if it's categorical
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the Random Forest model
rf.fit(X_train, y_train)


# In[17]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
bagging_model = BaggingClassifier(estimator=dt, n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)

y_pred_bagging = bagging_model.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bagging))


# In[21]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
boosting_model = AdaBoostClassifier(dt, n_estimators=100, random_state=42)
boosting_model.fit(X_train, y_train)

y_pred_boosting = boosting_model.predict(X_test)
print("Boosting Accuracy:", accuracy_score(y_test, y_pred_boosting))


# In[25]:


from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Bagging
bagging_model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)

# Boosting
boosting_model = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=42)
boosting_model.fit(X_train, y_train)
y_pred_boosting = boosting_model.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nBagging Classification Report:")
print(classification_report(y_test, y_pred_bagging))
print("Bagging Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_bagging))

print("\nBoosting Classification Report:")
print(classification_report(y_test, y_pred_boosting))
print("Boosting Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_boosting))

print("\nModel Comparison:")
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bagging))
print("Boosting Accuracy:", accuracy_score(y_test, y_pred_boosting))


# In[27]:


# Bagging (Bootstrap Aggregating)
print("\nBagging (Bootstrap Aggregating):")
print("Bagging is an ensemble learning method that combines multiple instances of the same base model,")
print("each trained on a random subset of the training data. The final prediction is made by")
print("voting or averaging the predictions of the individual models.")

# Boosting
print("\nBoosting:")
print("Boosting is an ensemble learning method that combines multiple instances of the same base model,")
print("each trained on a modified version of the training data. The final prediction is made by")
print("voting or averaging the predictions of the individual models, with more weight given to")
print("models that perform well on difficult samples.")


# In[28]:


print("\nHandling Imbalance in the Data:")
print("Imbalanced data occurs when one class has a significantly larger number of instances than")
print("the other classes. This can lead to biased models that perform well on the majority class")
print("but poorly on the minority classes.")

print("Some common techniques for handling imbalance include:")
print("1. Oversampling the minority class")
print("2. Undersampling the majority class")
print("3. Using class weights to give more importance to the minority class")
print("4. Using metrics that are insensitive to class imbalance, such as F1-score or AUC-ROC")


# 1. Explain Bagging and Boosting methods. How is it different from each other.
# 
# Bagging (Bootstrap Aggregating)
# 
# Bagging is an ensemble learning method that involves creating multiple instances of a model, each trained on a random subset of the training data. The idea is to reduce overfitting by averaging the predictions of multiple models. Here's how it works:
# 
# Create multiple bootstrap samples from the training data by randomly sampling with replacement.
# Train a model on each bootstrap sample.
# Combine the predictions of each model to produce a final prediction.
# Bagging is useful for reducing overfitting and improving the stability of a model. It's particularly useful for models that are prone to overfitting, such as decision trees.
# Boosting
# 
# Boosting is another ensemble learning method that involves creating multiple models, each trained on a different subset of the data. However, unlike bagging, boosting involves training each model on the residuals of the previous model. Here's how it works:
# 
# Initialize a model with a random set of weights.
# Train the model on the data and calculate the residuals (errors).
# Create a new model and train it on the residuals of the previous model.
# Repeat steps 2-3 until a desired level of accuracy is reached.
# Combine the predictions of each model to produce a final prediction.
# Boosting is useful for improving the accuracy of a model by focusing on the most difficult examples. It's particularly useful for models that are prone to underfitting, such as linear models.
# Key differences between Bagging and Boosting
# 
# Sampling: Bagging involves random sampling with replacement, while boosting involves training each model on the residuals of the previous model.
# Model interaction: In bagging, each model is trained independently, while in boosting, each model is trained on the residuals of the previous model.
# Error reduction: Bagging reduces overfitting by averaging the predictions of multiple models, while boosting reduces error by focusing on the most difficult examples.
#     2. Explain how to handle imbalance in the data?
# 
# What is class imbalance?
# 
# Class imbalance occurs when one class in a classification problem has a significantly larger number of instances than the other classes. This can lead to biased models that favor the majority class.
# 
# Why is class imbalance a problem?
# 
# Class imbalance can lead to:
# 
# Biased models that favor the majority class
# Poor performance on the minority class
# Inaccurate evaluation metrics
# How to handle class imbalance?
# Here are some common techniques to handle class imbalance:
# 
# Oversampling the minority class: Create additional instances of the minority class by applying techniques such as random oversampling, SMOTE (Synthetic Minority Over-sampling Technique), or ADASYN (Adaptive Synthetic Sampling).
# Undersampling the majority class: Reduce the number of instances of the majority class by applying techniques such as random undersampling or Tomek links.
# Class weighting: Assign higher weights to the minority class during training to penalize the model for misclassifying minority class instances.
# Cost-sensitive learning: Assign different costs to misclassification errors for different classes.
# Anomaly detection: Treat the minority class as anomalies and use anomaly detection techniques such as One-Class SVM or Local Outlier Factor (LOF).
# ** Ensemble methods**: Use ensemble methods such as bagging or boosting to combine multiple models trained on different subsets of the data.
#     Here are some common techniques to handle class imbalance:
# 
# Oversampling the minority class: Create additional instances of the minority class by applying techniques such as random oversampling, SMOTE (Synthetic Minority Over-sampling Technique), or ADASYN (Adaptive Synthetic Sampling).
# Undersampling the majority class: Reduce the number of instances of the majority class by applying techniques such as random undersampling or Tomek links.
# Class weighting: Assign higher weights to the minority class during training to penalize the model for misclassifying minority class instances.
# Cost-sensitive learning: Assign different costs to misclassification errors for different classes.
# Anomaly detection: Treat the minority class as anomalies and use anomaly detection techniques such as One-Class SVM or Local Outlier Factor (LOF).
# ** Ensemble methods**: Use ensemble methods such as bagging or boosting to combine multiple models trained on different subsets of the data.

# In[ ]:




