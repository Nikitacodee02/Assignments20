#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install ppscore


# In[9]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import IsolationForest
import ppscore as pps  # Correct import
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('adult_with_headers.csv')
data


# In[10]:


# Basic data exploration
print(data.info())  
print(data.describe())  


# In[ ]:


# Checking for missing values
missing_values = data.isnull().sum()
print(f"Missing values in each column:\n{missing_values}")


# In[ ]:


# Handle missing values
data.dropna(inplace=True)


# In[ ]:


# Scaling numerical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns


# In[ ]:


# Standard Scaling
scaler_standard = StandardScaler()
data_standard_scaled = data.copy()
data_standard_scaled[numerical_features] = scaler_standard.fit_transform(data_standard_scaled[numerical_features])


# In[ ]:


# Min-Max Scaling
scaler_min_max = MinMaxScaler()
data_min_max_scaled = data.copy()
data_min_max_scaled[numerical_features] = scaler_min_max.fit_transform(data_min_max_scaled[numerical_features])


# In[ ]:


print("Standard Scaling is useful when data has outliers, as it centers the data around mean.")
print("Min-Max Scaling is useful when we need to normalize the data within a specific range, usually between 0 and 1.")


# In[11]:


# Identifying categorical features
categorical_features = data.select_dtypes(include=['object']).columns


# In[ ]:


# One-Hot Encoding for categorical variables with less than 5 categories
data_one_hot_encoded = data.copy()
for feature in categorical_features:
    if data[feature].nunique() < 5:
        data_one_hot_encoded = pd.get_dummies(data_one_hot_encoded, columns=[feature], drop_first=True)


# In[ ]:


# Label Encoding for categorical variables with more than 5 categories
data_label_encoded = data.copy()
label_encoder = LabelEncoder()
for feature in categorical_features:
    if data[feature].nunique() >= 5:
        data_label_encoded[feature] = label_encoder.fit_transform(data_label_encoded[feature])


# In[ ]:


# Pros and cons
print("One-Hot Encoding is good for nominal data and does not assume any order in the categories, but it increases dimensionality.")
print("Label Encoding is good for ordinal data where there is an inherent order, but it can introduce a misleading relationship between categories for nominal data.")


# In[12]:


# Creating new features
data['age_per_hours_worked'] = data['age'] / (data['hours_per_week'] + 1)
data['income_per_education_years'] = data['education_num'] * (data['income'] == '>50K').astype(int)


# In[ ]:


# Transformation for skewed numerical feature
data['capital_gain_log'] = np.log1p(data['capital_gain'])


# In[ ]:


# Justification
print("Created 'age_per_hours_worked' to capture productivity as a function of age and work hours.")
print("Created 'income_per_education_years' to explore the impact of education on income.")
print("Applied log transformation on 'capital_gain' to reduce skewness and normalize the distribution.")


# In[13]:


import sys
print(sys.executable)
get_ipython().system('pip install --upgrade pip')



# In[14]:


import ppscore as pps


# In[18]:


# Using Isolation Forest to identify outliers
iso_forest = IsolationForest(contamination=0.05)
outliers = iso_forest.fit_predict(data[numerical_features])


# In[ ]:


# Removing outliers
data_no_outliers = data[outliers == 1]

# Discussing outliers
print("Outliers can skew the data and affect model performance by introducing noise. Removing them helps in stabilizing the model's predictions.")


# In[ ]:


# Applying PPS analysis
pps_matrix = pps.matrix(data_no_outliers)
print("PPS Matrix:")
print(pps_matrix)


# In[ ]:


# Plotting PPS matrix
sns.heatmap(pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore'), annot=True)
plt.title('Predictive Power Score Matrix')
plt.show()


# In[17]:


# Correlation Matrix
corr_matrix = data_no_outliers.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Discussing PPS and Correlation Matrix
print("PPS is useful for finding non-linear relationships between features, unlike Pearson correlation which only captures linear relationships.")


# In[ ]:





# In[ ]:




