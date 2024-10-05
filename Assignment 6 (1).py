#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('ToyotaCorolla - MLR.csv')
data


# In[3]:


print(data.isnull().sum())
data_encoded = pd.get_dummies(data, columns=['Fuel_Type'], drop_first=True)


# In[4]:


print(data_encoded.describe())


# In[5]:


plt.figure(figsize=(8, 5))
sns.histplot(data_encoded['Price'], kde=True, color='blue')
plt.title('Distribution of Car Prices')
plt.xlabel('Price (EURO)')
plt.ylabel('Frequency')
plt.show()


# In[6]:


sns.pairplot(data_encoded)
plt.show()


# In[7]:


plt.figure(figsize=(10,8))
sns.heatmap(data_encoded.corr(),annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# In[8]:


X = data_encoded.drop('Price', axis=1)
y = data_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


model1 = LinearRegression()
model1.fit(X_train, y_train)


# In[10]:


model2 = LinearRegression()
model2.fit(X_train[['Age_08_04', 'KM']], y_train)


# In[11]:


model3 = LinearRegression()
model3.fit(X_train[['HP', 'Weight', 'Automatic']], y_train)


# In[12]:


# Evaluate Model 1
y_pred1 = model1.predict(X_test)
print("Model 1 - All features:")
print("R-squared:", r2_score(y_test, y_pred1))
print("MSE:", mean_squared_error(y_test, y_pred1))
print("MAE:", mean_absolute_error(y_test, y_pred1))


# In[13]:


# Evaluate Model 2
y_pred2 = model2.predict(X_test[['Age_08_04', 'KM']])
print("\nModel 2 - Subset of features ('Age_08_04', 'KM'):")
print("R-squared:", r2_score(y_test, y_pred2))
print("MSE:", mean_squared_error(y_test, y_pred2))
print("MAE:", mean_absolute_error(y_test, y_pred2))


# In[14]:


# Evaluate Model 3
y_pred3 = model3.predict(X_test[['HP', 'Weight', 'Automatic']])
print("\nModel 3 - Subset of features ('HP', 'Weight', 'Automatic'):")
print("R-squared:", r2_score(y_test, y_pred3))
print("MSE:", mean_squared_error(y_test, y_pred3))
print("MAE:", mean_absolute_error(y_test, y_pred3))


# In[15]:


#Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("\nLasso Regression:")
print("R-squared:", r2_score(y_test, y_pred_lasso))
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("MAE:", mean_absolute_error(y_test, y_pred_lasso))


# In[16]:


# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("\nRidge Regression:")
print("R-squared:", r2_score(y_test, y_pred_ridge))
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("MAE:", mean_absolute_error(y_test, y_pred_ridge))


# ### interpretation

# The multiple linear regression models showed how different features, like age, mileage, and engine specifications,
# affect the car's price.
# Higher age and mileage typically reduced the price, while more horsepower and better engine features increased it. 
# Lasso and Ridge regression helped improve model stability by minimizing the effect of less important variables.

# 1. What is Normalization & Standardization and how is it helpful?
# 
# Normalization and Standardization are preprocessing techniques used to scale and transform features in a dataset to improve the performance and convergence speed of machine learning algorithms.
# 
# Normalization: This technique rescales the feature values to a fixed range, usually [0, 1].  
# Benefits:
# 
# Ensures that all features contribute equally to the distance metrics in algorithms like K-Nearest Neighbors and clustering.
# Helps algorithms that are sensitive to the scale of data, such as gradient descent-based methods.
# Standardization: This technique rescales the feature values to have a mean of 0 and a standard deviation of 
# 
# Benefits:
# 
# Ensures that features have similar scales, which is important for algorithms that assume normally distributed data or use distance metrics, such as Principal Component Analysis (PCA) and linear regression.
# Can improve the performance and convergence of algorithms that rely on gradient-based optimization.
# 
# 
# 
# 
# 
# 
# 
# 2. What techniques can be used to address multicollinearity in multiple linear regression?
# 
# Multicollinearity occurs when two or more predictor variables in a multiple linear regression model are highly correlated, leading to unstable estimates of coefficients and inflated standard errors.
# 
# Techniques to address multicollinearity include:
# 
# Remove Highly Correlated Predictors: Identify and remove one of the highly correlated predictors. This can be done using correlation matrices or variance inflation factors (VIF).
# 
# Principal Component Analysis (PCA): Transform the predictors into a set of orthogonal components (principal components) and use these components in the regression model. PCA reduces the dimensionality and mitigates multicollinearity.
# 
# Regularization Techniques: Apply regularization methods such as Ridge Regression or Lasso Regression. These techniques add a penalty term to the regression model that helps to control the size of the coefficients and reduces multicollinearity.
# 
# Ridge Regression: Adds a penalty proportional to the square of the magnitude of the coefficients.
# Lasso Regression: Adds a penalty proportional to the absolute value of the coefficients, which can also perform feature selection by shrinking some coefficients to zero.
# Combining Variables: Combine highly correlated variables into a single predictor. For example, you might create an index or aggregate measure that captures the essence of the correlated predictors.
# 
# Increase Sample Size: Sometimes, increasing the sample size can help reduce the impact of multicollinearity by providing more information for estimation.

# In[ ]:




