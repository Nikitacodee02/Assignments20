#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



# In[2]:


df = pd.read_csv('cardiotocographic.csv')

df


# In[3]:


print(df.head())


# In[4]:


# Check for missing values
print(df.isnull().sum())

# Handle missing values (for simplicity, let's fill them with the median of each column)
df.fillna(df.median(), inplace=True)


# In[5]:


# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    return df

# Example for the 'LB' column
df = remove_outliers_iqr(df, 'LB')


# In[ ]:





# In[6]:


# Statistical summary
print(df.describe())


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms
df.hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.show()



# In[8]:


# Boxplots
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()


# In[9]:


# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# If you see a cell with a value of 0.44 between MLTV  and Width, it indicates a strong positive correlation. This means that as the baseline fetal heart rate increases, the percentage of time with abnormal short-term variability also tends to increase.
# 
# If there’s a cell with a value of -0.35 between ALTV  and Width, it indicates a strong negative correlation. This means that as uterine contractions increase, the late decelerations tend to decrease.
# 
# 

# In[10]:


# Pair plots
sns.pairplot(df)
plt.show()


# In[11]:


# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)


# Positive Correlations: Variables like LB and ASTV, UC and DL show strong positive correlations, indicating that they tend to increase together.
# Negative Correlations: Variables like LB and DL, LB and DP show negative correlations, indicating that as one increases, the other tends to decrease.
# No Significant Correlation: Variables like FM and most other variables show very low correlation coefficients, indicating no significant relationship.
# 
# 
# Interpretation
#  
# Strong Positive Correlation: LB and ASTV (0.70) suggest that higher baseline fetal heart rates are associated with more abnormal short-term variability.
# Strong Negative Correlation: LB and DL (-0.50) suggest that higher baseline fetal heart rates are associated with fewer late decelerations.
# Strong Positive Correlation: UC and DL (0.60) suggest that more frequent uterine contractions are associated with more late decelerations, potentially indicating fetal distress.

# In[ ]:





# In[ ]:




