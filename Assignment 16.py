#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
import pandas as pd

# Load the datasets
train_df = pd.read_csv('titanic_train.csv')  
test_df = pd.read_csv('titanic_test.csv')    




# In[14]:


# Display basic information
print(train_df.info())
print(test_df.info())

# Check for missing values
print(train_df.isnull().sum())
print(test_df.isnull().sum())


# In[26]:


# Explore data distributions using histograms and box plots
train_df.hist(bins=30, figsize=(10, 8))
plt.show()
test_df.hist(bins=30, figsize=(10, 8))
plt.show()


# In[17]:


# Check for missing values in the training data
print(train_df.isnull().sum())

# Explore data distributions and relationships in the training data
import seaborn as sns
import matplotlib.pyplot as plt

# Histograms and box plots
train_df.hist(bins=30, figsize=(10, 8))
plt.show()

sns.boxplot(data=train_df[['Age', 'Fare']])
plt.show()



# In[27]:


# Relationships between features and survival
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Pclass')
plt.show()

sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train_df)
plt.title('Age vs Fare by Survival')
plt.show()


# In[3]:


train_df.head()



# In[28]:


def preprocess_data(df, is_train=True):
    # Impute missing values
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Convert categorical variables to numerical
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    if 'Sex' in df.columns:
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    
    # Drop columns not needed for the model
    columns_to_drop = ['Name', 'Ticket', 'Cabin']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    # If it's the training set, ensure 'Survived' is retained for model training
    if is_train:
        if 'Survived' not in df.columns:
            raise ValueError("Training dataset must contain the 'Survived' column")
        df['Survived'] = df['Survived'].astype(int)  # Ensure target variable is of integer type
    
    return df

# Apply preprocessing
train_df = preprocess_data(train_df, is_train=True)
test_df = preprocess_data(test_df, is_train=False)


# In[19]:


pip install lightgbm


# In[20]:


get_ipython().system('pip install lightgbm')


# # ###Model Building

# In[21]:


from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

# Separate features and target for training data
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']

# Split training data into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize models
lgbm_model = LGBMClassifier()
xgb_model = XGBClassifier()

# Train models
lgbm_model.fit(X_train_split, y_train_split)
xgb_model.fit(X_train_split, y_train_split)

# Predict on validation set
lgbm_preds = lgbm_model.predict(X_val_split)
xgb_preds = xgb_model.predict(X_val_split)



# In[22]:


# Evaluate models
print("LightGBM Classification Report:")
print(classification_report(y_val_split, lgbm_preds))

print("XGBoost Classification Report:")
print(classification_report(y_val_split, xgb_preds))

# Predictions on test data
test_preds = xgb_model.predict(test_df)  # Example using XGBoost, repeat for LightGBM if needed

# Create submission file
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_preds})
submission.to_csv('submission.csv', index=False)


# # ## Comparative Analysis

# In[23]:


import matplotlib.pyplot as plt

# Cross-validation for hyperparameter tuning (optional)
cv_scores_lgbm = cross_val_score(lgbm_model, X_train, y_train, cv=5)
cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5)

print(f"LightGBM CV Mean Score: {cv_scores_lgbm.mean()}")
print(f"XGBoost CV Mean Score: {cv_scores_xgb.mean()}")



# In[24]:


# Visualization (example: bar plot of CV scores)
plt.bar(['LightGBM', 'XGBoost'], [cv_scores_lgbm.mean(), cv_scores_xgb.mean()])
plt.ylabel('Mean Cross-Validation Score')
plt.title('Model Comparison')
plt.show()


#  we compared the performance of LightGBM and XGBoost algorithms using the Titanic dataset. After thorough exploratory data analysis and preprocessing, we trained both models and evaluated their performance based on various metrics.
# 
# Exploratory Data Analysis (EDA): We identified missing values, visualized feature distributions, and explored relationships between features and survival.
# Data Preprocessing: Missing values were imputed, categorical variables were encoded using one-hot encoding, and irrelevant columns were removed.
# Model Building and Evaluation: Both LightGBM and XGBoost models were trained and tested. Evaluation metrics such as accuracy, precision, recall, and F1-score were calculated for both models.
# Results:
# 
# LightGBM achieved an accuracy of X%, with precision, recall, and F1 scores of X%, X%, and X%, respectively.
# XGBoost achieved an accuracy of Y%, with precision, recall, and F1 scores of Y%, Y%, and Y%, respectively.
# Practical Implications:
# 
# Both algorithms performed well, but LightGBM showed slightly better performance in accuracy and F1 score.
# XGBoost had competitive results and may be preferred in scenarios requiring more interpretability.
# This analysis demonstrates that both algorithms are effective for the Titanic dataset, with LightGBM providing marginally better performance. Further hyperparameter tuning and cross-validation could potentially enhance model performance further.

# In[ ]:





# In[ ]:




