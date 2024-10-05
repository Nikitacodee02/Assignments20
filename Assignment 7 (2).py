#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[8]:


train_data = pd.read_csv('Titanic_train.csv')
test_data = pd.read_csv('Titanic_test.csv')


# In[9]:


print(train_data.info())
print(train_data.describe())


# In[10]:


train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data.drop(columns=['Cabin'], inplace=True)


# In[11]:


train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)


# In[12]:


plt.figure(figsize=(12, 8))
train_data[['Age', 'Fare']].hist(bins=20, color='skyblue', edgecolor='black', layout=(2, 2))
plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Survived', y='Fare', data=train_data)
plt.title("Fare Distribution by Survival")
plt.show()


# In[14]:


sns.pairplot(train_data[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived', palette="coolwarm")
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# In[16]:


X = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
y = train_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[18]:


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


# In[19]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)


# In[20]:


print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')


# In[21]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()




# In[22]:


# Coefficients of the logistic regression model
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

print(coefficients.sort_values(by='Coefficient', ascending=False))


# 1. What is the difference between precision and recall?
# 
# 
# 
# Precision and recall are two important metrics used to evaluate the performance of a classification model, especially in scenarios where class imbalance might be an issue.
# 
# Precision measures the accuracy of the positive predictions. It is the ratio of true positive predictions to the total number of positive predictions made by the model (i.e., the sum of true positives and false positives). In formula terms:
# 
# High precision means that when the model predicts a positive class, it is often correct.
# 
# Recall, on the other hand, measures the model's ability to find all the positive instances in the dataset. It is the ratio of true positive predictions to the total number of actual positive instances (i.e., the sum of true positives and false negatives). In formula terms:
# 
# High recall means that the model identifies most of the positive instances, but it might also include some false positives.
# 
# In summary, precision focuses on the quality of the positive predictions, while recall focuses on the quantity of the positive instances that are correctly identified.
# 
# 2.What is cross-validation, and why is it important in binary classification?
# 
# 
# Cross-validation is a technique used to assess how well a model generalizes to an independent dataset. It's important for evaluating model performance and avoiding issues like overfitting, where a model performs well on the training data but poorly on new, unseen data.
# 
# How it works: In cross-validation, the dataset is divided into several subsets or "folds." For example, in k-fold cross-validation, the dataset is split into k equally sized folds. The model is trained on k-1 of these folds and tested on the remaining fold. This process is repeated k times, with each fold being used as the test set exactly once. The final performance metric is the average of the performance measures from all k iterations.
# 
# Why it's important in binary classification: Cross-validation provides a more reliable estimate of a model's performance compared to a single train-test split. It helps ensure that the model's performance is not dependent on the particular way the data was split and gives a better sense of how the model will perform on new data. This is especially crucial in binary classification tasks where class imbalance or variability in the dataset might affect the model's performance. Cross-validation helps in obtaining a robust evaluation metric and tuning the model parameters effectively.
# 
# 
# 
# 
# 
# 
# 

# 
