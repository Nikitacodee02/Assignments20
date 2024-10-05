#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
df = pd.read_csv('mushroom.csv')

# Display basic information and the first few rows
print(df.info())
print(df.head())


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms
df.hist(figsize=(12, 10))
plt.show()

# Box plots (for numerical features)
sns.boxplot(data=df)
plt.show()

# Density plots
df.plot(kind='density', subplots=True, layout=(5,5), figsize=(15,15))
plt.show()


# In[3]:


# Compute the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[4]:


from sklearn.preprocessing import LabelEncoder

# Encode categorical features
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

print(df.head())


# In[5]:


from sklearn.model_selection import train_test_split

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[6]:


# Pair plots
sns.pairplot(df, hue='class')
plt.show()

# Scatter plots (example)
plt.scatter(df['cap_shape'], df['cap_color'], c=df['class'])
plt.xlabel('Cap Shape')
plt.ylabel('Cap Color')
plt.show()


# In[7]:


sns.countplot(x='class', data=df)
plt.show()


# In[8]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10]
}




# In[ ]:


# Perform grid search
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')


# In[ ]:


# Compare SVM with different kernels
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Kernel: {kernel}, Accuracy: {accuracy}')


# In[ ]:





# In[ ]:




