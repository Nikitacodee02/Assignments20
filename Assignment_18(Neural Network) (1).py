#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[21]:


data = pd.read_csv('Alphabets_data.csv')
data


# In[4]:


print(data.head())
print(data.info())
print(data.describe())


# In[5]:


print(data.isnull().sum())


# In[6]:


# Separate features and target
X = data.drop('letter', axis=1)
y = data['letter']


# In[7]:


# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[8]:


# Encode the target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# In[9]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


# In[10]:


# Build a basic ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer + Hidden layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(26, activation='softmax'))  # Output layer (26 classes for 26 letters)


# In[11]:


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))



# In[12]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[13]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


# In[14]:


# Function to create the model
def create_model(hidden_layers=1, neurons=32, activation='relu', learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(26, activation='softmax'))  # Output layer

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# In[15]:


# Define the grid of hyperparameters to search
param_grid = {
    'hidden_layers': [1, 2, 3],
    'neurons': [32, 64, 128],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'learning_rate': [0.001, 0.01, 0.1]
}


# In[16]:


best_params = None
best_score = 0
best_model = None


# In[17]:


# Manual Grid Search
for hidden_layers in param_grid['hidden_layers']:
    for neurons in param_grid['neurons']:
        for activation in param_grid['activation']:
            for learning_rate in param_grid['learning_rate']:
                model = create_model(hidden_layers=hidden_layers, neurons=neurons, 
                                     activation=activation, learning_rate=learning_rate)
                model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
                y_pred = model.predict(X_test).argmax(axis=-1)
                score = accuracy_score(y_test, y_pred)
                
                print(f"Params: layers={hidden_layers}, neurons={neurons}, "
                      f"activation={activation}, lr={learning_rate} => Accuracy: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = (hidden_layers, neurons, activation, learning_rate)
                    best_model = model


# In[18]:


# best model
print(f"\nBest model parameters: {best_params} with accuracy: {best_score:.4f}")


# In[19]:


#Evaluate the best model
y_pred_best = best_model.predict(X_test).argmax(axis=-1)
print("Classification Report:\n", classification_report(y_test, y_pred_best, target_names=encoder.classes_))


# In[ ]:





# In[ ]:




