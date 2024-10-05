#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk



# In[9]:


nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('blogs.csv')
df
# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())


# In[2]:


# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back to string
    return ' '.join(tokens)

# Apply preprocessing
df['Data'] = df['Data'].apply(preprocess_text)

# Check the first few rows after preprocessing
print(df.head())


# In[3]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Data'], df['Labels'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[10]:


# Initialize the Naive Bayes model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = nb_model.predict(X_test_tfidf)


# In[4]:


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[5]:


import nltk

# Download the vader_lexicon
nltk.download('vader_lexicon')


# In[6]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the vader_lexicon
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['Sentiment'] = df['Data'].apply(get_sentiment)

# Analyze sentiment distribution
print(df['Sentiment'].value_counts())

# Plot sentiment distribution
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()


#  Final Thoughts
# Evaluation: Discuss the performance of your Naive Bayes model, noting any challenges or limitations. Reflect on the sentiment analysis results and how they might inform content strategies or insights.
# 
# Report: Include all your findings, code explanations, and visualizations in a comprehensive report.

# In[ ]:




