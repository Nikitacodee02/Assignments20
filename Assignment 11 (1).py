#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd

df = pd.read_csv('amine.csv')
df


# In[24]:


print(df.head())


# In[11]:


import matplotlib.pyplot as plt

# Distribution of Ratings
plt.hist(df['rating'], bins=20, edgecolor='black')
plt.title('Distribution of Anime Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[12]:


# Correlation between rating and members
correlation = df['rating'].corr(df['members'])
print(f"Correlation between rating and members: {correlation}")


# In[14]:


df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

# Now filter anime with rating > 9 and more than 50 episodes
high_rating_long_anime = df[(df['rating'] > 9) & (df['episodes'] > 50)]

print("High Rating, Long Anime:")
print(high_rating_long_anime[['name', 'rating', 'episodes']])


# In[15]:


top_rated_anime = df.sort_values(by='rating', ascending=False).head(5)
print("Top 5 Rated Anime:")
print(top_rated_anime[['name', 'rating']])


# In[16]:


# Split genre by commas into separate rows for accurate groupby
df_genre_split = df.assign(genre=df['genre'].str.split(',')).explode('genre')
average_rating_by_genre = df_genre_split.groupby('genre')['rating'].mean().sort_values(ascending=False)
print("Average Rating by Genre:")
print(average_rating_by_genre)


# In[17]:


most_episodes_anime = df.sort_values(by='episodes', ascending=False).head(5)
print("Top 5 Anime with Most Episodes:")
print(most_episodes_anime[['name', 'episodes']])


# In[18]:


# Fill NaN values in the 'genre' column with an empty string
df['genre'] = df['genre'].fillna('')




# In[19]:


action_anime = df[df['genre'].str.contains('Action')]
print("Action Anime:")
print(action_anime[['name', 'rating', 'episodes']])


# In[20]:


# Filter anime with rating > 9 and more than 50 episodes
high_rating_long_anime = df[(df['rating'] > 9) & (df['episodes'] > 50)]
print("High Rating, Long Anime:")
print(high_rating_long_anime[['name', 'rating', 'episodes']])


# In[21]:


action_anime.to_excel('action_anime.xlsx', index=False)
print("Filtered action anime data saved to action_anime.xlsx")



# Intepretation
# By analyzing patterns and similarities in data, it can predict and recommend content (anime)
# that aligns with a user's past interests or is popular among similar users. 
# This enhances user experience by providing personalized suggestions.

# # ####Interview Questions

# 1. Can you explain the difference between user-based and item-based collaborative filtering?
# 
# Answer:
#     
# Collaborative filtering is a method used in recommendation systems to suggest items to users based on the preferences and 
# behaviors of other users.
# There are two main types of collaborative filtering: user-based and item-based.
# User-based collaborative filtering focuses on finding similarities between users. 
# For example, if two users have rated a bunch of items similarly,then it’s likely they have similar tastes.
# So, if one user likes an item that the other user hasn’t seen yet, we might recommend that item to
# the other user based on the similarity between their preferences.
# 
# Item-based collaborative filtering, on the other hand,focuses on the similarities between items. 
# It looks at how users have rated different items and finds items that are similar based on these ratings.
# For instance, if a user likes a certain movie, we look for other movies that are rated similarly 
# by users who liked the first movie. The idea is that if two items have been rated similarly by many users, 
# they are probably similar, and we can recommend them to users who liked one of the items.
# 
# 
# 

#  2. What is collaborative filtering, and how does it work?
# 
# Answer:
# 
# Collaborative filtering is a technique used in recommendation systems to predict what a user might like based on the preferences of other users. 
# It’s called "collaborative" because it relies on the collaboration of many users to provide recommendations to each other.
# 
# The basic idea behind collaborative filtering is that people who have agreed in the past will agree in the future. 
# So, if I liked certain books and another user liked the same books, 
# it's likely we have similar tastes, and I might like other books that this user has liked as well.
# There are two main approaches to collaborative filtering: user-based and item-based .
# 
# - User-based collaborative filtering- works by finding users who are similar to the active user and recommending items
# that those similar users have liked. So, if I’m similar to User A, I might get recommendations based on what User A liked.
# 
# - Item-based collaborative filtering- looks at the relationship between items.
# It identifies items that are similar to what a user has liked in the past and recommends those similar items. 
# For example, if I liked a particular movie, the system would recommend other movies that are similar based on
# what other users who liked that movie also liked.
# 
# Collaborative filtering works well in systems with lots of user data, 
# as it relies on many users' input to find patterns of behavior that can be used to make accurate recommendations.
