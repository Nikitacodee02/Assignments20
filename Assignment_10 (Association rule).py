#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_excel('Online retail.xlsx', header=None)

# Display the first few rows 
df.head()


# In[8]:


# Split the items in each transaction by commas
transactions = df[0].apply(lambda x: x.split(','))

# Create a list of unique items 
all_items = sorted(set(item.strip() for sublist in transactions for item in sublist))

# Create a binary matrix (rows represent transactions, columns represent products)
basket = pd.DataFrame(0, index=transactions.index, columns=all_items)



# In[14]:


# Mark products that were purchased in each transaction
for i, transaction in enumerate(transactions):
    for item in transaction:
        item = item.strip()  # Clean the item name
        if item in basket.columns:  # Ensure the item exists in the columns
            basket.at[i, item] = 1

# Display the first few rows of the basket to verify the transformation
basket.head()


# In[9]:


from mlxtend.frequent_patterns import apriori, association_rules

# Apply the Apriori algorithm with a minimum support threshold
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Display the first few frequent itemsets
print(frequent_itemsets.head())


# In[10]:


# Generate the association rules using lift as the metric
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Sort rules by lift to get the most significant ones
rules = rules.sort_values(by='lift', ascending=False)

# Display the top 5 association rules
print(rules.head())


# In[11]:


# Filter rules based on confidence and lift thresholds
filtered_rules = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 1.2)]

# Display the filtered rules
print(filtered_rules.head())


# In[16]:


# Check the column names and index
print("Column names:", data.columns)
print("Index:", data.index)
# Display more information about the dataset
print(data.info())
print(data.head())


# In[12]:


# Example interpretation of the first rule
for i, rule in filtered_rules.iterrows():
    print(f"Rule: If a customer buys {rule['antecedents']} they are likely to also buy {rule['consequents']}")
    print(f" - Support: {rule['support']}")
    print(f" - Confidence: {rule['confidence']}")
    print(f" - Lift: {rule['lift']}")
    print("\n")


# In[13]:


import matplotlib.pyplot as plt

# Visualize rules: Scatter plot of support, confidence, and lift
plt.scatter(filtered_rules['support'], filtered_rules['confidence'], alpha=0.5, marker="o")
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()



# In[ ]:




