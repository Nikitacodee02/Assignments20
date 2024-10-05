#!/usr/bin/env python
# coding: utf-8

# In[4]:


number = int(input("Enter a number to check if it's prime: "))
if is_prime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")


# In[6]:


import random

def product_of_random_numbers():

    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    correct_answer = num1 * num2
    
    print(f"Multiply {num1} and {num2}.")
    user_answer = int(input("Enter the product: "))
    
    if user_answer == correct_answer:
        print("Correct! Well done.")
    else:
        print(f"Incorrect. The correct answer is {correct_answer}.")
        
product_of_random_numbers()


# In[7]:


def squares_of_even_numbers():
    
    for i in range(100, 201):
        if i % 2 == 0:
            print(f"Square of {i} is {i**2}")

def squares_of_odd_numbers():
    
    for i in range(100, 201):
        if i % 2 != 0:
            print(f"Square of {i} is {i**2}")


squares_of_even_numbers()


# In[8]:


def word_counter(input_text):
   
    words = input_text.split()
    word_count = {}
    
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    for word, count in word_count.items():
        print(f"'{word}': {count}")

input_text = "This is a sample text. This text will be used to demonstrate the word counter."
word_counter(input_text)


# In[9]:


def is_palindrome(s):
   
    s = ''.join(e for e in s if e.isalnum()).lower()  
    return s == s[::-1]


string = "A man, a plan, a canal, Panama"
if is_palindrome(string):
    print(f'"{string}" is a palindrome.')
else:
    print(f'"{string}" is not a palindrome.')


# In[ ]:




