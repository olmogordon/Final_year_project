#!/usr/bin/env python
# coding: utf-8

# In[149]:


import string
from collections import Counter
import numpy as np


# In[143]:


def word_counter(filename, N):
    #load text
    file = open(filename, 'rt')
    text = file.read()
    
    # split into words by white space
    words = text.split()
    
    # clean up the words that appear in the book
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words] #remove punctuation
    words_clean = [word.lower() for word in stripped] #convert to lower case
    
    #get the N most frequent words. Returns a list with ('word', frequency) entries
    text_words = Counter(words_clean)
    most_occur = text_words.most_common(N)

    #transform into a list with just the N most frequnt words
    words_list = []
    for i in range(N):
        if (most_occur[i][0] != ''):
            words_list.append(most_occur[i][0])
    words_list.append('OTHER')
    
    return words_list
    


# In[144]:


alice_words = word_counter('alice_in_wonderland.txt', 1000)


# In[145]:


metamorphosis_words = word_counter('metamorphosis_clean.txt', 1000)


# In[ ]:



for i in len(alice_words):
    


# In[ ]:


def one_hot(words):
    words_matrix = np.zeros((len(words),len(words)), dtype=np.float64)
    for i in range(len(words)):
        for j in 
    
    

