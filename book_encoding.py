#!/usr/bin/env python
# coding: utf-8

# In[173]:


import string
from collections import Counter
import numpy as np


# In[217]:


def word_counter(filename, N):
    #load text
    file = open(filename, 'rt')
    text = file.read()
    
    # split into words by white space
    words = text.split()
    
    # clean up the words that appear in the book
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words] #remove punctuation
    #stripped1 = [w.strip('') for w in stripped]
    words_clean = [w.lower() for w in stripped] #convert to lower case
    
    #get the N most frequent words. Returns a list with ('word', frequency) entries
    text_words = Counter(words_clean)
    most_occur = text_words.most_common(N)

    #transform into a list with just the N most frequnt words
    words_list = []
    word_to_index = {}
    for i in range(N-1):
        words_list.append(most_occur[i][0])
        word_to_index[words_list[i]]=i
    
    words_list.append('OTHER')
    word_to_index['OTHER'] = N-1
    
    return words_list, word_to_index
    


# In[226]:


def word_to_onehot(word,word_to_index):
    n = len(word_to_index)
    one_hot=np.zeros(n)
    if(word in word_to_index):
        one_hot[word_to_index[word]]=1
    else:
        one_hot[n-1]=1

    return one_hot


# In[237]:


metamorphosis_words, met_index_map = word_counter('metamorphosis_clean.txt', 1000)


# In[238]:


one_hot_matrix = []
for word in metamorphosis_words:
    one_hot_matrix.append(word_to_onehot(word, met_index_map))


# In[239]:


alice_words, alice_index_map = word_counter('alice_in_wonderland.txt', 1000)


# In[240]:


one_hot_alice = []
for word in alice_words:
    one_hot_alice.append(word_to_onehot(word, alice_index_map))


# In[ ]:




