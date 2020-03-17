#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
from collections import Counter
import numpy as np
import math
import random
import matplotlib.pyplot as plt


# In[2]:


#squared error calculation function
def error(x, y, w_1, w_2):
    y_hat = np.matmul(w_2, np.matmul(w_1, x))
    return np.sum((y - y_hat)**2)

#weight matrix gradient calculation function
def gradient(x, y, w_1, w_2, n_1, n_2, n_3):

    #linear function of the deep neural network
    h = np.matmul(w_1, x)
    y_hat = np.matmul(w_2, h)

    x_matrix = np.reshape(x, (x.shape[0],1))
    h_matrix = np.reshape(h, (h.shape[0],1))
    y_difference = np.reshape((y - y_hat), (y.shape[0], 1))
    
    delta_w_1 = np.matmul(np.transpose(w_2) ,np.matmul(y_difference, np.transpose(x_matrix)))
    delta_w_2 = np.matmul(y_difference, np.transpose(h_matrix))
    
    return delta_w_1, delta_w_2


# In[3]:


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
    word_clean = [w.lower() for w in stripped] #convert to lower case
    words_clean = [ w for w in word_clean if w!='']
    
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


# In[4]:


def word_to_onehot(word,word_to_index):
    n = len(word_to_index)
    one_hot=np.zeros(n)
    if(word in word_to_index):
        one_hot[word_to_index[word]]=1
    else:
        one_hot[n-1]=1

    return one_hot


# In[5]:


metamorphosis_words, met_index_map = word_counter('metamorphosis_clean.txt', 1000)


# In[6]:


one_hot_met = []
for word in metamorphosis_words:
    one_hot_met.append(word_to_onehot(word, met_index_map))


# In[11]:


n_1 = 1000
n_2 = 10
n_3 = 1000
lower = -0.001
upper = 0.001
lamda = 0.01


# In[12]:


w_1 = np.random.uniform(lower,upper,[n_2, n_1])
w_2 = np.random.uniform(lower,upper,[n_3, n_2])


# In[ ]:


#clean text to be processed keeping full stops
file = open('metamorphosis_clean.txt', 'rt')
text = file.read()
text = text.split()
text_lower = [w.lower() for w in text] 
text_clean = [w.replace(',', '') for w in text_lower]
text_final = [w for w in text_clean if w!='']

errors = []

#go through text and take pairs of consecutive pairs in the book that are part of the most frequent 1000
#training the network with these pairs of words. 
for i, w in enumerate(text_final):
    if ((w[-1] != '.') and (w in metamorphosis_words) and (text_final[i+1] in metamorphosis_words)):
        x = word_to_onehot(w, met_index_map)
        y = word_to_onehot(text_final[i+1], met_index_map)
        #print(w, text_final[i+1])
        dw_1, dw_2 = gradient(x, y, w_1, w_2, n_1, n_2, n_3)
        w_1 = w_1 + lamda * dw_1
        w_2 = w_2 + lamda * dw_2
        print(error(x, y, w_1, w_2))
        errors.append(error(x, y, w_1, w_2))


# In[ ]:


alice_words, alice_index_map = word_counter('alice_in_wonderland.txt', 1000)


# In[ ]:


one_hot_alice = []
for word in alice_words:
    one_hot_alice.append(word_to_onehot(word, alice_index_map))

