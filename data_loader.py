#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import import_ipynb
import Animal #not working...?


# In[2]:


data = np.loadtxt('data.csv', delimiter=',')


# In[3]:


feature = np.loadtxt('feature.csv', delimiter=',')


# In[5]:


feature


# In[14]:


#Animal class that encapsulates data generated from tree (animals and their features)
class Animal(object):
    def __init__(self, animal_n, total_n, feature):
        self.input = np.zeros(total_n,dtype=np.float64)
        self.input[animal_n] = 1.0
        self.feature = feature


# In[15]:


#creating a list of n_1 pairs of item/feature
n_1 = 16
animals = np.array([])
for i in range(0,n_1):
    animals = np.append(animals, Animal(i,n_1,data[:,i]))


# In[16]:


animals


# In[17]:


for i in range(0,n_1):
    print(animals[i].feature)


# In[18]:


for i in range(0,n_1):
    print(animals[i].input)

