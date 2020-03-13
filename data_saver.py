#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pickle
#data generation functions
def doubler (v, e):
    # Takes a node and splits into two descendants with probability epsilon of switching value
    t = np.array([])
    for i in range(0, len(v)):
        t = np.append(t, [np.random.choice([v[i],-v[i]], p = [1-e, e]), np.random.choice([v[i],-v[i]], p = [1-e, e])]).astype(int)
    
    return t

def leaf_maker (v, e, B):
    # Takes a node and splits into B descendants with probability epsilon of switching value
    # v is parent node, e is epsilon and B is number of descendants
    t = np.array([])
    for i in range(0, len(v)):
        to_add = np.array([])
        for nodes in range(B):
            to_add = np.append(to_add, np.random.choice([v[i],-v[i]], p = [1-e, e]))
        t = np.append(t,to_add).astype(int)
    
    return t

def sample_generator(e, P, B, n):
    # Produces leaf nodes for n data points
    # e is epsilon, P is total number of output nodes, B is number of descendants, n is number of datapoints
    levels = int((math.log(P, B)))
    result = np.zeros((1,P))
    features = np.array([])
    for n in range(n):
        v = np.array([np.random.choice([1,-1], p = [0.5,0.5])])
        features = np.append(features, v)
        for level in range(0, levels):
            v = leaf_maker(v,e,B)
            # B can be set to change with level, would need to input B as a matrix etcetc
        result = np.vstack([result,v])
    
    return features, result[1:,:]  


# In[3]:


#Animal class that encapsulates data generated from tree (animals and their features)
class Animal(object):
    def __init__(self, animal_n, total_n, feature):
        self.input = np.zeros(total_n,dtype=np.float64)
        self.input[animal_n] = 1.0
        self.feature = feature


# In[4]:


#variables for data generation
epsilon = 0.2 # small probability of value flipping 
P = 16 #total leaf nodes in tree
B = 2 #number of descendants
total_data = 10 #number of data points

#variables for size of matrices and learning process
n_1 = 16 #number of inputs/ item/ animals
#n_2 = 5 #number of hidden layer nodes
n_3 = 10 #number of features/ output
lower = 0 #lower bound for range that generated initial values for weight matrices
upper = 0.01 #upper bound
#trial_n = 1000 # number of trials for learning process
lamda = 0.1 #learning rate of backpropagation


# In[5]:


#generate sample of data
feature, item = sample_generator(epsilon, P, B, total_data)


# In[6]:


#root nodes (badly named!)
feature


# In[7]:


#feature values across all 16 examples/items/animals (eg for the first feature, every animal has the same root value -1)
#for the second feature all the animals have its same value except for the last 2, where it flipped
item


# In[8]:


#creating a list of n_1 pairs of item/feature
animals = np.array([])
for i in range(0,n_1):
    animals = np.append(animals, Animal(i,n_1,item[:,i]))


# In[9]:


for i in range(0,n_1):
    print(animals[i].feature)


# In[ ]:





# In[ ]:





# In[10]:


np.savetxt('data.csv', item, delimiter=',')


# In[11]:


np.savetxt('feature.csv', feature, delimiter=',')

