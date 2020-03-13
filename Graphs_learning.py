#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import random
import matplotlib.pyplot as plt


# In[2]:


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

def sample_generator(e ,P, B, n):
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
    
    return features, result[1:,:]  #forward function for passing on inputs and generating predictions

#forward function for passing on inputs and generating predictions
def forward(x, w_1, w_2):
    h = np.matmul(w_1, x)
    return np.matmul(w_2, h)

#squared error calculation function
def error(w_1, w_2, animals):
    error = 0
    for animal in animals:
        y_hat = forward(animal.input, w_1, w_2)
        y = animal.feature
        error = error + np.sum((y - y_hat)**2)
    return error

#weight matrix gradient calculation function
def gradient(i, w_1, w_2, n_1, n_2, n_3, animals):
    #item/ particular animal performing the learning on
    #separate input and output
    x = animals[i].input
    y = animals[i].feature

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


#variables for size of matrices and learning process
n_1 = 16 #number of inputs/ item/ animals
n_2 = 5 #number of hidden layer nodes
n_3 = 10 #number of features/ output
lower = -0.001 #lower bound for range that generated initial values for weight matrices
upper = 0.001 #upper bound
trial_n = 2000 # number of trials for learning process
lamda = 0.1 #learning rate of backpropagation


# In[4]:


#Animal class that encapsulates data generated from tree (animals and their features)
class Animal(object):
    def __init__(self, animal_n, total_n, feature):
        self.input = np.zeros(total_n,dtype=np.float64)
        self.input[animal_n] = 1.0
        self.feature = feature
    


# In[5]:


data = np.loadtxt('data.csv', delimiter=',')
feature = np.loadtxt('feature.csv', delimiter=',')


# In[6]:


#creating a list of n_1 pairs of item/feature
animals = np.array([])
for i in range(0,n_1):
    animals = np.append(animals, Animal(i,n_1,data[:,i]))


# In[7]:


for i in range(0,n_1):
    print(animals[i].feature)


# In[8]:


#----------------------------------------------------------------------------------------------------------------------#


# In[9]:


def plotter(n_1, n_2, n_3, trial_n, animals, lamda, lower, upper, plot = True):
    all_errors = np.zeros((10, trial_n))
    for i in range(10):
        w_1 = np.random.uniform(lower,upper,[n_2, n_1])
        w_2 = np.random.uniform(lower,upper,[n_3, n_2])
        errors=[]
        for trial_c in range(0,trial_n):
            dw_1, dw_2 = gradient(random.randint(0,n_1-1), w_1, w_2, n_1, n_2, n_3, animals)
            w_1 = w_1 + lamda * dw_1
            w_2 = w_2 + lamda * dw_2
            errors.append(error(w_1, w_2, animals))
        all_errors[i,:] = errors

    avg = np.mean(all_errors, axis=0)
    if plot == True:
        plt.figure(figsize=(5,5))
        plt.xlabel("Trial", fontsize=13)
        plt.ylabel("Squared error", fontsize=13)
        plt.xlim([0, 1000])
        plt.ylim([0, 170])
        for i in range(10):
            plt.plot(all_errors[i,:])
        plt.plot(avg, "k", linewidth = 3, label="Average of 10 iterations")
        plt.legend(loc="upper right", prop={'size': 10})
        plt.show() 
    return avg


# In[10]:


avg_1 = plotter(n_1, 1, n_3, trial_n, animals, lamda, lower, upper)
avg_2 = plotter(n_1, 2, n_3, trial_n, animals, lamda, lower, upper)
avg_5 = plotter(n_1, 5, n_3, trial_n, animals, lamda, lower, upper)
avg_10 = plotter(n_1, 10, n_3, trial_n, animals, lamda, lower, upper)
#average of final error it tends to (taking the last 1000 trials)
#final_error = np.mean(avg_1[1000:1999])


# In[11]:


#plotting graph showing difference between three averages for 1, 2 and 5 nodes in the hidden layers
plt.figure(figsize=(5,5))
plt.xlim([0, 1000])
plt.ylim([0, 200])
plt.xlabel("Trial", fontsize=13)
plt.ylabel("Squared error", fontsize=13)
plt.plot(avg_1, linewidth = 3, label = "1 node")
plt.plot(avg_2, linewidth = 3, label = "2 nodes")
plt.plot(avg_5, linewidth = 3, label = "5 nodes")
plt.legend(loc="upper right", prop={'size': 20})
plt.show()


# In[ ]:




