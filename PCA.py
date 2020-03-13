#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA


# In[2]:


#forward function for passing on inputs and generating predictions
def forward(x, w_1, w_2):
    h = np.matmul(w_1, x)
    return np.matmul(w_2, h)

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


# In[21]:


#variables for size of matrices and learning process
n_1 = 16 #number of inputs/ item/ animals
n_2 = 10 #number of hidden layer nodes
n_3 = 10 #number of features/ output
lower = -0.001 #lower bound for range that generated initial values for weight matrices
upper = 0.001 #upper bound
trial_n = 1000 # number of trials for learning process
lamda = 0.1 #learning rate of backpropagation


# In[22]:


#Animal class that encapsulates data generated from tree (animals and their features)
class Animal(object):
    def __init__(self, animal_n, total_n, feature):
        self.input = np.zeros(total_n,dtype=np.float64)
        self.input[animal_n] = 1.0
        self.feature = feature


# In[23]:


data = np.loadtxt('data.csv', delimiter=',')
feature = np.loadtxt('feature.csv', delimiter=',')


# In[24]:


#creating a list of n_1 pairs of item/feature
animals = np.array([])
for i in range(0,n_1):
    animals = np.append(animals, Animal(i,n_1,data[:,i]))


# In[25]:


#for each row i, each entry j indicates whether animal i presents feature j or not
for i in range(0,n_1):
    print(animals[i].feature)


# In[26]:


#one hot vector inputs representing each item/ animal put together in one big 16x16 matrix
X = np.zeros((16,16))
for j in range(0,n_1):
    for i in range(0,n_1):
        X[i,j] = animals[i].input[j]


# In[27]:


#getting back the leaf values of the hierarchical tree generated from one root feature (value taken from flipping a coin)
#this is the same as the "item" array from the data_saver notebook (no real need for this.) 10x16 matrix
z = np.zeros((10,16))
for j in range(0,n_3):
    for i in range(0,n_1):
        z[j,i] = animals[i].feature[j]


# In[28]:


#----------------------------------------------------------------------------------------------------------------------#


# In[29]:


def h(n_1, n_2, n_3, trial_n, animals, lamda, lower, upper, w_1, w_2, freq):
    #h matrix with final values of middle layer activations
    h_final = np.zeros((n_2, n_1))
    
    #big matrix with one-hot vectors in rows
    X = np.zeros((16,16)) 
    for j in range(0,n_1):
        for i in range(0,n_1):
            X[i,j] = animals[i].input[j]   
    #matrix to store all hidden layer activations along the way. there should be "trial_n/freq" 5x16 matrices stored in H
    H = np.zeros((trial_n//freq,n_2,n_1)) 
    #counter for updating entry of H matrix
    t = 0
    #learning of network over trial_n trials
    for trial_c in range(0,trial_n):
        dw_1, dw_2 = gradient(random.randint(0,n_1-1), w_1, w_2, n_1, n_2, n_3, animals)
        w_1 = w_1 + lamda * dw_1
        w_2 = w_2 + lamda * dw_2
        if(trial_c % freq == 0):
            H[t,:,:] = np.matmul(w_1,X)
            t = t+1
    #final activation h values after learning, should be 5 16 dimensional vectors.
    h_final = np.matmul(w_1,X)    
    return h_final, H

def pca(data, pc_count = None):
    pca = PCA(n_components = 2)
    return pca.fit_transform(data), pca.components_, pca.explained_variance_


# In[30]:


#plot function for single PCA pair points
def plot_single_pca(a,b):
    colours = ["black", "silver", "indianred", "orangered", "darkorange", "gold", "olive", "lightgreen", "darkgreen", "aqua", "teal", "dodgerblue", "royalblue", "mediumpurple", "indigo", "fuchsia"]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("PC1", fontsize=13)
    plt.ylabel("PC2", fontsize=13)
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    ax1.scatter(a, b, color = colours, s = 15)
    plt.show()


# In[39]:


#function to plot progression of 2D PCA values 
import matplotlib.colors as mcolors
def pca_plotter(h_all, h_final_mean, C, trial_n, freq):
    colours = ["black", "silver", "indianred", "orangered", "darkorange", "gold", "olive", "lightgreen", "darkgreen", "aqua", "teal", "dodgerblue", "royalblue", "mediumpurple", "indigo", "fuchsia"]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("PC1", fontsize=13)
    plt.ylabel("PC2", fontsize=13)
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    for i in range(0,(trial_n//freq)-1):
        h_temp = h_all[i,:,:] - h_final_mean.reshape(10,1)*np.ones((1,16))
        a = np.matmul(np.transpose(h_temp), C[0,:])
        b = np.matmul(np.transpose(h_temp), C[1,:])
        ax1.scatter(a, b, color = colours, s = 15)
    plt.show()


# In[40]:


#small random initial weights
w_1 = np.random.uniform(lower,upper,[n_2, n_1])
w_2 = np.random.uniform(lower,upper,[n_3, n_2])


# In[41]:


#final h activation values (5x16) and an array (h_all) with "intermediate" activation functions taken at regular intervals trial_n/10. 
h_final, h_all = h(n_1, n_2, n_3, trial_n, animals, lamda, lower, upper, w_1, w_2, 1)


# In[42]:


#mean along columns (averaging over samples) to whiten the data: 1x5
h_final_mean = np.mean(h_final,axis=1)
#to whiten data, we need to transform our mean values to a 5x16 matrix that we can then subtract from h_final
h_final = h_final - (h_final_mean.reshape(10,1)*np.ones((1,16)))
#h_final


# In[43]:


#perform pca transform to reduce over colums, from 5 dimensions to 2. end up with 16 2-dimensional vectors (16x2)
pca1, C, var = pca(np.transpose(h_final)) 
#print(pca1)
#print(C)
#print(var)

#these should be 1x16 vectors corresponding to the columns of pca (and they are!)
a_final = np.matmul(np.transpose(h_final), C[0,:])
b_final = np.matmul(np.transpose(h_final), C[1,:])


# In[44]:


pca_plotter(h_all, h_final_mean, C, trial_n, 1)


# In[45]:


h_initial = np.matmul(w_1, X)
h_initial = h_initial - h_final_mean.reshape(10,1)*np.ones((1,16))
a_initial = np.matmul(np.transpose(h_initial), C[0,:])
b_initial = np.matmul(np.transpose(h_initial), C[1,:])
plot_single_pca(a_initial, b_initial)


# In[46]:


#entire process
h_final, h_all = h(n_1, n_2, n_3, 100, animals, lamda, lower, upper, w_1, w_2, 1)
h_final_mean = np.mean(h_final,axis=1)
h_final = h_final - (h_final_mean.reshape(10,1)*np.ones((1,16)))
pca1, C, var = pca(np.transpose(h_final)) 
#plot_single_pca(np.matmul(np.transpose(h_final), C[0,:]),np.matmul(np.transpose(h_final), C[1,:]))
pca_plotter(h_all, h_final_mean, C, 100, 1)


# In[48]:


#entire process
h_final, h_all = h(n_1, n_2, n_3, 10000, animals, lamda, lower, upper, w_1, w_2, 10)
h_final_mean = np.mean(h_final,axis=1)
h_final = h_final - (h_final_mean.reshape(10,1)*np.ones((1,16)))
pca1, C, var = pca(np.transpose(h_final)) 
#plot_single_pca(np.matmul(np.transpose(h_final), C[0,:]),np.matmul(np.transpose(h_final), C[1,:]))
pca_plotter(h_all, h_final_mean, C, 10000, 10)


# In[ ]:




