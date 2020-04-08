#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import math
import random
import matplotlib.pyplot as plt
import string
from collections import Counter
from scipy import linalg
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import pickle


# In[8]:


# PCA functions
# PCA transform function
def pca(data, pc_count = None):
    pca = PCA(n_components = 3)
    return pca.fit_transform(data), pca.components_, pca.explained_variance_

#plot function for single PCA pair points
def plot_single_pca(a,b):
    number_of_colors = 1000
    colours = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("PC1", fontsize=13)
    plt.ylabel("PC2", fontsize=13)
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    ax1.scatter(a, b, color = colours, s = 15)
    plt.show()
    
#function to plot progression of 2D PCA values 
def pca_plotter(comp1, comp2, n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, points):
    
    number_of_colors = points
    colours = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("PC1", fontsize=13)
    plt.ylabel("PC2", fontsize=13)
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    for i in range(0,trial_n*(len(text_final)//freq)-1):
        h_temp = H[i,:,:] - h_final_mean.reshape(n_2,1)*np.ones((1,n_1))
        a = np.matmul(np.transpose(h_temp), C[comp1,:])[:points]
        b = np.matmul(np.transpose(h_temp), C[comp2,:])[:points]
        ax1.scatter(a, b, color = colours, s = 15)
    plt.show()


# In[17]:


#clean text keeping full stops
text_final = txt_full_stops('metamorphosis_clean.txt')

#count most frequent words and output a list of them together with an indexed dictionary
metamorphosis_words, met_index_map = word_counter('metamorphosis_clean.txt', 1000)

# change most frequent words to one-hot vectors one by one in order
one_hot_met = []
for word in metamorphosis_words:
    one_hot_met.append(word_to_onehot(word, met_index_map))
    
# convert to a np.array to make future operations easier (matmul, loop through it, etc)
# this is input matrix X of all the inputs to calculate hidden layer activation matrix at each stage
X = np.asarray(one_hot_met)


# In[9]:


n_1 = 1000
n_2 = 500
n_3 = 1000
lower = -0.001
upper = 0.001
lamda = 0.1
freq = 300
trial_n = 1


# In[12]:


H = pickle.load(open("H_1trial_500h_300freq.pickle", "rb"))


# In[ ]:


#X should be a (hidden layer size) x (number of words in X) matrix
#you should average over the second axis giving a (hidden later size) vector, 
#and then use the ones to get back to the original size and then take that way.


# In[13]:


#mean along columns (averaging over samples) to whiten the data: 1xn_2
h_final_mean = np.mean(H[-1],axis=1)

#to whiten data, we need to transform our mean values to a n_2x1000 matrix that we can then subtract from h_final
h_final = H[-1] - (h_final_mean.reshape(n_2,1)*np.ones((1,n_1)))


# In[ ]:





# In[14]:


#perform pca transform to reduce over colums, from 5 dimensions to 2. end up with 16 2-dimensional vectors (16x2)
pca, C, var = pca(np.transpose(h_final))

#print(pca)
#print(C)
#print(var)

#these should be 1x16 vectors corresponding to the columns of pca (and they are!)
a_final = np.matmul(np.transpose(h_final), C[2,:])
b_final = np.matmul(np.transpose(h_final), C[1,:])


# In[15]:


#plot_single_pca(a_final, b_final)
#plot_single_pca(a_final, np.random.rand(1000))


# In[24]:


pca_plotter(0,1,n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, 1000)


# In[25]:


pca_plotter(0,2,n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, 1000)


# In[26]:


pca_plotter(1,2,n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, 1000)


# In[ ]:





# In[ ]:


n_1 = 1000
n_2 = 500
n_3 = 1000
lower = -0.001
upper = 0.001
lamda = 0.1
freq = 1000
trial_n = 20


# In[ ]:





# In[ ]:





# In[ ]:




