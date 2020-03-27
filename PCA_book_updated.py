#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import random
import matplotlib.pyplot as plt
import string
from collections import Counter
from scipy import linalg
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


# In[2]:


# squared error calculation function
def error(x, y, w_1, w_2):
    y_hat = np.matmul(w_2, np.matmul(w_1, x))
    return np.sum((y - y_hat)**2)

# weight matrix gradient calculation function
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

# clean text to be processed keeping full stops
def txt_full_stops(filename):
    file = open(filename, 'rt')
    text = file.read()
    text = text.split()
    text_lower = [w.lower() for w in text] 
    text_clean = [w.replace(',', '') for w in text_lower]
    text_final = [w for w in text_clean if w!='']
    
    return text_final

# function that returns N most frequent words from a given txt file and it returns a library of these words indexed in order.
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

# encoding function from words to one-hot vectors taking the (word, index) library generated by word_counter()
def word_to_onehot(word, word_to_index):
    n = len(word_to_index)
    one_hot=np.zeros(n)
    if(word in word_to_index):
        one_hot[word_to_index[word]]=1
    else:
        one_hot[n-1]=1

    return one_hot


# In[3]:


# PCA functions
# PCA transform function
def pca(data, pc_count = None):
    pca = PCA(n_components = 2)
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


# In[4]:


# function that does the learning and stores hidden layer activation values to perform PCA afterwards
def H(X, n_1, n_2, n_3, lamda, lower, upper, w_1, w_2, trial_n, text_final, index_map, freq_words, freq):
    #h matrix with final values of middle layer activations
    h_final = np.zeros((n_2, n_1))
    
    #matrix to store all hidden layer activations along the way
    H = np.zeros((trial_n * (len(text_final)//freq),n_2,n_1))
    
    #counter for updating entry of H matrix
    t = 0
    
    #learning of network over trial_n trials
    for trial in range(trial_n):
        x=word_to_onehot(text_final[0], index_map)
        for i in range(1,len(text_final)):
            if (text_final[i] in freq_words):
                y=word_to_onehot(text_final[i], index_map)
                dw_1, dw_2 = gradient(x, y, w_1, w_2, n_1, n_2, n_3)
                w_1 = w_1 + lamda * dw_1
                w_2 = w_2 + lamda * dw_2
                if (i % freq == 0):
                    H[t,:,:] = np.matmul(w_1,X)
                    t += 1
            else:
                i+=1
                if i<len(text_final):
                    while not(text_final[i] in freq_words):
                        i+=1
                    x=word_to_onehot(text_final[i], index_map)
                

    #final activation h values after learning, should be "n_2" "n_1" dimensional vectors.
    h_final = np.matmul(w_1,X)    
    
    return h_final, H


# In[5]:


#function to plot progression of 2D PCA values 
def pca_plotter(n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, step, points):
    
    number_of_colors = points
    colours = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("PC1", fontsize=13)
    plt.ylabel("PC2", fontsize=13)
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    for i in range(0,trial_n*(len(text_final)//freq)-1, step):
        h_temp = H[i,:,:] - h_final_mean.reshape(n_2,1)*np.ones((1,n_1))
        a = np.matmul(np.transpose(h_temp), C[0,:])[:points]
        b = np.matmul(np.transpose(h_temp), C[1,:])[:points]
        ax1.scatter(a, b, color = colours, s = 15)
    plt.show()


# In[10]:


n_1 = 1000
n_2 = 300
n_3 = 1000
lower = -0.001
upper = 0.001
lamda = 0.1
freq = 100
trial_n = 10


# In[11]:


w_1 = np.random.uniform(lower,upper,[n_2, n_1])
w_2 = np.random.uniform(lower,upper,[n_3, n_2])


# In[12]:


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


# In[ ]:


h_final, H = H(X, n_1, n_2, n_3, lamda, lower, upper, w_1, w_2, trial_n, text_final, met_index_map, metamorphosis_words, freq)


# In[16]:


#mean along columns (averaging over samples) to whiten the data: 1xn_2
h_final_mean = np.mean(h_final,axis=1)

#to whiten data, we need to transform our mean values to a 50x1000 matrix that we can then subtract from h_final
h_final = h_final - (h_final_mean.reshape(n_2,1)*np.ones((1,n_1)))


# In[17]:


#perform pca transform to reduce over colums, from 5 dimensions to 2. end up with 16 2-dimensional vectors (16x2)
pca1, C, var = pca(np.transpose(h_final))

#print(pca1)
#print(C)
#print(var)

#these should be 1x16 vectors corresponding to the columns of pca (and they are!)
a_final = np.matmul(np.transpose(h_final), C[0,:])
b_final = np.matmul(np.transpose(h_final), C[1,:])


# In[20]:


pca_plotter(n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, 1, 100)


# In[21]:


pca_plotter(n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, 1, 10)


# In[22]:


pca_plotter(n_1, n_2, H, h_final_mean, C, trial_n, freq, text_final, 1, 5)

