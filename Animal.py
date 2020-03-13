#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Animal class that encapsulates data generated from tree (animals and their features)
class Animal(object):
    def __init__(self, animal_n, total_n, feature):
        self.input = np.zeros(total_n,dtype=np.float64)
        self.input[animal_n] = 1.0
        self.feature = feature
    


# In[ ]:




