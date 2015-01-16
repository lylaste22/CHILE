
# In[33]:

import numpy as np
import math
import matplotlib.pyplot as plt


# In[39]:

def first_index(N):
    out = []
    for i in range(1,N):
        for j in range(i,N):
            out.append(j)
    out = np.array(out)
    return out


# In[40]:

def second_index(N):
    out = []
    for i in range(N):
        for j in range(i,N-1):
            out.append(i)
    out = np.array(out)   
    return out 


# In[41]:

singles = np.genfromtxt("singles.csv", delimiter=',', dtype=None, comments='#')

N = singles[:,0].size

ind1 = first_index(N)
ind2 = second_index(N)
indN = ind1.size

# normalizing the distribution
ra = singles[:,0]-np.mean(singles[:,0])
dec = singles[:,1]-np.mean(singles[:,1])
time = singles[:,2]-np.mean(singles[:,2])

# determining the same epochs
# same = 1 (yes) or 0 (no)
same = []
for i in range(ind1.size):
    if (time[ind1[i]] == time[ind2[i]]):
        same.append(1)
    else:
        same.append(0)
same = np.array(same)


# In[44]:

x0=[];
y0=[];
V=[];
theta=[];

for i in range(indN):
    if (same[i]==0): #if not the same epoch
        x0_aux = (ra[ind1[i]]*time[ind2[i]] - ra[ind2[i]]*time[ind1[i]])/(time[ind2[i]]-time[ind1[i]]) 
        y0_aux = (dec[ind1[i]]*time[ind2[i]] - dec[ind2[i]]*time[ind1[i]])/(time[ind2[i]]-time[ind1[i]])
        V_aux = np.sqrt( (np.power(ra[ind1[i]]-ra[ind2[i]],2) + np.power(dec[ind1[i]]-dec[ind2[i]],2))/(time[ind1[i]]-time[ind2[i]]) )
        theta_aux = math.atan2( (dec[ind1[i]] - dec[ind2[i]]) , (ra[ind1[i]]-ra[ind2[i]]) )
        
        x0.append(x0_aux)
        y0.append(y0_aux)
        V.append(V_aux)
        theta.append(theta_aux)

x0 = np.array(x0)
y0 = np.array(y0)
V = np.array(V)
theta = np.array(theta)


# In[ ]:



