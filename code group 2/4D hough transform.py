
# In[8]:

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pylab


# In[8]:




# In[9]:

# r, theta: hough transform, v: velocity
def hough_transformation(x1, x2, y1, y2, t1, t2):
    
    t_diff = t2 - t1
    t_absdiff  = np.absolute(t_diff)
    x0 = (x1*t2 - x2*t1)/t_diff
    y0 = (y1*t2 - y2*t1)/t_diff
    y_diff = y1 - y2
    x_diff = x1 - x2
    v = np.sqrt((x_diff)**2+(y_diff)**2)/t_absdiff
    theta = np.arctan(y_diff/x_diff)
    
    return x0, y0, theta, v


# In[10]:

def date_choose(data, day):
    time = np.sort(np.asarray(list(set(data[:,2]))))
    if day==0:
        data = data[data[:,2]<time[4],:]
    if day==4:
        data = data[data[:,2]>=time[16],:]
    if day > 0 and day < 4:
        data = data[data[:,2]<time[day*4+4],:]
        data = data[data[:,2]>=time[day*4],:]
                         
    return data


# In[11]:

def moving_object(data, day):
    data = date_choose(data, day)
    
    para = np.zeros(8*data.shape[0]**2).reshape((data.shape[0]**2,8))
    
    n = data.shape[0]
    
    meanx = data[:,0].mean()
    meany = data[:,1].mean()
    meant = data[:,2].mean()
    
    data[:,0] -= meanx
    data[:,1] -= meany
    data[:,2] -= meant
    

    for i in xrange(n):
        para[(i*n):(i*n+n),0] = data[i,0]
        para[(i*n):(i*n+n),1] = data[:,0]
        para[(i*n):(i*n+n),2] = data[i,1]
        para[(i*n):(i*n+n),3] = data[:,1]
        para[(i*n):(i*n+n),4] = data[i,2]
        para[(i*n):(i*n+n),5] = data[:,2]
        para[(i*n):(i*n+n),6] = i
        para[(i*n):(i*n+n),7] = np.linspace(0,n-1,n)
        
    difft = para[:,5] - para[:,4]
    para = para[difft>0,:]
    x, y, theta, v = hough_transformation(para[:,0],para[:,1],para[:,2],para[:,3],para[:,4],para[:,5])
    
    X = np.empty(x.size*4).reshape((x.size, 4))
    X[:,0] = x
    X[:,1] = y
    X[:,2] = theta
    X[:,3] = v
    X = np.nan_to_num(X)
    
    db = DBSCAN(eps=0.01, min_samples=6).fit(X)
    
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    asteroid = []
    
    data[:,0] += meanx
    data[:,1] += meany
    data[:,2] += meant
    
    
    for i in xrange(n_clusters_):
        re = para[labels==i,6:8]
        m = re.shape[0]
        index = np.empty(m*2)
        index[0:m] = re[:,0]
        index[m:(2*m)] = re[:,1]
        num = np.asarray(list(set(index)), dtype=int)
        
        if len(num) == 4:
            asteroid.append(data[num,0:2])
        
            
    return data, asteroid


# In[12]:

def plot_moving(data, day):
    data, asteroid = moving_object(data, day)
    
    plt.clf()
    plt.scatter(data[:,0],data[:,1], s=1)
    if len(asteroid)>0:
        col = np.linspace(0,1,len(asteroid))
        for i in xrange(len(asteroid)):
            plt.scatter(asteroid[i][:,0], asteroid[i][:,1])
    #plt.savefig('image_%d.png'%(day))
    plt.show()


# In[13]:

data = pd.read_csv("singles.csv", header=-1).values


# In[14]:

for day in xrange(5):
    plot_moving(data,day)


# In[14]:

day


# Out[14]:

#     1

# In[ ]:



