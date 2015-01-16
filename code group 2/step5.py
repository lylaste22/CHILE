import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import linear_model
import pylab
from mpl_toolkits.mplot3d import Axes3D
import sys

FIELD = sys.argv[1]
dir_in = "/home/student02/ast3/"
dir_out = "/home/student02/ast4/"
#dir_out2 = "/home/student02/ast2/"

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


julian_threshold = 0.00001
def filter_min_epochs(asteroids, n_epochs):
    for ast in asteroids:
        occurrences = ast.shape[0] 
        #print('ast shape'),
        #print(occurrences)        
        if (occurrences < 6):
            #print("in")
            removearray(asteroids, ast)

def filter_same_epoch(asteroids, unique_epochs):
    for epoch in unique_epochs:
        for i in range(len(asteroids)):
            ast = asteroids[i]
            sources = ast[np.absolute(ast[:,2] - epoch) < julian_threshold,:]
            if(sources.shape[0] > 1):
#                 print('filtrando asteroide '),
#                 print(ast)
                asteroids[i] = filter_asteroid(ast, epoch)
                
def filter_asteroid(asteroid, epoch):
    same_epoch_sources = asteroid[np.absolute(asteroid[:,2] - epoch) < julian_threshold,:]
    other_sources = asteroid[np.absolute(asteroid[:,2] - epoch) >= julian_threshold,:]
    x = other_sources[:,[0,2]]
    y = other_sources[:,1]
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    fitness = np.zeros(same_epoch_sources.shape[0])
    for i in range(same_epoch_sources.shape[0]):
        y_prediction = regr.predict(same_epoch_sources[i,[0,2]])
#         print('y prediction '),
#         print(y_prediction)
        y_real = same_epoch_sources[i,1]
#         print('y real '),
#         print(y_real)
        fitness[i] = np.absolute(y_real-y_prediction)
    index = np.argmin(fitness)
    selected_source = same_epoch_sources[index,:]
    return np.vstack((other_sources,selected_source))
    
    
def plot_labels(labels):
    plt.clf()
    plt.hist(labels, bins=np.arange(labels.min(), labels.max()+1))
    plt.show()


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

def moving_object(data):
    #data = date_choose(data, day)
    
    para = np.zeros(8*data.shape[0]**2).reshape((data.shape[0]**2,8))
    
    n = data.shape[0]
    
    meanx = data[:,1].mean()
    meany = data[:,2].mean()
    meant = data[:,3].mean()
    
    data[:,1] -= meanx
    data[:,2] -= meany
    data[:,3] -= meant
    

    for i in xrange(n):
	#iprint i
        para[(i*n):(i*n+n),0] = data[i,1]
        para[(i*n):(i*n+n),1] = data[:,1]
        para[(i*n):(i*n+n),2] = data[i,2]
        para[(i*n):(i*n+n),3] = data[:,2]
        para[(i*n):(i*n+n),4] = data[i,3]
        para[(i*n):(i*n+n),5] = data[:,3]
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
    
    db = DBSCAN(eps=0.0025, min_samples=4).fit(X)
    
    labels = db.labels_
  #  labels = filter_labels(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
  #  asteroid = []
    
    data[:,1] += meanx
    data[:,2] += meany
    data[:,3] += meant
    
    
#     for i in xrange(n_clusters_):
#         re = para[labels==i,6:8]
#         m = re.shape[0]
#         index = np.empty(m*2)
#         index[0:m] = re[:,0]
#         index[m:(2*m)] = re[:,1]
#         num = np.asarray(list(set(index)), dtype=int)
        
#         if len(num) <= 5  :
#             asteroid.append(data[num,0:2])
        

    return labels, para



def get_asteroids(data, labels, para):
    asteroids = []
    n_clusters_ = len(set(labels))
    print(n_clusters_)
    for i in xrange(n_clusters_):
        re = para[labels==i,6:8]
        m = re.shape[0]
        index = np.empty(m*2)
        index[0:m] = re[:,0]
        index[m:(2*m)] = re[:,1]
        num = np.asarray(list(set(index)), dtype=int)       
        asteroids.append(data[num,:])
    return asteroids


def plot_data(asteroid):
    plt.clf()
    plt.scatter(data[:,0],data[:,1], s=1)
    if len(asteroid)>0:
        colors = plt.cm.Spectral(np.linspace(0, 1, len(asteroid)))
        for i in xrange(len(asteroid)):
            col = colors[i]
            plt.plot(asteroid[i][:,0], asteroid[i][:,1], 'o', color = col)
            plt.plot(asteroid[i][:,0], asteroid[i][:,1], color = col)
    plt.show()


def plot_data_3d(asteroid):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

   # plt.scatter(data[:,0],data[:,1], s=1)
    if len(asteroid)>0:
        colors = plt.cm.Spectral(np.linspace(0, 1, len(asteroid)))
        for i in xrange(len(asteroid)):
            col = colors[i]
            ax.scatter(asteroid[i][:,0], asteroid[i][:,1], asteroid[i][:,2],marker='o',color = col)
            ax.plot(asteroid[i][:,0], asteroid[i][:,1], asteroid[i][:,2], color = col)
        
    plt.show()


def plot_data_3d_2(asteroid):

   # plt.scatter(data[:,0],data[:,1], s=1)
    if len(asteroid)>0:
        colors = plt.cm.Spectral(np.linspace(0, 1, len(asteroid)))
        for i in xrange(len(asteroid)):
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            col = colors[i]
            ax.scatter(asteroid[i][:,0], asteroid[i][:,1], asteroid[i][:,2],marker='o',color = col)
            ax.plot(asteroid[i][:,0], asteroid[i][:,1], asteroid[i][:,2], color = col)     
            plt.show()


def write_txt(asteroid, f):
    num = len(asteroid)
    for i in xrange(num):
	e_num = asteroid[i].shape[0]
	for j in xrange(e_num):
            res = str(asteroid[i][j,0]) + " " + str(asteroid[i][j,1]) + " " + str(asteroid[i][j,2]) + " " + str(asteroid[i][j,3]) + " " + str(asteroid[i][j,4]) + " " + str(asteroid[i][j,5]) + "\n"# + str(asteroid[i][j,6]) + "\n"
	    #f.write(asteroid[i][j,0])
            #f.write(" ")
            #f.write(asteroid[i][j,1])
            #f.write(" ")
            #f.write(asteroid[i][j,2])
            #f.write(" ")
            #f.write(asteroid[i][j,3])
            #f.writelines(" ")
            f.writelines(res)
	f.writelines("\n")

####
data = pd.read_csv(dir_in+"field"+FIELD+".csv", header=-1).values
#data = pd.io.pickle.read_pickle(dir_in+CCD+"moving").values
#data = np.loadtxt(dir_in, delimiter=",")

labels, para = moving_object(data)



asteroids = get_asteroids(data, labels, para)
#print asteroids
#f1 = open(dir_out1+CCD+"asteroids.txt","w")
#write_txt(asteroids, f1)
#f1.close()

unique_epochs = list(set(data[:,3]))
print unique_epochs
filter_same_epoch(asteroids,unique_epochs)
filter_min_epochs(asteroids,len(unique_epochs))


f2 = open(dir_out+FIELD+"_asteroids.txt","w")
write_txt(asteroids, f2)
f2.close()

#plot_data(asteroids)


#plot_data_3d(asteroids)


#plt.clf()
#plt.hist(labels)
#plt.show()





