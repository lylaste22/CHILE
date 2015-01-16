import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter
import glob
import re
import pandas as pd
import sys
import pyfits
from astropy.table import Table

CCD = sys.argv[1]
dir_in_CCD = '/home/student01/out_team1/'+CCD
dir_in_CCD_image = '/home/student01/out_team1/'+CCD
dir_out_CCD = '/home/student02/dic/' + CCD +'object'
dir_out_CCD_moving = '/home/student02/dic/' + CCD +'moving'

def get_column_name(filename):
    column_names = []
    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                column_names.append(line.split()[2])
            else:
                break
    
    return column_names

def file_number(file_name):
    #_index = [m.start() for m in re.finditer('_', file_name)][-1]
    #ext_index = [m.start() for m in re.finditer('\.', file_name)][-1]
    pieces = file_name.split('/')[-1]
    #print pieces
    numbers = pieces.split('_')[3]
    #print numbers
    return float(numbers)
    #return int(file_name[_index+1:ext_index])


files = glob.glob(dir_in_CCD+"*.fits.cat")
#print files
files = sorted(files, key=lambda f: file_number(f))

#print files
#images = glob.glob(dir_in_CCD_image+"*.fz")

#print files
#print images
#columns = [0,55,56,174,11,10,180]
column_names = get_column_name(files[0])
Data = None
#start = 0
#data_by_epochs = []

info_name = ['NUMBER','X_WORLD','Y_WORLD','FLAGS','FLUXERR_AUTO','FLUX_AUTO','MAG_AUTO']
for i, file_ in enumerate(files):
   
    #f = np.loadtxt(file_)[:,columns]
    #hdulist = pyfits.open(images[i])
    #epoc_time = hdulist[0].header['MJD-OBS']i
    #epoc_time = i
    #f = np.loadtxt(file_)
    #f = pd.DataFrame(data=f, columns = column_names)
    t = Table.read(file_, table_id=0)
    f = pd.DataFrame(np.array(t))
    f = f[info_name].values
    #print f.shape
    #f = np.hstack((f,np.zeros(f.shape[0], dtype=float).reshape((f.shape[0],1))+epoc_time))
    f = np.hstack((f,np.zeros(f.shape[0], dtype=int).reshape((f.shape[0],1))+i))
    f = np.hstack((f,np.zeros(f.shape[0], dtype=int).reshape((f.shape[0],1))+file_number(file_)))
    #f = np.hstack((f,np.array(f.shape[0]*[file_]).reshape((f.shape[0],1))))

    #print f[0,:]
    if Data is None:
       Data = f
    else:
       Data = np.vstack((Data,f))

flag_threshold = 0
filt = np.ones(Data.shape[0], dtype=bool)
filt[Data[:,3]>flag_threshold] = False
Data = Data[filt,:]

snr_threshold = 0.15
filt2 = np.ones(Data.shape[0], dtype=bool)
snr = Data[:,4]/Data[:,5]
filt2[snr>snr_threshold] = False
Data = Data[filt2,:]

X = Data[:,[1,2]]

num_epoch = int(len(files)/4)

db = DBSCAN(eps=0.0003, min_samples= num_epoch ).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = np.asarray(db.labels_, dtype=int)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)

n_stat = labels[labels>-1].size

static = np.zeros(n_clusters_*len(files)).reshape((n_clusters_,len(files))) - 1
#print Data.shape
#print len(labels)

for i in xrange(len(labels)):
#for i in xrange(len(labels)):
  if labels[i]>-1:
    static[labels[i],Data[i,7]] = Data[i, 0]
#print labels
pdstatic= pd.DataFrame(data=static, columns = files)

moving = Data[:, [0, 1,2,8,5,4, 6]]

moving = moving[labels==-1,:]

#for i, file_ in enumerate(files):
#	Data[Data[:,-1]==i,-1] = file_

pdstatic.to_pickle(dir_out_CCD)

pdmoving = pd.DataFrame(data=moving, columns = ["NUMBER","X","Y","Julian","FLUX","FLUX_ERR","MAG_AUTO"])
pdmoving.to_pickle(dir_out_CCD_moving)
