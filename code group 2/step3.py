import cPickle as pickle
import numpy as np
import pandas as pd
import glob
import pylab
import pyfits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from astropy.table import Table

CCD = sys.argv[1]
epoch = sys.argv[2]
name = sys.argv[3]
dir_cat = "/home/student01/out_team1/"
dir_ima = "/home/student01/out_team1/"
dir_out_fits = "/home/student02/fits4/" + name +"/"
dir_out_image = "/home/student02/image4/" + name + "/"
dir_dic = "/home/student02/dic/"+CCD + name

dic = pd.io.pickle.read_pickle(dir_dic)

#drawPlot(CCD, epoch, dir_out, dic)

def drawPlot(CCD, epoch, dir_out_fits, dir_out_image, dic):
    imageName = dir_ima + CCD + epoch + "_image_clean.fits.fz"
    fileName = dir_cat + CCD + epoch + "_image_clean.fits.cat"
    image = pyfits.open(imageName)
    data_image = image[0].data
   
    choose = ['X_IMAGE', 'Y_IMAGE', 'A_IMAGE']
    subdic =  dic[fileName]
    header = pyfits.getheader(imageName)
    t = Table.read(fileName, table_id=0)
    data = pd.DataFrame(np.array(t))
    data = data[choose].values
    
    for j in xrange(len(subdic)):
        print j, len(subdic) - j
        sizex = max(40,2.5*data[j,2])
        sizey = max(40,2.5*data[j,2])
        new = np.zeros((sizex, sizey))
        if subdic[j]!= -1:
            x = np.around(data[subdic[j]-1,0])
     
            y = np.around(data[subdic[j]-1,1])
            
            ymin = max(0, y-sizey/2+1)
            ymax = min(data_image.shape[0], y+sizey/2+1)
            
            xmin = max(0, x-sizex/2+1)
            xmax = min(data_image.shape[1], x+sizex/2+1)
            
            new = data_image[ymin:ymax,xmin:xmax]
        
            pyfits.writeto(dir_out_fits+CCD+epoch+'_%d.fits'%(subdic[j]),new,header)
            pylab.imshow(new, cmap='gray',interpolation = 'nearest')
            pylab.savefig(dir_out_image+CCD+epoch+'_%d.png'%(subdic[j]))
            #plt.show()

drawPlot(CCD, epoch, dir_out_fits, dir_out_image, dic)

