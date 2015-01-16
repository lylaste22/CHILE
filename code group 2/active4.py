import cPickle as pickle
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pylab
import pyfits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.table import Table
import sys

CCD = sys.argv[1]

dir_cat = '/home/student02/dic/' + CCD
dir_image = '/home/student01/out_team1/' + CCD
dir_out = '/home/student02/act/' + CCD
obj_dict_name =  'object'
star_dict_name =  'star'
galaxy_dict_name =  'galaxy'
no_dict_name =  'undetermined'

ind_PCA = [0, 11, 14, 18, 20, 21, 22, 23]
#num_feature = len(info_name)


dir_train = "/home/student02/train/"

def readFromFile(filename, column_names, info_name, fit_data, add_position, total_obs, dic_sub):
    #data = np.loadtxt(filename)
    #data = pd.DataFrame(data=data, columns = column_names)
    t = Table.read(filename, table_id=0)
    data = pd.DataFrame(np.array(t))
    data = data[info_name].as_matrix()
    plt.hist(data[:,2], bins = 10)
    for i in xrange(total_obs):
       if dic_sub[i]!=-1:
  #          ##################
            ##################
            ## Starting 1#####
            ##################
            ##################
            #print dic_sub[i]
            fit_data[i][int(add_position[i]),:] = data[int(dic_sub[i])-1,:]
            add_position[i] += 1
    return fit_data, add_position

def findMedian(fit_data):
    return np.median(fit_data,axis=0)


#def combine_feature(dir_train):
#    files =  glob.glob(dir_train+"*feature")
#    res = None
#    print len(files)
#    for i, f in enumerate(files):
        #print i
#	if res==None:
#	   res = pd.io.pickle.read_pickle(f).values
#	else:
#           res = np.vstack((res, pd.io.pickle.read_pickle(f).values))
 #   num = res.shape[1]
    #feature = res[:,0:(num-1)]
    #label = res[:,(num-1)]

  #  return res

#info_name = info_name.pop()
#feature  = combine_feature(dir_train)
#print len(label)

#feature = pd.DataFrame(feature)
#feature.to_pickle(dir_train+"train_set")

#feature = pd.io.pickle.read_pickle(dir_train+"train_set").values

#info_name = pd.io.pickle.read_pickle(dir_train+"train_set").columns

#label = feature[:,-1]

#test = (label-0.01)*(0.99-label)

#num = feature.shape[1]

#feature = [test<=0,0:(num-1)]
#label = label[test<=0]

#feature.to{pickle("train_final")
#feature = feature[test<=0,:]
#feature2 = pd.DataFrame(feature, columns = info_name)
#feature2.to_pickle(dir_train+"train_final")
#feature = feature[:,0:(num-1)]
#print len(label)

#realTrainIndex = np.random.randint(2,size=feature.shape[0])
#train = feature[realTrainIndex==1,:]
#test = feature[realTrainIndex==0,:]
#trainLabel = label[realTrainIndex==1]
#testLabel = label[realTrainIndex==0]

#print train.shape, test.shape

#def findMedian(fit_data):
#    return np.median(fit_data,axis=0)


def feature_PCA(trainPara, testPara, ind0, ind1):
    train = trainPara[:,ind0:ind1]
    test = testPara[:,ind0:ind1]

    if ind1 - ind0 == 1:
        return train.reshape((train.shape[0],1)), test.reshape((test.shape[0],1))
    if ind1 - ind0 == 2:
        return train.mean(axis = 1).reshape((train.shape[0],1)), test.mean(axis = 1).reshape((test.shape[0],1))
    
    if ind1 - ind0 > 2:
        pca = PCA()
        train_pca = pca.fit_transform(train)
        test_pca = pca.fit_transform(test)
        lam = pca.explained_variance_ratio_
        
        n = 0
        lam_sum = 0
        
        while lam_sum <=0.95:
            lam_sum += lam[n]
            n += 1
        
        return train_pca[:,0:n], test_pca[:,0:n]

def change_set(trainPara, testPara, ind_PCA):
    train = None
    test = None
    for i in xrange(len(ind_PCA)-1):
        if train is None:
            train, test = feature_PCA(trainPara, testPara, ind_PCA[i], ind_PCA[i+1])
        else:
            train0, test0  = feature_PCA(trainPara, testPara, ind_PCA[i], ind_PCA[i+1])
          
	    train = np.hstack((train, train0))
	    test = np.hstack((test, test0))

    return train, test

use = pd.io.pickle.read_pickle(dir_train+"train_final")
trainPara = use.values
#print trainPara[0,:]
num = trainPara.shape[1]
trainPara = trainPara[:,0:(num-1)]
trainLabel = use.values[:,-1]
num_feature = num - 1
#info_name = use.columns
#info_name = info_name.pop()
#print info_name
#train, test = change_set(trainPara, testPara, ind_PCA)

#print train.shape, test.shape
#trainPara, testPara = change_set()
info_name = ['MAG_ISO','MAG_ISOCOR','MAG_AUTO','MAG_PETRO', 'MAG_BEST', 'MAG_WIN',
             'MAG_PSF', 'MAG_MODEL','MAG_POINTSOURCE', 'MAG_SPHEROID', 'MAG_DISK',
             'MU_EFF_MODEL', 'MU_EFF_SPHEROID', 'MU_EFF_DISK',
             'CHI2_PSF', 'CHI2_MODEL', 'NITER_MODEL', 'NITER_PSF',
             'KRON_RADIUS', 'PETRO_RADIUS',
             'ELLIPTICITY',
             'ISOAREA_IMAGE',
             'FWHM_IMAGE',
             'CLASS_STAR']


def clf_method(method, trainPara, testPara, trainLabel, n_forest):
    if method =='LDA':
        clf = LDA()
    elif method =='QDA':
        clf = QDA()
    elif method == 'SVC':
        clf = svm.SVC()
    elif method == 'NuSVC':
        clf = svm.NuSVC()
    elif method =='RF':
        clf = RandomForestClassifier(n_estimators=n_forest)
    else:
        print 'wrong method'
    clf.fit(trainPara, trainLabel)
    testLabel = clf.predict(testPara)
    return testLabel

#def clf_error(method, train, test, trainLabel, testLabel, n_forest):
#    newLabel = clf_method(method, train, test, trainLabel, n_forest)
#    print len(newLabel)
#    return 1 - abs(testLabel - newLabel).mean()/2

#print clf_error('LDA', train, test, trainLabel, testLabel, 10)
#print clf_error('QDA', train, test, trainLabel, testLabel, 10)
#print clf_error('SVC', train, test, trainLabel, testLabel, 10)
#print clf_error('NuSVC', train, test, trainLabel, testLabel, 10)
#print clf_error('RF', train, test, trainLabel, testLabel, 10)


#def clf_choose(trainPara, trainLabel, n_forest, total_methods):
 #   realTrainIndex = np.random.randint(2,size=trainPara.shape[0])
  #  trainPara0 = trainPara[realTrainIndex==1,:]
  #  trainPara1 = trainPara[realTrainIndex==0,:]
  #  trainLabel0 = trainLabel[realTrainIndex==1]
  #  trainLabel1 = trainLabel[realTrainIndex==0]
    
 #   error = np.empty(len(total_methods))
 #   for i, method in enumerate(total_methods):
  #      error[i] =  clf_error(method, trainPara0, trainPara1, trainLabel0, trainLabel1, n_forest)

   # choose = np.sort(error)[0:2]
    
 #   use_methods = [total_methods[error == choose[0]],total_methods[error == choose[1]]]
    
  #  return use_methods
    


def clf_result(trainPara, testPara, trainLabel, n_forest):
    testLabelLDA = clf_method('LDA', trainPara, testPara, trainLabel, n_forest)
    #testLabelQDA = clf_method('QDA', trainPara, testPara, trainLabel, n_forest)
    testLabelRF = clf_method('RF', trainPara, testPara, trainLabel, n_forest)
    
    testLabel = np.zeros(testPara.shape[0])
    
    for i in xrange(testLabel.size):
        #if testLabelLDA[i] == testLabelSVC[i] and testLabelLDA[i] == testLabelRF[i]:
        if testLabelLDA[i] == testLabelRF[i]: # and testLabelLDA[i] == testLabelQDA[i]:
            testLabel[i] = testLabelLDA[i]
        #testLabel[i] = testLabelRF[i]
    #print testLabel        
    return testLabel



def get_dictionary(dir_cat, obj_dict_name):
    dic = pd.io.pickle.read_pickle(dir_cat + obj_dict_name)
    filenames = dic.columns.values
    files = dic.columns.values.copy()
    #print files
    dic = dic.values
    total_obs = dic.shape[0]
    num_obs = np.empty(total_obs)
    num_index = np.zeros(total_obs)
    for i in xrange(total_obs):
        num_obs[i] = len(dic[i,dic[i,]!=-1])
        if num_obs[i] > 0:
            num_index[i] = 1
            
    dic = dic[num_index==1,:]
    num_obs = num_obs[num_index==1]
    
    total_obs = dic.shape[0]
    
    #for i, f in enumerate(filenames):
        #files[i] = dir_cat + files[i]

    return dic, filenames, files, total_obs, num_obs    



#def get_column_name(filename):
 #   column_names = []
  #  with open(filename) as f:
   #     for line in f:
    #        if line[0] == '#':
     #           column_names.append(line.split()[2])
      #      else:
       #         break
    
    #return column_names



def get_parameter(num_feature, info_name, ind_PCA, dir_cat, obj_dict_name, trainPara):
    dic, filenames, files, total_obs, num_obs = get_dictionary(dir_cat, obj_dict_name)
    #column_names = get_column_name(files[0])
    column_names = []
    #MAGPLOT = np.zeros(total_obs)
    fit_data = []
    for i in xrange(total_obs):
        fit_data.append(np.empty(num_feature*num_obs[i]).reshape((num_obs[i],num_feature)))
    
    add_position = np.zeros(total_obs)
    
    for i in xrange(files.size):
        fit_data, add_position = readFromFile(files[i], column_names, info_name, fit_data, add_position, total_obs, dic[:,i])
    
    testPara  = np.empty(total_obs*num_feature).reshape((total_obs, num_feature))
    
    for i in xrange(total_obs):
        testPara[i,:] = findMedian(fit_data[i])
    
    #MAGPLOT = fit[:,2]
    #print MAGPLOT
    #plt.hist(MAGPLOT, bins= 10)

    #plt.savefig("/home/student02/hist.png")
    #para = None
    #for i in xrange(len(ind_PCA)-1):
        #if para is None:
            #para = feature_PCA(train, test, ind_PCA[i], ind_PCA[i+1])
      #  else:
       #     para = np.hstack((para, feature_PCA(fit, ind_PCA[i], ind_PCA[i+1])))

    train, test = change_set(trainPara, testPara, ind_PCA)

    return train, test, dic, filenames
   # totalLabel = np.zeros(para.shape[0])
    
   # totalLabel[fit[:,-1]>=0.9] = 1
   # totalLabel[fit[:,-1]<=0.1] = -1
    
#    trainPara = para[totalLabel!=0,:]
 #   testPara = para[totalLabel==0,:]
  #  trainLabel = totalLabel[totalLabel!=0]
    
 
   # return trainPara, testPara, trainLabel, totalLabel, dic, filenames



def star_dictionary(num_feature, info_name, ind_PCA, dir_cat, obj_dict_name, star_dict_name, galaxy_dict_name, no_dict_name, trainPara, trainLabel):
    train, test, dic, filenames = get_parameter(num_feature, info_name, ind_PCA, dir_cat, obj_dict_name, trainPara)
    totalLabel = clf_result(train, test, trainLabel, 10)
    #print totalLabel 
   # step = 0
    
    #for i in xrange(totalLabel.size):
     #   if totalLabel[i] == 0:
      #      totalLabel[i] = testLabel[step]
       #     step+= 1
    
    star = pd.DataFrame(dic[totalLabel == 1,:], columns = filenames)
    pickle.dump(star, open(dir_out + star_dict_name,'wb'))    
    star.to_pickle(dir_out+star_dict_name)

    galaxy = pd.DataFrame(dic[totalLabel == -1,:], columns = filenames)
    pickle.dump(galaxy, open(dir_out + galaxy_dict_name,'wb'))
    galaxy.to_pickle(dir_out+galaxy_dict_name)
   
    no = pd.DataFrame(dic[totalLabel == 0,:], columns = filenames)
    pickle.dump(no, open(dir_out + no_dict_name,'wb'))    
    no.to_pickle(dir_out+no_dict_name)

star_dictionary(num_feature, info_name, ind_PCA, dir_cat, obj_dict_name, star_dict_name, galaxy_dict_name, no_dict_name, trainPara, trainLabel)
