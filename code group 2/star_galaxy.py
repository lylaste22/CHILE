
# In[92]:

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


# In[93]:

def readFromFile(filename, column_names, info_name, fit_data, add_position, total_obs, dic_sub):
    data = np.loadtxt(filename)
    data = pd.DataFrame(data=data, columns = column_names)
    data = data[info_name].as_matrix()
    
    for i in xrange(total_obs):
        if dic_sub[i]!=-1:
            fit_data[i][add_position[i],:] = data[dic_sub[i],:]
            add_position[i] += 1
    #print dic_sub
    return fit_data, add_position


# In[94]:

def findMedian(fit_data):
    return np.median(fit_data,axis=0)


# In[95]:

def feature_PCA(fit, ind0, ind1):
    feature = fit[:,ind0:ind1]
    if ind1 - ind0 == 1:
        return feature.reshape((feature.shape[0],1))
    if ind1 - ind0 == 2:
        return feature.mean(axis = 1).reshape((feature.shape[0],1))
    
    if ind1 - ind0 > 2:
        pca = PCA()
        feature_pca = pca.fit_transform(feature)
        lam = pca.explained_variance_ratio_
        
        n = 0
        lam_sum = 0
        while lam_sum <=0.95:
            lam_sum += lam[n]
            n += 1
        
        return feature_pca[:,0:n]


# In[96]:

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


# In[97]:

def clf_error(method, trainPara0, trainPara1, trainLabel0, trainLabel1, n_forest):
    trainLabel = clf_method(method, trainPara0, trainPara1, trainLabel0, n_forest)
    return abs(trainLabel1 - trainLabel).mean()/2


# In[98]:

def clf_choose(trainPara, trainLabel, n_forest, total_methods):
    realTrainIndex = np.random.randint(2,size=trainPara.shape[0])
    trainPara0 = trainPara[realTrainIndex==1,:]
    trainPara1 = trainPara[realTrainIndex==0,:]
    trainLabel0 = trainLabel[realTrainIndex==1]
    trainLabel1 = trainLabel[realTrainIndex==0]
    
    error = np.empty(len(total_methods))
    for i, method in enumerate(total_methods):
        error[i] =  clf_error(method, trainPara0, trainPara1, trainLabel0, trainLabel1, n_forest)

    choose = np.sort(error)[0:2]
    
    use_methods = [total_methods[error == choose[0]],total_methods[error == choose[1]]]
    
    return use_methods
    


# In[99]:

def clf_result(trainPara, testPara, trainLabel, n_forest):
    testLabelLDA = clf_method('LDA', trainPara, testPara, trainLabel, n_forest)
    testLabelSVC = clf_method('SVC', trainPara, testPara, trainLabel, n_forest)
    testLabelRF = clf_method('RF', trainPara, testPara, trainLabel, n_forest)
    
    testLabel = np.zeros(testPara.shape[0])
    
    for i in xrange(testLabel.size):
        if testLabelLDA[i] == testLabelSVC[i] and testLabelLDA[i] and testLabelLDA[i] == testLabelRF[i]:
            testLabel[i] = testLabelLDA[i]
            
    return testLabel


# In[122]:

def get_dictionary(dir_cat, dir_in, obj_dict_name):
    dic = pd.io.pickle.read_pickle(dir_in + obj_dict_name)
    filenames = dic.columns.values
    files = dic.columns.values.copy()
    #dic = dic.as_matrix(dic)
    dic = dic.values
    print dic[0,0]
    total_obs = dic.shape[0]
    num_obs = np.empty(total_obs)
    num_index = np.zeros(total_obs, dtype=int)
    for i in xrange(total_obs):
        num_obs[i] = len(dic[i,dic[i,]!=-1])
        if num_obs[i] > 0:
            num_index[i] = 1
    
    dic = dic[num_index==1,:]
    num_obs = num_obs[num_index==1]
    
    total_obs = dic.shape[0]
    
    for i, f in enumerate(filenames):
        files[i] = dir_cat + files[i]
    
    
        
    return dic, filenames, files, total_obs, num_obs    


# In[123]:

def get_column_name(filename):
    column_names = []
    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                column_names.append(line.split()[2])
            else:
                break
    
    return column_names


# In[124]:

def get_parameter(num_feature, info_name, ind_PCA, dir_cat, dir_in, obj_dict_name):
    dic, filenames, files, total_obs, num_obs = get_dictionary(dir_cat, dir_in, obj_dict_name)
    column_names = get_column_name(files[0])
    
    fit_data = [np.zeros(num_feature*num_obs[i]).reshape((num_obs[i],num_feature)) for i in xrange(total_obs)]
    for i in xrange(total_obs):
        fit_data.append(np.empty(num_feature*num_obs[i]).reshape((num_obs[i],num_feature)))
    
    #print dic[0,0]
    add_position = np.zeros(total_obs, dtype=int)
    
    for i in xrange(files.size):
        fit_data, add_position = readFromFile(files[i], column_names, info_name, fit_data, add_position, total_obs, dic[:,i])
    
    fit = np.empty(total_obs*num_feature).reshape((total_obs, num_feature))
    
    for i in xrange(total_obs):
        fit[i,:] = findMedian(fit_data[i])
    
    para = None
    for i in xrange(len(ind_PCA)-1):
        if para is None:
            para = feature_PCA(fit, ind_PCA[i], ind_PCA[i+1])
        else:
            para = np.hstack((para, feature_PCA(fit, ind_PCA[i], ind_PCA[i+1])))

    totalLabel = np.zeros(para.shape[0])
    
    totalLabel[fit[:,-1]>=0.9] = 1
    totalLabel[fit[:,-1]<=0.1] = -1
    
    trainPara = para[totalLabel!=0,:]
    testPara = para[totalLabel==0,:]
    trainLabel = totalLabel[totalLabel!=0]
    
 
    return trainPara, testPara, trainLabel, totalLabel, dic, filenames


# In[125]:

def star_dictionary(num_feature, info_name, ind_PCA, dir_cat,dir_in,dir_out, obj_dict_name, star_dict_name):
    trainPara, testPara, trainLabel, totalLabel, dic, filenames = get_parameter(num_feature, info_name, ind_PCA, dir_cat,dir_in, obj_dict_name)
    testLabel = clf_result(trainPara, testPara, trainLabel, 10)
    
    step = 0
    
    for i in xrange(totalLabel.size):
        if totalLabel[i] == 0:
            totalLabel[i] = testLabel[step]
            step+= 1
    
    dictionary = pd.DataFrame(dic[totalLabel == 1], columns = filenames)
    dictionary.to_pickle(dir_out + star_dict_name)

# In[126]:

info_name = ['MAG_ISO','MAG_ISOCOR','MAG_AUTO','MAG_PETRO', 'MAG_BEST', 'MAG_WIN', 
             'MAG_PSF', 'MAG_MODEL','MAG_POINTSOURCE', 'MAG_SPHEROID', 'MAG_DISK',
             'MU_EFF_MODEL', 'MU_EFF_SPHEROID', 'MU_EFF_DISK', 
             'CHI2_PSF', 'CHI2_MODEL', 'NITER_MODEL', 'NITER_PSF',
             'KRON_RADIUS', 'PETRO_RADIUS',
             'ELLIPTICITY', 
             'ISOAREA_IMAGE',
             'FWHM_IMAGE',
             'CLASS_STAR']

ind_PCA = [0, 11, 14, 18, 20, 21, 22, 23]
num_feature = int(len(info_name))


# In[127]:

dir_cat = '../cat/'
dir_in = '../dic/'
dir_out = '../dic/'
obj_dict_name = 'pickle_pandas_table'
star_dict_name = 'star_pandas_table'


# In[128]:

star_dictionary(num_feature, info_name, ind_PCA, dir_cat, dir_in, dir_out,obj_dict_name, star_dict_name)


# In[ ]:



