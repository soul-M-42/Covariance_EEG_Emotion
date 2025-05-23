import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt

datadir = '/mnt/dataset0/**/AutoICA_Processed_EEG/Faced/Processed_data_filter_epoch_0.50_47_Auto_ICA_def_Threshold/processed_data'
files = os.listdir(datadir)
files = sorted(files)
# print(files)

data_all = np.zeros((123,30,228848))
for i in range(123):
    data_all[i] = sio.loadmat(os.path.join(datadir, files[i]))['data_all_cleaned']

n_samples_one = sio.loadmat(os.path.join(datadir, files[i]))['n_samples_one'][0]
n_samples_cum = np.cumsum(np.concatenate((np.array([0]), np.round(n_samples_one)))).astype(int)
n_points_cum = np.cumsum(np.concatenate((np.array([0]), np.round(n_samples_one*125)))).astype(int)

fs = 125
fea = np.zeros((123,30,30*28))
count = 0
for i in range(len(n_samples_one)):
    data_tmp = data_all[:,:,n_points_cum[i]:n_points_cum[i+1]]
    
    for j in range(int(np.ceil(n_samples_one[i]))-30, int(np.ceil(n_samples_one[i]))):
        if j < int(np.ceil(n_samples_one[i]))-1:
            fea[:,:,count] = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data_tmp[:,:,j*fs:(j+1)*fs], -1))).squeeze()
        else:
            fea[:,:,count] = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data_tmp[:,:,j*fs:], -1))).squeeze()
        
        count += 1
        
print(fea.shape)
