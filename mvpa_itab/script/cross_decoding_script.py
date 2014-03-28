from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, ndimage
import nibabel as ni
from sklearn import linear_model, svm
from sklearn.utils import check_random_state
from sklearn.cross_validation import KFold
from sklearn.feature_selection import f_regression

import nibabel

from nilearn import decoding
import nilearn.masking

from mvpa2.suite import *

def create_figure(size, roi_size, position=[1,2,3,4]):
    
    a = np.zeros((size,size, size))
    
    b = 2
    
    for p in position:
        x = np.array([b,b+roi_size])
        y = np.array([b,b+roi_size])
        if np.mod(p,2) == 0:
            y = -y[::-1]
        
        if p > 2:
            x = -x[::-1]
            
        a[x[0]:x[1], y[0]:y[1], :] = 1
       
    return a


def create_simulation_data(img, snr=0, n_samples=2 * 10, random_state=1):
    generator = check_random_state(random_state)
    
    smooth_X = 1 
    
    x_size = img.shape[0]
    y_size = img.shape[1]
    z_size = img.shape[2]
    
    img = img.ravel()
    
    ### Generate smooth background noise
    XX = generator.randn(x_size, y_size, z_size, n_samples)
    noise = []
    for i in range(n_samples):
        Xi = ndimage.filters.gaussian_filter(XX[:, :, :, i], smooth_X)
        Xi = Xi.ravel()
        noise.append(Xi)
    noise = np.array(noise)
    ### Generate the signal y
    #y = generator.randn(n_samples)
    #X = np.dot(y[:, np.newaxis], a[np.newaxis])
    y = np.ones(n_samples)
    X = np.dot(y[:, np.newaxis], img[np.newaxis])
    norm_noise = linalg.norm(X, 2) / np.exp(snr / 20.)
    noise_coef = norm_noise / linalg.norm(noise, 2)
    noise *= noise_coef
    snr_db = 20 * np.log(linalg.norm(X, 2) / linalg.norm(noise, 2))
    print ("SNR: %.1f dB" % snr_db)
    ### Mixing of signal + noise and splitting into train/test
    X += noise

    return X, y

##################### DATA GENERATION #################################à
a = create_figure(12, 3, position=[1,2,4,3])
b = create_figure(12, 3, position=[2,4])

c = create_figure(12, 3, position=[1,4])
d = create_figure(12, 3, position=[3]) 

ds_set = [a,b,c,d]

for i in range(len(ds_set)):
    ni.save(ni.Nifti1Image(ds_set[i], np.eye(4)), '/media/DATA/fmri/cross_decoding_searchlight/dataset_'+str(i+1)+'.nii.gz')

snr_s = [1,2,3]

ls_x = []
ls_y = []
for snr in snr_s:
    for x,i in zip(ds_set, range(len(ds_set))):
        X_, y_ = create_simulation_data(x, snr=snr, n_samples=100, random_state=1)
        y_ += i
        ni.save(ni.Nifti1Image(X_.T.reshape(12,12,12,-1), np.eye(4)), 
               '/media/DATA/fmri/cross_decoding_searchlight/dataset_'+str(i+1)+'_snr_'+str(snr)+'_.nii.gz')
    
        ls_x.append(X_)
        ls_y.append(y_)
 
####################################################################

ds1 = fmri_dataset(samples, targets, chunks, mask, sprefix, tprefix, add_fa)


for snr in snr_s:
    img1 = ni.load('/media/DATA/fmri/cross_decoding_searchlight/dataset_'+str(1)+'_snr_'+str(snr)+'_.nii.gz')
    img2 = ni.load('/media/DATA/fmri/cross_decoding_searchlight/dataset_'+str(2)+'_snr_'+str(snr)+'_.nii.gz')
    img3 = ni.load('/media/DATA/fmri/cross_decoding_searchlight/dataset_'+str(3)+'_snr_'+str(snr)+'_.nii.gz')
    img4 = ni.load('/media/DATA/fmri/cross_decoding_searchlight/dataset_'+str(4)+'_snr_'+str(snr)+'_.nii.gz')
    
    ds1 = fmri_dataset([img1, img2], targets=y1, chunks=ds1.chunks)
    ds2 = fmri_dataset([img3, img4], targets=y1, chunks=ds1.chunks)
    
    ds1.sa['task'] = [1 for s in ds.targets]
    ds2.sa['task'] = [2 for s in ds.targets]
    
    ds_merged = vstack((ds1,ds2))
    
    ds_merged.a.update(ds2.a)
    
    clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
    cv = CrossValidation(clf, NFoldPartitioner(attr='task'))
    
    sl = sphere_searchlight(cv, 1.2, space = 'voxel_indices')
    sl_map = sl(ds_merged)
    
    sl_map.samples *= -1
    sl_map.samples +=  1
    

    
    ni.save(ni.Nifti1Image(sl_map.samples[0].reshape((12,12,12)), np.eye(4)), 
            '/home/robbis/sl_crossdecoding_snr_'+str(snr)+'_1_.nii.gz')
    ni.save(ni.Nifti1Image(sl_map.samples[0].reshape((12,12,12)), np.eye(4)), 
            '/home/robbis/sl_crossdecoding_snr_'+str(snr)+'_2_.nii.gz')
    

############# Haxby example #############################à

datapath = '/home/robbis/PhD/fmri_datasets/tutorial_data/data/'
if __debug__:
    debug.active += ["SLC"]

attr = SampleAttributes(os.path.join(datapath, 'attributes.txt'))

dataset = fmri_dataset(
                samples=os.path.join(datapath, 'bold.nii.gz'),
                targets=attr.targets,
                chunks=attr.chunks,
                mask=os.path.join(datapath, 'mask_vt.nii.gz'),
                #add_fa={'vt_thr_glm': os.path.join(datapath, 'mask_vt.nii.gz')},
                )

poly_detrend(dataset, polyord=1, chunks_attr='chunks')
zscore(dataset, chunks_attr='chunks', dtype='float32')

ds1 = dataset.copy()
ds2 = dataset.copy()

cond1 = 'shoe'
cond2 = 'scissors'
cond3 = 'chair'
cond4 = 'bottle'


ds1 = ds1[np.array([l in [cond1, cond2] for l in dataset.targets], dtype='bool')]
ds2 = ds2[np.array([l in [cond3, cond4] for l in dataset.targets], dtype='bool')]

ds1.sa['task'] = [1 for s in ds1.targets]
ds2.sa['task'] = [2 for s in ds1.targets]

m_cond_1 = ds2.targets == cond3
m_cond_2 = ds2.targets == cond4

ds2.targets[m_cond_1] = cond1
ds2.targets[m_cond_2] = cond2

ds_merged = vstack((ds1,ds2))

ds_merged.a.update(ds2.a)
clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
cv = CrossValidation(clf, NFoldPartitioner(attr='task'))

sl = sphere_searchlight(cv, 3, space = 'voxel_indices')

sl_map = sl(ds_merged)

sl_map.samples *= -1
sl_map.samples +=  1


ni.save(ni.Nifti1Image(sl_map.samples, ds1.a.imghdr.get_base_affine()), 
            '/home/robbis/sl_crossdecoding_haxby__.nii.gz')

clf = LinearCSVMC(C=1, probability=1, enable_ca=['probabilities'])
cv = CrossValidation(clf, HalfPartitioner(attr='chunks'))

sl = sphere_searchlight(cv, 3, space = 'voxel_indices')

sl_map = sl(ds1)
sl_map.samples *= -1
sl_map.samples +=  1

sl_ni = map2nifti(sl_map, imghdr=ds1.a.imghdr)
ni.save(sl_ni, '/home/robbis/sl_crossdecoding_haxby_ds1_.nii.gz')


sl_map = sl(ds2)
sl_map.samples *= -1
sl_map.samples +=  1

sl_ni = map2nifti(sl_map, imghdr=ds2.a.imghdr)
ni.save(sl_ni, '/home/robbis/sl_crossdecoding_haxby_ds2_.nii.gz')