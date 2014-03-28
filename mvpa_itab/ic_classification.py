import os
from scipy.io import loadmat
from sklearn import cross_validation
from sklearn.preprocessing import scale
from numpy.lib.scimath import log10
import numpy as np
import nibabel as ni
from mvpa.datasets.mri import fmri_dataset
from nitime.timeseries import TimeSeries



path = '/media/DATA/fmri/gianni_param/'
fn = '130429valcal0003_RestI2_prepro.mat'
data = loadmat(os.path.join(path, fn))

param = data['Param_val'].T
param[:,3] = log10(param[:,3])

param_scale = scale(param)

y = np.zeros(param_scale.shape[0])
y[data['art']-1] = 1


X_train, X_test, y_train, y_test = cross_validation.train_test_split(param_scale,
                                                                     y, 
                                                                     test_size=0.4, 
                                                                     random_state=0)



###################### Piero #########################################
mask = '/media/DATA/fmri/ica_classification/ICA/mask.nii.gz'

img_s00 = ni.load('/media/DATA/fmri/ica_classification/shuffled_IC/shuffled_dr_stage2_subject00000_Z.nii.gz')
img_s01 = ni.load('/media/DATA/fmri/ica_classification/shuffled_IC/shuffled_dr_stage2_subject00001_Z.nii.gz')
img_s02 = ni.load('/media/DATA/fmri/ica_classification/shuffled_IC/shuffled_dr_stage2_subject00002_Z.nii.gz')
img_s03 = ni.load('/media/DATA/fmri/ica_classification/shuffled_IC/shuffled_dr_stage2_subject00003_Z.nii.gz')

img_s00 = ni.load('/media/DATA/fmri/ica_classification/ICA/dr_stage2_subject00000_Z.nii.gz')
img_s01 = ni.load('/media/DATA/fmri/ica_classification/ICA/dr_stage2_subject00001_Z.nii.gz')
img_s02 = ni.load('/media/DATA/fmri/ica_classification/ICA/dr_stage2_subject00002_Z.nii.gz')
img_s03 = ni.load('/media/DATA/fmri/ica_classification/ICA/dr_stage2_subject00003_Z.nii.gz')


ds0 = fmri_dataset(img_s00, mask=mask)
ds1 = fmri_dataset(img_s01, mask=mask)
ds2 = fmri_dataset(img_s02, mask=mask)
ds3 = fmri_dataset(img_s03, mask=mask)

d0 = fmri_dataset(img_00, mask=mask)
d1 = fmri_dataset(img_01, mask=mask)
d2 = fmri_dataset(img_02, mask=mask)
d3 = fmri_dataset(img_03, mask=mask)

ts0 = TimeSeries(ds0.samples, sampling_interval=1)
ts1 = TimeSeries(ds1.samples, sampling_interval=1)













