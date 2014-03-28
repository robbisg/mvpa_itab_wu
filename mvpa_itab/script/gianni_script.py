from scipy.io import loadmat
from scipy.stats.mstats import *


data_ts = loadmat('/media/DATA/fmri/gianni_param/130429valcal0002_RestI1_prepro.mat')
data_tr = loadmat('/media/DATA/fmri/gianni_param/130429valcal0003_RestI2_prepro.mat')

param_ts = data_ts['Param_val'].T
param_tr = data_tr['Param_val'].T

param_ts = zscore(param_ts, axis=0)

