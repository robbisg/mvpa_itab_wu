from scipy.io import savemat
import numpy as np
import os
from ..connectivity import load_matrices, z_fisher, plot_matrix
from nitime.analysis import SeedCorrelationAnalyzer
from nitime.timeseries import TimeSeries
from scipy.io import loadmat
from scipy.stats import ttest_ind
from scipy.stats import zscore as sscore
from mvpa_itab.lib_io import load_dataset
from mvpa_itab.conn.io import load_fcmri_dataset

path = '/media/robbis/DATA/fmri/monks/0_results/'

results_dir = os.listdir(path)
 
results_dir = [r for r in results_dir if r.find('connectivity') != -1]
roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str)

subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)


for r in results_dir:
    
    fname = os.path.join(path, r, 'zcorrelation_matrix.mat')
    fdata = loadmat(fname)
    
    data = fdata['z_matrix']
    
    nrun = data.shape[2]
    print nrun
    ds = load_fcmri_dataset(data, 
                            subjects.T[0], 
                            ['Samatha', 'Vipassana'], 
                            fdata['groups'], 
                            fdata['level'].squeeze(),
                            n_run=nrun)
    
    cv = CrossValidation(LinearCSVMC(C=1), 
                         NFoldPartitioner(cvtype=2), 
                         enable_ca=['stats'])
    
    
    ds.samples = sscore(ds.samples, axis=1)
    
    #zscore(ds, chunks_attr=None)
    
    ds.targets = ds.sa.group
    ds = ds[~np.logical_or(ds.sa.level == 300, ds.sa.level == 600)]
        
    #for g in np.unique(fdata['groups']):
    for g in ['Samatha','Vipassana']:
        print '---------------------------'
        print r+' ----- '+g
        err = cv(ds[ds.sa.meditation == g])
        print cv.ca.stats
        
        
    '''
    print '--------------- '+r
    err = cv(ds)
    print cv.ca.stats
    '''