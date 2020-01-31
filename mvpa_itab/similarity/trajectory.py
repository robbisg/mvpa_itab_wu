import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
import nibabel as ni
import os
from mvpa_itab.similarity.connectivity import subject_pattern_connectivity,\
    speed_connectivity
from mvpa_itab.main_wu import slice_dataset



def trajectory_connectivity(ds, conditions={'subject': ['name'], 
                                            'decision': ['F', 'I']}):   
    
    parcellation_list = [m for m in ds.fa.keys() if m != 'voxel_indices']
    results = {m:[] for m in parcellation_list}
    results['full_brain'] = []
    
    for parcel in parcellation_list:

        # I can use slice_dataset
        subject_ds = slice_dataset(ds, conditions)
        
        roi_values = subject_ds.fa[parcel].value
        
        tc = zscore(get_speed_timecourse(subject_ds, roi_values), axis=1)
        
        m = speed_connectivity(tc)
        
        brain_values = np.ones(subject_ds.shape[1])
        btc = zscore(get_speed_timecourse(subject_ds, brain_values), axis=1)
        
        results[parcel].append([tc, m])
        results['full_brain'].append(btc.squeeze())
            
        
    return results






def get_speed_timecourse(ds, roi_mask):
    
    roi_trajectory = []
     
    roi_unique = [v for v in np.unique(roi_mask) if v != 0]
     
     
    for roi in roi_unique:
            
        mask_roi = roi_mask == roi
            
        ds_ = ds[:, mask_roi]
        
        trajectory = [euclidean(ds_.samples[i+1], ds_.samples[i]) for i in range(ds_.shape[0]-1)]
        roi_trajectory.append(np.array(trajectory))
    
    # return a n_rois x n_timepoints array
    return np.array(roi_trajectory)



def get_partial_correlation(subject_tc, subject_brain_tc):
    
    partial_corr = []
    n_subjects = len(subject_tc)
    
    for i in range(n_subjects):
        X = np.array(subject_tc[i])
        Z = np.array(subject_brain_tc[i])[np.newaxis, :]
        
        pc = partial_correlation(X, Z)
        
        partial_corr.append(pc)
    
    return partial_corr
    



def partial_correlation(X, Z):
    """
    Returns the partial correlation coefficients between 
    elements of X controlling for the elements in Z.
    """
 
     
    X = np.asarray(X).transpose()
    Z = np.asarray(Z).transpose()
    n = X.shape[1]
 
    partial_corr = np.zeros((n,n), dtype=np.float)
    
    for i in range(n):
        partial_corr[i,i] = 0
        for j in range(i+1,n):
            beta_i = np.linalg.lstsq(Z, X[:,j])[0]
            beta_j = np.linalg.lstsq(Z, X[:,i])[0]
 
            res_j = X[:,j] - Z.dot(beta_i)
            res_i = X[:,i] - Z.dot(beta_j)
 
            corr = np.corrcoef(res_i, res_j)
 
            partial_corr[i,j] = corr.item(0,1)
            partial_corr[j,i] = corr.item(0,1)
 
    return partial_corr
        
    