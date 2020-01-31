import numpy as np
from nitime.timeseries import TimeSeries
from nitime.analysis import SeedCorrelationAnalyzer, CorrelationAnalyzer
from sklearn.model_selection import *
from mvpa2.mappers.fx import mean_group_sample
from mvpa_itab.similarity.analysis import SeedAnalyzer
from mvpa_itab.pipeline import SearchlightAnalysisPipeline
from mvpa_itab.pipeline.deprecated.searchlight import LeaveOneSubjectOutSL
from mvpa_itab.conn.connectivity import z_fisher


def get_estimator(estimator):
    estimator_mapper = {
                        'mean': mean_group_sample,
                                                
                        }
    
    return estimator_mapper[estimator]


def get_measure(measure):
    measure_mapper = {
                      'euclidean': euclidean_measure,
                      'correlation' : correlation_measure
                      }
    
    return measure_mapper[measure]
     


def pattern_connectivity(ds, condition, estimator='mean', measure='correlation', cv_attribute='chunks'):
    """
    Function used to calculate how the pattern of different areas
    covariates all-together. 
    
    Params
    ------
    ds: pymvpa dataset 
        The dataset to be used, it must be build using add_fa = {"roi_labels": atlas_nifti}
    
    condition: dictionary
        The name of the sample attibute with values of the labels to be used to 
        esteem the pattern of activity of the specified condition
    
    estimator: string ('mean', 'pca', 'difference')
        How the pattern is calculated in the area.
        
    cv_attribute: string
        How to split the data to calculate the pattern and the correlation.
        
    """
 
    
    # Pattern estimated for a particular condition in whole brain
    pattern_ds = get_pattern(ds, condition, estimator, cv_attribute)
    cross_validation = np.unique(ds.sa[cv_attribute].value)
    
    # Get the connectivity metrics
    measure_fx = get_measure(measure)
    
    cv_results = []
    
    for fold in cross_validation:
        
        # Split train and testing
        mask_test = ds.sa[cv_attribute].value == fold
        mask_train = pattern_ds.sa[cv_attribute].value != fold
        
        seed_ds = pattern_ds[mask_train]
        target_ds = ds[mask_test]
        
        # Get training samples
        mask_task = get_condition_mask(target_ds, condition)
        target_ds = target_ds[mask_task]
        
        roi_correlation = measure_fx(seed_ds, target_ds)
            
        cv_results.append(roi_correlation)
                
    return cv_results



def euclidean_measure(seed_ds, target_ds):
    
    roi_correlation = []
    
    for roi in np.unique(seed_ds.fa.roi_labels)[1:]:
            
        mask_roi = seed_ds.fa.roi_labels == roi
        
        seed_ts = TimeSeries(seed_ds[:, mask_roi], sampling_interval=1.)        
        target_ts = TimeSeries(target_ds[:, mask_roi], sampling_interval=1.)      
        
        S = SeedAnalyzer(seed_ts, target_ts)
        
        roi_correlation.append(S.measure)

    return roi_correlation



def correlation_measure(seed_ds, target_ds):
    
    roi_correlation = []
    rois = [k for k in seed_ds.fa.keys() if k != 'voxel_indices']
    roi_values = seed_ds.fa[rois[0]].value
    
    for roi in np.unique(roi_values)[1:]:
            
        mask_roi = roi_values == roi
        
        seed_ts = TimeSeries(seed_ds[:, mask_roi], sampling_interval=1.)        
        target_ts = TimeSeries(target_ds[:, mask_roi], sampling_interval=1.)      
        
        S = SeedCorrelationAnalyzer(seed_ts, target_ts)
        
        roi_correlation.append(S.corrcoef)

    return roi_correlation
    



def speed_connectivity(timecourses):
    
    ts = TimeSeries(timecourses, sampling_interval=1.)
    C = CorrelationAnalyzer(ts)
    
    return z_fisher(C.corrcoef)
    
    

def subject_pattern_connectivity(timecourses):
    """
    timecourses is a n_subjects list
    the list is composed by a n_roi x n_timecourses
    """
    
    matrices = []
    for t in timecourses:
        matrices.append(speed_connectivity(t))
        
    return matrices



def get_condition_mask(ds, condition):
    """
    This function gives the mask to delete volumes in the ds
    with label equal to condition
    """
    mask = np.zeros_like(ds.samples[:,0], dtype=np.bool)
    
    for key, values in condition.iteritems():
        for c in values:
            mask = np.logical_or(mask, ds.sa[key].value == c)
            
    return mask




def get_pattern(ds, condition, estimator, cv_attribute):
    """
    This function is used to estimate the pattern of interest
    Using the labels equal to condition, the attibute in which 
    use a mapper to estimate the pattern.
    """
    
    estimator_fx = get_estimator(estimator)
    mask = get_condition_mask(ds, condition)
    ds_ = ds[mask].get_mapped(estimator_fx([cv_attribute]))
    
    return ds_
        
"""
conf = {
        'condition_names': ['evidence', 'task'],
        'configuration_file': 'ofp.conf',
        'data_type': 'OFP',
        'evidence': 3,
        'mask_area': 'glm_atlas_mask_333.nii',
        'n_balanced_ds': 1,
        'n_folds': 3,
        'normalization': 'both',
        'partecipants': 'subjects.csv',
        'path': '/home/carlos/mount/wkpsy01/carlo_ofp/',
        'project': 'carlo_ofp',
        'split_attr': 'subject_ofp',
        'task': 'decision'}

path = "/home/carlos/mount/wkpsy01/carlo_ofp/1_single_ROIs"
mask_list = [#'cluster_level_effect_p0-0033_q0-05_mask.nii',
             'cluster_omnibus_p0-0014_q0-01_mask.nii',
             #'cluster_level_effect_p0-0033_q0-05_mask.nii',
             'cluster_conjunction_level-0-003_omnibus-0-001.nii.gz']

result_dict = {}
for mask in mask_list:
    mask_img = ni.load(os.path.join(path, mask))
    sl_analysis = LeaveOneSubjectOutSL(roi_labels=mask_img, **conf)
    ds = sl_analysis.load_dataset()
    r = []
    for s in np.unique(ds.sa.subject):
        subject_ds = ds[ds.sa.subject == s]
        subject_ds = subject_ds[np.logical_or(subject_ds.sa.decision == 'F', subject_ds.sa.decision == 'L')]
        r.append(zscore(trajectory_connectivity(subject_ds), axis=1))
    #r = pattern_connectivity(ds, condition={'decision': ['L','F']}, cv_attribute='subject')
    m = subject_pattern_connectivity(r)
    #key = "%s_%s" % (mask, condition)
    key = "%s" % (mask)
    result_dict[key] = [r, m]
"""      
        
        





