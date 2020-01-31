from mvpa_itab.test_wu import _test_spatial
from mvpa_itab.io.base import load_subject_file, load_dataset,\
    read_configuration
from mvpa_itab.utils import enable_logging
import os.path as p
import numpy as np
from mvpa_itab.main_wu import detrend_dataset, normalize_dataset,\
    change_target, slice_dataset, balance_dataset, spatial, searchlight
import mvpa_itab.results as rs
import nibabel as ni


#enable_logging()

path = '/home/carlos/fmri/carlo_ofp'
conf_file = 'ofp.conf'
task = 'RESIDUALS'
subject_file = 'subjects.csv'
frames = [0]#[1,2,3,4,5,6]
mask = ni.load(p.join(path, '1_single_ROIs', 'cluster_conjunction_level-0-003_omnibus-0-001.nii.gz'))

subjects, extra_sa = load_subject_file(p.join(path, 'subjects.csv'))

target = 'decision'
selected_variables = {
                      target: ['L', 'F'],
                      }

evidences = [1,2,3]

# Setup result utilities
conf = read_configuration(path, conf_file, task)
conf['analysis_type'] = 'searchlight'
conf['analysis_task'] = 'temporal_residual'
summarizers = [rs.SearchlightSummarizer()]
#savers = [rs.DecodingSaver(fields=['classifier', 'stats'])]
savers = [rs.SearchlightSaver()]
result = rs.ResultsCollection(conf, path, summarizers)

for subj in subjects[:1]:
    
    conf = read_configuration(path, conf_file, task)
    
    # Load dataset
    ds_orig = load_dataset(path, subj, task, roi_labels={'conjunction': mask}, **conf)
    
    # Change the target
    ds_orig = change_target(ds_orig, target) 
    
    # Process dataset
    ds_orig = detrend_dataset(ds_orig, task, **conf)
    
    # Balance dataset
    balancer = balance_dataset(balancer__count=5, **conf)
    
    for ev in evidences:       
        for slice_condition in frames:
            
            selected_variables.update({
                                       'frame':[1,2,3,4,5],
                                       'evidence':[ev]})
            
            print selected_variables
            ds = slice_dataset(ds_orig, selected_variables)
            
            
            for i, ds_bal in enumerate(balancer.generate(ds)):
                
                ds_ = normalize_dataset(ds_bal, normalization='both', chunk_number=None, **conf)
                #ds_ = ds_bal
                
                """
                for roi in ds_.fa.keys():
                    
                    if roi == 'voxel_indices':
                        continue
            
                    for label in np.unique(ds_.fa[roi])[1:]:
                        
                        # Build a function
                        ds_roi = ds_[:, ds_.fa[roi] == label]
                        
                        ds_roi = normalize_dataset(ds_roi, normalization='both', chunk_number=7, **conf)
                                                
                        r = spatial(ds_roi, **conf)
                        r = searchlight
                
                        subj_ = "%s_roi_%s_ev_%s_ds_%s_tf_%s" %(subj,
                                                                str(int(label)),
                                                                str(ev), 
                                                                str(i+1), 
                                                                str(slice_condition))
                        #print selected_variables
                        print subj_
                        
                        subj_result = rs.SubjectResult(subj_, r, savers)
            
                        result.add(subj_result)
                        
                """
                
                r = searchlight(ds_)
                subj_ = "%s_roi_%s_ev_%s_ds_%s_tf_%s" %(subj,
                                                        str(int(0)),
                                                        str(ev), 
                                                        str(i+1), 
                                                        str(slice_condition))
                
                subj_result = rs.SubjectResult(subj_, r, savers)
                result.add(subj_result)
        
    
# Run Analysis
result.summarize()    



    
    
