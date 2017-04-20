from mvpa_itab.test_wu import _test_spatial
import os
import numpy as np



def get_configuration(default_config):
    
    import itertools
    
    values = default_config.values()
    keys = default_config.keys()
    
    configurations = []
    for v in itertools.product(*values):
        conf = dict(zip(keys, v))
        
        configurations.append(conf)
        
    return configurations        
    



def analysis(path, subject_file='subjects.csv', **kwargs):
    
    roi_path = '1_single_ROIs'
    rois = os.listdir(os.path.join(path,roi_path))
    rois = [roi for roi in rois if roi.find('nii.gz') != -1]

    
    
    default_config = {
                      #'targets': ['decision', 'memory'],
                      'mask_label__evidence': [1, 3, 5],
                      'mask_area': rois[:2],                 
                      'normalization': ['feature','sample','both'],
                      #'balance__count': [50],
                      'chunks':['adaptive', 5]
                      }
    
    default_config.update(kwargs)
    
    configuration_generator = get_configuration(default_config)
    
    subjects = np.loadtxt(os.path.join(path,'subjects.csv'),
                          dtype=np.str, 
                          delimiter=',')
    subjects_ = subjects[1:3,0]
    
    
    config = {
              'target':'decision',
              'balancer__count':50,
              'chunk_number':5,
              'normalization':'feature'
              }
    
    
    for conf in configuration_generator:

        config.update(conf)

        analysis_type = "%s_%s_%s" % (conf['target'],
                                      conf['mask_label__evidence'],
                                      conf['normalization'])
        
        res = _test_spatial(path, subjects_, 'memory.conf', 'BETA_MVPA', analysis_type=analysis_type, **conf)
        
    
    
    