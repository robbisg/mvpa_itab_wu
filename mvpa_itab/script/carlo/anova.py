import numpy as np
import nibabel as ni
import os
from mvpa_itab.stats.anova import design_matrix, build_contrast, anova_ftest
from mvpa_itab.io.carlo_utils import load_total_subjects
import itertools
     

def analysis(path, 
             subjects, 
             file_pattern,
             factor_index=2,
             test="zero",
             **kwargs):
    
    default_conf = {
                    'subject': subjects,
                    'experiment':['memory', 'evidence'],
                    'level': [1,3,5],
                    'ds_num' : [1,2,3,4,5]  ,
                    'ds_type':['BETA']                  
                    }
    
    # path = '/home/robbis/mount/fmri/memory/0_results/sl_k_3/'
    
    default_conf.update(kwargs)
    
    #file_pattern="%s_%s_%s_MVPA_evidence_%s_balance_ds_%s_radius_3_searchlight_total_map_mean_demenead.nii.gz"

    data, labels, affine = load_total_subjects(path,
                                               subjects,
                                               file_pattern=file_pattern,
                                               **kwargs)
    
    mask_ = np.abs(data.mean(3)) > 0.01
    mask_index = np.array(np.nonzero(mask_)).T
    
    
    X, factor_labels, factor_num = design_matrix(labels)
        
    contrast, const_terms = build_contrast(factor_num, factor_index, test)
    
    map_ = anova_ftest(data, X, contrast, const_terms, mask_index)
    
    str_join = '_'
    filetype = file_pattern[file_pattern.find('map'):file_pattern.find('.nii.')]
    save_fname = "anova_test_%s_%s_%s_%s.nii.gz" % (
                                                    filetype,
                                                    str_join.join(default_conf['experiment']),
                                                    default_conf.keys()[factor_index],
                                                    test                                             
                                                    )

    ni.save(ni.Nifti1Image(map_, affine), 
            os.path.join(path, save_fname))
    
    return map_
    
################################################

experiments = [#'memory', 
               'decision']
stable_file_pattern = "%s_%s_%s_MVPA_evidence_%s_balance_ds_%s_radius_3_searchlight_total_map"

file_patterns = [
                 #stable_file_pattern+".nii.gz",
                 #stable_file_pattern+"_mean_demenead.nii.gz",
                 stable_file_pattern+"_demeaned.nii.gz",
                 #stable_file_pattern+"_mean.nii.gz",                 
                 ]


tests = [#"zero", 
         "all"]

confs = itertools.product(experiments, file_patterns, tests)
path = '/home/robbis/mount/fmri/memory/0_results/sl_k_3/'

maps_ = []

for elements in confs:
    print elements
    map_ = analysis(path, 
                     subjects, 
                     file_pattern=elements[1],
                     test=elements[2],
                     experiment=[elements[0]],
                     ds_num=[1,2,3])
    
    maps_.append(map_)
    
    


    

