from __future__ import print_function

import os
import itertools
import numpy as np
import nibabel as ni



def load_group(path, experiment, level):
    
    file_pattern = "sl_group_%s_evidence_%s_demeaned.nii.gz" % (str(experiment), str(level))
    file_path = os.path.join(path, 'group_sl', 'subject_wise', file_pattern)
    
    img = ni.load(file_path)
    
    if len(img.shape) == 3:
        n_volumes = 1
    else:
        n_volumes = img.get_data().shape[-1]
        
    labels = []
    group = 1
    for i in range(n_volumes):
        
        if i>=12:
            group = 2
        
        labels.append([group, experiment, level, i+1])
    
    return img, labels, img.get_affine()




def load_group_total(path, experiment, level=[1,2,3]):
    
    total_data = []
    total_labels = []
    img_list = []
    
    for l in level:
        img, labels, affine = load_group(path, experiment, l)

        total_data.append(img.get_data())
        total_labels.append(labels)
        img_list.append(img)
        
    total_data = np.rollaxis(np.array(total_data), 0, 4)
    reshape_ = list(total_data.shape[:3]) + [-1]
        
    return np.array(total_data).reshape(reshape_), np.vstack(total_labels), affine, img_list    




def load_beta_subjects(path, subjects, **kwargs):
    
    default_conf = {
                    'subjects': ['110929angque','110929anngio','111004edolat'],
                    'roi':['L_FFA.nii', 'L_PPA.nii'],
                    'encoding': ["F", "L"],
                    'level' : ["1","2","3"],              
                    }
    
    default_conf.update(kwargs)
    default_conf['subjects'] = subjects
    conf_product = itertools.product(*default_conf.values())
    
    
    total_data = []
    total_labels = []
    img_list = []
    
    key_dict = dict(zip(default_conf.keys(), range(len(default_conf.keys()))))
    print(key_dict)
    print(default_conf)
    
    for elements in conf_product:
        
        subject = elements[key_dict['subjects']]
        roi = elements[key_dict['roi']]
        level = elements[key_dict['level']]
        encoding = elements[key_dict['encoding']]        

        data, labels, affine = load_data_beta(path, subject, roi, encoding, level)

        total_data.append(data.get_data())
        total_labels.append(labels)
        img_list.append(data)
    
    total_data = np.rollaxis(np.array(total_data), 0, 4)
    reshape_ = list(total_data.shape[:3]) + [-1]
        
    return np.array(total_data).reshape(reshape_), np.vstack(total_labels), affine, img_list




def load_subject_cross_validation(path, 
                                  subject, 
                                  experiment, 
                                  level, 
                                  ds_num=1, 
                                  ds_type="BETA",
                                  **kwargs):
    
    
    """
    Loads the 
    
    """
    """
    path = "/media/robbis/DATA/fmri/memory/0_results/balanced_analysis/local/"
    subject = "120112jaclau"
    experiment = "memory"
    level = str(3)
    """
    
    default_conf = {'file_pattern': \
                    "%s_%s_%s_MVPA_evidence_%s_balance_ds_%s_radius_3_searchlight_total_map.nii.gz",
                    'dir_pattern': "%s_%s_%s_MVPA_evidence_%s_balance_ds_%s"
                    }
    
    default_conf.update(kwargs)
    
    file_pattern = default_conf['file_pattern']
    dir_pattern = default_conf['dir_pattern']
    
    file_path = os.path.join(path,
                             dir_pattern % (subject, experiment, ds_type, level, str(ds_num)),
                             file_pattern % (subject, experiment, ds_type, level, str(ds_num)),
                             )
    
    
    img = ni.load(file_path)
    
    if len(img.shape) == 3:
        n_volumes = 1
    else:
        n_volumes = img.get_data().shape[-1]
    
    
    labels = []
    for i in range(n_volumes):
        labels.append([subject, experiment, level, ds_num, i+1])
    
    return img, labels, img.get_affine()



def load_data_beta(path, subject, roi, encoding, level, **kwargs):
    
    
    default_conf = {'file_pattern': \
                    "beta_%s_%s%sC_%s.nii.gz",
                    'dir_pattern': ""
                    }
    
    default_conf.update(kwargs)
    
    file_pattern = default_conf['file_pattern']
    dir_pattern = default_conf['dir_pattern']
    
    file_path = os.path.join(path,
                             dir_pattern,
                             file_pattern % (roi, encoding, level, subject),
                             )


    img = ni.load(file_path)
    
    if len(img.shape) == 3:
        n_volumes = 1
    else:
        n_volumes = img.get_data().shape[-1]
    
    
    labels = []
    for i in range(n_volumes):
        labels.append([subject, roi, encoding, level])
    
    return img, labels, img.get_affine()



def load_total_subjects(path, subject, **kwargs):
    
    default_conf = {
                    'subject': ['110929angque','110929anngio','111004edolat'],
                    'experiment':['memory', 'decision'],
                    'level': [1,3,5],
                    'ds_num' : [1,2,3,4,5]  ,
                    'ds_type':['BETA'],
                    'file_pattern': \
                    "%s_%s_%s_MVPA_evidence_%s_balance_ds_%s_radius_3_searchlight_total_map.nii.gz"               
                    }
    
    default_conf.update(kwargs)
    default_conf['subject'] = subject
    file_pattern = default_conf.pop("file_pattern")
    print(file_pattern)
    conf_product = itertools.product(*default_conf.values())
    
    
    total_data = []
    total_labels = []
    img_list = []
    
    key_dict = dict(zip(default_conf.keys(), range(len(default_conf.keys()))))
    print(key_dict)
    print(default_conf)
    
    for elements in conf_product:
        
        subject = elements[key_dict['subject']]
        experiment = elements[key_dict['experiment']]
        level = elements[key_dict['level']]
        ds_num = elements[key_dict['ds_num']]
        ds_type = elements[key_dict['ds_type']]
        
        if experiment == 'memory' and ds_num > 1:
            continue
        #print elements
        data, labels, affine = load_subject_cross_validation(path, 
                                                         subject, 
                                                         experiment, 
                                                         level, 
                                                         ds_num, 
                                                         ds_type,
                                                         file_pattern=file_pattern)
        #total_data.append(data)
        total_data.append(data.get_data())
        total_labels.append(labels)
        img_list.append(data)
    
    total_data = np.rollaxis(np.array(total_data), 0, 4)
    reshape_ = list(total_data.shape[:3]) + [-1]
        
    return np.array(total_data).reshape(reshape_), np.vstack(total_labels), affine, img_list
    
    
    

def remove_mean(**kwargs):
    
    default_conf = {
                    'subject': ['110929angque','110929anngio','111004edolat'],
                    'experiment':['memory', 'decision'],
                    'level': [1,3,5],
                    'ds_num' : [1,2,3,4,5]  ,
                    'ds_type':['BETA'], 
                                 
                    'file_pattern': \
                    "%s_%s_%s_MVPA_evidence_%s_balance_ds_%s_radius_3_searchlight_total_map.nii.gz",
                    'dir_pattern': "%s_%s_%s_MVPA_evidence_%s_balance_ds_%s" ,
                    "path": "/home/robbis/mount/fmri/memory/0_results/sl_k_3/",
                    "mask_dir": "/home/robbis/mount/fmri/data7/Carlo_MDM/"
                    }
    
    from nilearn.masking import apply_mask, unmask
    

    default_conf.update(kwargs)
    
    dir_pattern = default_conf.pop("dir_pattern")
    file_pattern = default_conf.pop("file_pattern")
    path = default_conf.pop("path")
    mask_dir = default_conf.pop("mask_dir")
    
    key_dict = dict(zip(default_conf.keys(), range(len(default_conf.keys()))))
    conf_product = itertools.product(*default_conf.values())
    
    
    for elements in conf_product:
        
        subject = elements[key_dict['subject']]
        experiment = elements[key_dict['experiment']]
        level = elements[key_dict['level']]
        ds_num = elements[key_dict['ds_num']]
        ds_type = elements[key_dict['ds_type']]
        
        if experiment == 'memory' and ds_num > 1:
            continue
        
        file_path = os.path.join(path,
                                 dir_pattern % (subject, 
                                                experiment, 
                                                ds_type, 
                                                level, 
                                                str(ds_num)),
                                 file_pattern % (subject, 
                                                 experiment, 
                                                 ds_type, 
                                                 level, 
                                                 str(ds_num)),
                                 )        
        
        
        img = ni.load(file_path)
        
        mask_path = os.path.join(mask_dir,
                                 subject,
                                 "RESIDUALS_MVPA",
                                 "brain_mask.nii.gz",
                                 )
        
        mask_img = ni.load(mask_path)
        
        masked_data = apply_mask(img, mask_img)
        
        mean_ = masked_data.mean()
        demeaned_data = masked_data - mean_
        
        demeaned_img = unmask(demeaned_data, mask_img)
        
        save_pattern = file_pattern % (subject, 
                                       experiment, 
                                       ds_type, 
                                       level, 
                                       str(ds_num)
                                       )
        
        save_pattern_ = save_pattern[:-7]+"_demeaned.nii.gz"
                                 
        
        save_path = os.path.join(path,
                                 dir_pattern % (subject, 
                                                experiment, 
                                                ds_type, 
                                                level, 
                                                str(ds_num)),
                                 save_pattern_
                                 )
                                 
        
        print subject, level, experiment, ds_num, mean_
        
        ni.save(demeaned_img, save_path)
        
        mean_img = unmask(masked_data.mean(0), mask_img)
    
        save_pattern_ = save_pattern[:-7]+"_mean.nii.gz"
        save_path = os.path.join(path,
                                 dir_pattern % (subject, 
                                                experiment, 
                                                ds_type, 
                                                level, 
                                                str(ds_num)),
                                 save_pattern_
                                 )
        ni.save(mean_img, save_path)
        
        demeaned_mean = masked_data.mean(0) - masked_data.mean(0).mean()
        demeaned_mean_img = unmask(demeaned_mean, mask_img)
        
        save_pattern_ = save_pattern[:-7]+"_mean_demenead.nii.gz"
        save_path = os.path.join(path,
                                 dir_pattern % (subject, 
                                                experiment, 
                                                ds_type, 
                                                level, 
                                                str(ds_num)),
                                 save_pattern_
                                 )
        ni.save(demeaned_mean_img, save_path)