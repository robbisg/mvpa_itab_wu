import nibabel as ni
import os
import numpy as np
from nitime.timeseries import TimeSeries
from nitime.analysis.correlation import SeedCorrelationAnalyzer
from mvpa_itab.similarity.analysis import SeedAnalyzer
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing.label import LabelEncoder
from scipy.stats.stats import ttest_1samp
from timeit import itertools




def get_condition_mask(condition_list, fields, conditions, conditions_dict):
    
    mask = np.ones_like(condition_list.T[0], dtype=np.bool)
    
    for f in fields:
        condition = conditions[f]
        
        c_mask = np.zeros_like(mask)
        for c in condition:
            m = condition_list.T[conditions_dict[f]] == c
            c_mask = np.logical_or(c_mask, m)
    
    
        mask = np.logical_and(mask, c_mask)
        
    
    return mask




def beta_series(brain_ts, **kwargs):
    
    seed_fname = kwargs['seed_fname'][0]
    img_data = kwargs['img_data']
    time_mask = kwargs['time_mask']
    
    seed = ni.load(seed_fname).get_data().squeeze()
    seed_data = img_data[np.nonzero(seed)][:,time_mask]
    seed_data = seed_data.mean(0)[np.newaxis,:]
    seed_ts = TimeSeries(seed_data, sampling_interval=1.)
    
    S = SeedCorrelationAnalyzer(seed_ts, brain_ts)
    measure = S.corrcoef
    
    return measure




def roc(brain_ts, **kwargs):
    
    conditions = kwargs['conditions']
    
    #label_data = LabelEncoder().fit_transform(conditions_)
    label_data = np.zeros_like(conditions, dtype=np.int)
    label_data[conditions == "C"] = 1
    seed_ts = TimeSeries(label_data[np.newaxis,:], sampling_interval=1.)

    S = SeedAnalyzer(seed_ts, brain_ts, roc_auc_score)   
    measure = S.measure
    
    return measure


def difference(brain_ts, **kwargs):
    
    seed_list = kwargs['seed_fname']
    img_data = kwargs['img_data']
    time_mask = kwargs['time_mask']
    
    multiseed_data = []
    
    for s in seed_list:
        seed = ni.load(s).get_data().squeeze()
        seed_data = img_data[np.nonzero(seed)][:,time_mask]
        seed_data = seed_data.mean(0)[np.newaxis,:]
        multiseed_data.append(seed_data)
        
    multiseed_data = np.vstack(multiseed_data)
    diff_ = np.abs(np.diff(multiseed_data, axis=0)[0])
    
    seed_ts = TimeSeries(diff_, sampling_interval=1.)
    S = SeedCorrelationAnalyzer(seed_ts, brain_ts)
    measure = S.corrcoef
    
    return measure




def analysis(subjects, **kwargs):
    
    default_config = {
                      "path":'/root/robbis/fmri/carlo_ofp/',
                      "file_dir":"analysis_SEP/DE_ASS_noHP/SINGLE_TRIAL_MAGS_voxelwise/",
                      "fields": ["encoding", "level", "response"],
                      "conditions":{
                                    "encoding": ["F", "L"],
                                    "response": ["C"],
                                    "level": ["1","2","3"]
                                      },
                      "condition_dir_dict":{
                                            "encoding":0,
                                            "level":1,
                                            "response":2
                                            },
                      "cond_file":"%s_condition_list.txt",
                      "filename":"residuals.nii.gz",
                      "brain_mask":"glm_atlas_mask_333.nii",
                      "mask_dir":"1_single_ROIs",
                      "seed_mask": ["L_FFA.nii", "L_PPA.nii"],
                      "analysis":"beta"
                      }
    
    
    default_config.update(kwargs)
    
    mapper = {
              "beta": beta_series,
              "roc": roc,
              "abs_difference": difference, 
              }
    
    method = mapper[default_config["analysis"]]
    method_name = default_config["analysis"]
    
    path = default_config['path']
    filename = default_config['filename']
    cond_file = default_config['cond_file']
    brain_mask = default_config['brain_mask']
    seed_mask = default_config['seed_mask']
    mask_dir = default_config['mask_dir']
    file_dir = default_config["file_dir"]
    
    fields = default_config["fields"]
    conditions = default_config["conditions"]
    conditions_dict = default_config["condition_dir_dict"]
    
    
    mask_fname = os.path.join(path, mask_dir, brain_mask)
    seed_fname = [os.path.join(path, mask_dir, s) for s in seed_mask]
    default_config['seed_fname'] = seed_fname
    
    mask = ni.load(mask_fname)
    mask_data = mask.get_data().squeeze()
    
    total_results = []
    
    
    for s in subjects:

        condition_list = np.genfromtxt(os.path.join(path, s, file_dir, cond_file %(s)), 
                                       dtype=np.str, 
                                       delimiter=',')
        
        
        condition_mask = get_condition_mask(condition_list, 
                                            fields, 
                                            conditions, 
                                            conditions_dict)
        
        conditions_ = condition_list[condition_mask].T[-1]
        
        
        img_fname = os.path.join(path, s, file_dir, filename)
        
        time_mask = condition_mask
        default_config['time_mask'] = time_mask
        
        # Load brain
        img = ni.load(img_fname)
        img_data = img.get_data()
        default_config['img_data'] = img_data
        
        brain_img = img_data[np.nonzero(mask_data)][:,time_mask]
        brain_ts = TimeSeries(brain_img, sampling_interval=1.)    
            
            
        measure = method(brain_ts, **default_config)
        
        ### Save file
        result = np.zeros(mask.shape)
        result[...,0][np.nonzero(mask_data)] = measure
        
        condition_unique = np.unique([''.join(c) for c in condition_list[condition_mask]])
        
        fname = "%s_%s_%s_%s.nii.gz" % (method_name, '-'.join(seed_mask), '_'.join(condition_unique), s)
        ni.save(ni.Nifti1Image(result, mask.affine), os.path.join(path, "0_results", fname))
        
        total_results.append(result)
        
    total_results = np.concatenate(total_results, axis=3)
    
    fname = "0_%s_%s_%s_%s.nii.gz" % (method_name, '-'.join(seed_mask), '_'.join(condition_unique), "total")
    ni.save(ni.Nifti1Image(total_results, mask.affine), os.path.join(path, "0_results", fname))
    
    fname = "0_%s_%s_%s_%s.nii.gz" % (method_name, '-'.join(seed_mask), '_'.join(condition_unique), "total_avg")
    ni.save(ni.Nifti1Image(total_results.mean(3), mask.affine), os.path.join(path, "0_results", fname))
  
       

def main(subjects):
    import itertools
    encoding = [["F", "L"], "F", "L"]
    level = [["1","2","3"]]
    masks_ = [["L_PPA_enc.nii",  "L_HIPP_enc.nii"], 
              ["L_PPA.nii", "L_FFA.nii"]]
    
    conditions_dict = {
                        "encoding": ["F", "L"],
                        "response": ["C"],
                        "level": ["1"]
                          }
    
    p = itertools.product(masks_, encoding, level)
    
    for opt in p:
        conditions_dict["encoding"] = opt[1]
        conditions_dict["level"] = opt[2]
        mask_name = opt[0]
        
        print opt
        
        analysis(subjects, seed_mask=mask_name, analysis="abs_difference", conditions=conditions_dict)
        


def roc_stats():
    path = "/root/robbis/fmri/carlo_ofp/0_results/"
    import glob
    ##### ROC t-test ####
    
    rocfile = glob.glob(os.path.join(path, "0_roc*total.nii.gz"))
    rocimg = ni.load(rocfile[0])
    
    t, p = ttest_1samp(rocimg.get_data(), 0.5, axis=3)
    q = np.zeros_like(p)
    q[np.logical_not(p == 0)] = 1 - p[np.logical_not(p == 0)]
    t[np.isinf(t)] = 0
    
    p[np.isnan(p)] = 0
    
    
    ni.save(ni.Nifti1Image(q, rocimg.affine), os.path.join(path, "0_roc_ttest_q.nii.gz"))
    ni.save(ni.Nifti1Image(p, rocimg.affine), os.path.join(path, "0_roc_ttest_p.nii.gz"))
    ni.save(ni.Nifti1Image(t, rocimg.affine), os.path.join(path, "0_roc_ttest_t.nii.gz"))
    
    

def beta_stats(mask_area, test_field="encoding", test=["F", "L"], **kwargs):
    
    
    default_config = {
                      "encoding": ["F", "L"],
                      "level": ["3"],
                      "response" :["C"],
                      }

    condition_dir_dict = {
                          "encoding":0,
                          "level":1,
                          "response":2
                          }
    
    encoding = default_config["encoding"]
    level = default_config["level"]
    response = default_config["response"]
    
    items = itertools.product(encoding, level, response)
    
    conditions = ["".join(v) for v in items]
    
    path = "/home/robbis/mount/wkpsy01/fmri/carlo_ofp/0_results/"
    import glob
    
    pattern = "0_beta_%s_%s_total.nii.gz"
    
    maps = dict()
    
    for condition in default_config[test_field]:
        condition_pattern = [v for v in conditions if v.find(condition)!=-1]
        print condition_pattern
        condition_pattern.sort()
        rocfile = glob.glob(os.path.join(path, pattern % (mask_area, "_".join(condition_pattern))))
        print rocfile
        map[condition] = ni.load(rocfile[0])
        
    
    from scipy.stats import ttest_ind
    
    t, p = ttest_ind(map[test[0]].get_data(), map[test[1]].get_data(), axis=3)
    
    q = np.zeros_like(p)
    q[np.logical_not(p == 0)] = 1 - p[np.logical_not(p == 0)]
    t[np.isinf(t)] = 0
    
    p[np.isnan(p)] = 0
    
    def get_image(img):
        return ni.Nifti1Image(img, map[test[0]].affine)
    
    
    def get_name(mask_area, test, map_type):
        pattern = "0_beta_ttest_%s_%s_%s_3.nii.gz" % (mask_area, "_".join(test), map_type)
        return pattern
        
    
    ni.save(get_image(q), os.path.join(path, get_name(mask_area, test, "q")))
    ni.save(get_image(p), os.path.join(path, get_name(mask_area, test, "p")))
    ni.save(get_image(t), os.path.join(path, get_name(mask_area, test, "t")))
    
        
        
