##############è Convert ######################
from mvpa_itab.io.base import load_dataset, read_configuration
from mvpa_itab.main_wu import preprocess_dataset
import nibabel as ni
import mvpa_itab.results as rs
from mvpa_itab.wrapper.sklearn import SKLCrossValidation
from sklearn.cross_validation import StratifiedKFold

from mvpa2.suite import *
from mvpa_itab.pipeline.searchlight import SearchlightAnalysis

#path = '/media/robbis/DATA/fmri/memory/'
path = '/home/robbis/fmri/memory/'
#path = '/DATA11/roberto/fmri/memory/'


from mvpa_itab.script.carlo import subjects, tasks, evidences


path_results = os.path.join(path,'0_results','220715_decision_searchlight')
os.system('mkdir '+path_results)

for i, subj in enumerate(subjects):
    path_subj = os.path.join(path_results, subj)
    os.system('mkdir '+path_subj)
    for j, data_type in enumerate(tasks):
        
        data_type = str.lower(data_type)
        index_ = len(subjects)*i + j
        
        r = res[index_]
        
        mean_map = []
        
        for k, map_ in enumerate(r):
            
            name_ = data_type+'_map_no_'+str(k)+'.nii.gz'
            
            # Print each balance map
            ni.save(map_, os.path.join(path_subj, name_))
            
            # Average across-fold
            map_data = map_.get_data().mean(axis=3)
            mean_map.append(map_data)
        
        # Mean map across balance
        mean_map = np.array(mean_map).mean(axis=3)
        fname_ = os.path.join(path_subj, data_type+'_balance_avg.nii.gz')
        ni.save(ni.Nifti1Image(mean_map, map_.get_affine()), fname_)


from nipy.algorithms.statistics.empirical_pvalue import *

group1 = []

result = dict()
keys_ = ['group1', 'group2']
ds_list = [1]

for t_ in tasks:
    
    for ev_ in evidences:
        
        result['group1'] = []
        result['group2'] = []
        cross_t = []
        cross_p = []
        
        #for i, subj in enumerate(subjects):
        for i, subj in enumerate(group1):
    
            
            subj_map = []
            
            if t_ == 'decision':
                ds_list = [1, 2, 3, 4, 5]
            else:
                ds_list = [1]

            for ds_name in ds_list:
            
                folder_ = '%s_%s_BETA_MVPA_evidence_%s_balance_ds_%s' \
                            % (subj, t_, str(ev_), str(ds_name))
                 
                mean_map =  '%s_%s_BETA_MVPA_evidence_%s_balance_ds_%s_radius_3_searchlight_mean_map.nii.gz' \
                            % (subj, t_, str(ev_), str(ds_name))
                
                total_map =  '%s_%s_BETA_MVPA_evidence_%s_balance_ds_%s_radius_3_searchlight_total_map.nii.gz' \
                            % (subj, t_, str(ev_), str(ds_name))
                    
                
                img_name = os.path.join(path, subj, folder_, mean_map)
                img_ = ni.load(img_name)
                
                subj_map.append(img_.get_data())
                
            subj_map = np.array(subj_map)
            subj_map = np.rollaxis(subj_map, 0, 4)
            
            t, p = ttest_1samp(subj_map, 0.5, axis=3)
            
            s_name = '%s_%s_evidence_%s_balance_ds_%s' % (subj, t_, str(ev_), str(ds_name))
            s_total = os.path.join(path, subj, s_name)
            
            ni.save(ni.Nifti1Image(subj_map.mean(3), img_.get_affine()), s_total+'_mean.nii.gz')
            ni.save(ni.Nifti1Image(t, img_.get_affine()), s_total+'_t.nii.gz')
            ni.save(ni.Nifti1Image(p, img_.get_affine()), s_total+'_p.nii.gz')
            
            img_mean= subj_map.mean(3)
            
            # check group
            if subj in group1:
                array_ = result['group1']
            else:
                array_ = result['group2']
                
                
            #array_.append(img_.get_data())
            array_.append(img_mean)
            
            t, p = check_total(folder_, total_map, t_, subj, ev_)
            cross_t.append(t)
            cross_p.append(p)
            
        for k_ in keys_:
            arr_ = np.array(result[k_])
            arr_ = np.rollaxis(arr_, 0, 4)
            
            #save_stuff(k_, t_, str(ev_), arr_, img_.get_affine())
        
        cross_t = np.array(cross_t)
        cross_t = np.rollaxis(cross_t, 0, 4)
        
        cross_p = np.array(cross_p)
        cross_p = np.rollaxis(cross_p, 0, 4)
        
        fname_save = os.path.join(path, '000_total_cross_validation_t_%s_evidence_%s.nii.gz' %(t_, str(ev_)))
        #ni.save(ni.Nifti1Image(cross_t, img_.get_affine()), fname_save)
        
        fname_save = os.path.join(path, '000_total_cross_validation_p_%s_evidence_%s.nii.gz' %(t_, str(ev_)))
        #ni.save(ni.Nifti1Image(cross_p, img_.get_affine()), fname_save)
        
        total_map = np.concatenate((np.array(result[keys_[0]]), np.array(result[keys_[1]])), axis=0)
        total_map = np.rollaxis(total_map, 0, 4)
        
        #save_stuff('total', t_, str(ev_), total_map, img_.get_affine())
        

def check_total(folder_, total_, task_, subj_, evidence_):
    
    img_name = os.path.join(path, subj_, folder_, total_)
    img_ = ni.load(img_name)
    
    map_ = img_.get_data()
    affine = img_.get_affine()
    
    t, p = ttest_1samp(map_, 0.5, axis=3)
    
    
    fname_save = os.path.join(path, subj_, '%s_%s_evidence_%s_%s.nii.gz' %(subj_, task_, str(evidence_), 't'))
    ni.save(ni.Nifti1Image(t, affine), fname_save)
    
    fname_save = os.path.join(path, subj_, '%s_%s_evidence_%s_%s.nii.gz' %(subj_, task_, str(evidence_), 'p'))
    ni.save(ni.Nifti1Image(p, affine), fname_save)
    
    fname_save = os.path.join(path, subj, '%s_%s_evidence_%s_p_fdr_corr.nii.gz' %(subj_, task_, str(evidence_)))
    
    fdr_ = fdr(p[np.logical_not(np.isnan(p))])
    fdr_map = np.zeros_like(p)
    fdr_map[np.logical_not(np.isnan(p))] = fdr_
    
    ni.save(ni.Nifti1Image(1.-fdr_map, affine), fname_save)   
    
    return t, p

        
        
def save_stuff(group, task_, evidence, map_, affine):
    
    fname_save = os.path.join(path, '%s_%s_evidence_%s_mean.nii.gz' %(group, task_, str(evidence)))
    ni.save(ni.Nifti1Image(map_.mean(3), affine), fname_save)
    
    fname_save = os.path.join(path, '%s_%s_evidence_%s_total.nii.gz' %(group, task_, str(evidence)))
    ni.save(ni.Nifti1Image(map_, affine), fname_save)
    
    for chance_ in [0.575, 0.5]:
        t, p = ttest_1samp(map_, chance_, axis=3)

        fname_save = os.path.join(path, '%s_%s_evidence_%s_t_chance_%s.nii.gz' %(group, task_, str(evidence), str(chance_)))
        ni.save(ni.Nifti1Image(t, affine), fname_save)
        fname_save = os.path.join(path, '%s_%s_evidence_%s_p_chance_%s.nii.gz' %(group, task_, str(evidence), str(chance_)))
        ni.save(ni.Nifti1Image(p, affine), fname_save)
        fname_save = os.path.join(path, '%s_%s_evidence_%s_p_chance_%s_fdr_corr.nii.gz' %(group, task_, str(evidence), str(chance_)))
        fdr_ = fdr(p[np.logical_not(np.isnan(p))])
        fdr_map = np.zeros_like(p)
        fdr_map[np.logical_not(np.isnan(p))] = fdr_
        ni.save(ni.Nifti1Image(1.-fdr_map, affine), fname_save)            

#############################################################à

path = '/media/robbis/DATA/fmri/memory/0_results/balanced_analysis/local'
for s in subjects:
    dirs_ = os.listdir(path)
    
    dirs_ = [d for d in dirs_ if d.find(s) != -1]
    
    print 'mkdir '+os.path.join(path, s)
    n_dir = os.path.join(path, s)
    
    for d in dirs_:
        print 'mv '+os.path.join(path, d)+' '+os.path.join(path, s, d)
        
                    
                