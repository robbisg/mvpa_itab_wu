##############è Convert ######################
from mvpa_itab.io.base import load_dataset, read_configuration
from mvpa_itab.main_wu import preprocess_dataset
from mvpa_itab.timewise import AverageLinearCSVM, ErrorPerTrial, StoreResults
from mvpa2.measures.base import CrossValidation, Dataset
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import mean_group_sample
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error
import nibabel as ni
import numpy as np
from mvpa2.clfs.base import Classifier
from mvpa2.generators.resampling import Balancer
import mvpa_itab.results as rs

from sklearn.svm import SVC
from mvpa2.clfs.skl.base import SKLLearnerAdapter

#path = '/media/robbis/DATA/fmri/memory/'


if __debug__:
    debug.active += ["SLC"]

evidences = [1,3,5]
#subjects = ['110929angque']#, '110929anngio']
#tasks = ['BETA_MVPA']#, 'RESIDUALS_MVPA']
subjects = os.listdir(path)
tasks = ['memory', 'decision']
res = []
result_dict = dict()
 
conf = read_configuration(path, 'remote_memory.conf', 'RESIDUALS_MVPA')

conf['analysis_type'] = 'searchlight'
conf['analysis_task'] = 'decision_weights'

summarizers = [rs.SearchlightSummarizer()]
savers = [rs.SearchlightSaver()]
collection = rs.ResultsCollection(conf, path, summarizers)
for ev in evidences:
    for subj in subjects:
        for task_ in tasks:
            
            
            data_type = 'BETA_MVPA'
            conf = read_configuration(path, 'remote_memory.conf', data_type)
            #conf['mask_area'] = 'PCC'
            ds = load_dataset(path, subj, data_type, **conf)
            
            ev = str(ev)
            
            # label managing
            if task_ == 'memory':
                field_ = 'stim'
                conf['label_dropped'] = 'F0'
                conf['label_included'] = 'N'+ev+','+'O'+ev
                count_ = 1
            else: # decision
                field_ = 'decision'
                conf['label_dropped'] = 'FIX0'
                conf['label_included'] = 'NEW'+ev+','+'OLD'+ev
                count_ = 250

            ds.targets = np.core.defchararray.add(np.array(ds.sa[field_].value, dtype=np.str), 
                                                  #np.array(ds.sa.stim, dtype=np.str),
                                                  np.array(ds.sa.evidence,dtype= np.str))

            
            ds = preprocess_dataset(ds, data_type, **conf)
    
            balanc = Balancer(count=count_, apply_selection=True)
            gen = balanc.generate(ds)
            #clf = AverageLinearCSVM(C=1)
            
            clf = LinearCSVMC(C=1)
            #avg = TrialAverager(clf)
            cv_storage = StoreResults()
            
            #skclf = SVC(C=1, kernel='linear', class_weight='auto')
            #clf = SKLLearnerAdapter(skclf)
            
            cvte = CrossValidation(clf, 
                                   NFoldPartitioner(cvtype=2),
                                   #errorfx=ErrorPerTrial(), 
                                   #callback=cv_storage,
                                   enable_ca=['stats', 'probabilities'])
    
            sl = sphere_searchlight(cvte, radius=3, space = 'voxel_indices')
            maps = []
            
            
            for i, ds_ in enumerate(gen):
                
                #Avoid balancing!
                #ds_ = ds
                
                sl_map = sl(ds_)
                sl_map.samples *= -1
                sl_map.samples +=  1
                #sl_map.samples = sl_map.samples.mean(axis=len(sl_map.shape)-1)
                map = map2nifti(sl_map, imghdr=ds.a.imghdr)
                maps.append(map)
                
                name = "%s_%s_evidence_%s_balance_ds_%s" %(subj, task_, data_type, str(ev), str(i+1))
                result_dict['radius'] = 3
                result_dict['map'] = map
                
                subj_result = rs.SubjectResult(name, result_dict, savers)
                collection.add(subj_result)
            
            res.append(maps)
                #err = cvte(ds_)

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
        
        
        
    