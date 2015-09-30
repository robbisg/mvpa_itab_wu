############## Convert ######################
from mvpa_itab.io.base import load_dataset, read_configuration
from mvpa_itab.main_wu import preprocess_dataset
from mvpa_itab.time import AverageLinearCSVM, ErrorPerTrial, StoreResults
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

path = '/media/robbis/DATA/fmri/memory/'
    
subjects = ['110929angque']#, '110929anngio']
tasks = ['RESIDUALS_MVPA']#, 'BETA_MVPA']
res = []

for subj in subjects:
    for task_ in tasks:
        
        conf = read_configuration(path, 'memory.conf', task_)
        ds = load_dataset(path, subj, task_, **conf)
        
        ds = preprocess_dataset(ds, task_, **conf)
        
        ds.targets = ds.sa.memory_status
        
        balanc = Balancer(count=1, apply_selection=True)
        gen = balanc.generate(ds)
        clf = AverageLinearCSVM(C=1)
        
        #clf = LinearCSVMC(C=1)
        #avg = TrialAverager(clf)
        cv_storage = StoreResults()

        cvte = CrossValidation(clf, 
                               HalfPartitioner(),
                               errorfx=ErrorPerTrial(), 
                               callback=cv_storage,
                               enable_ca=['stats', 'probabilities'])

        sl = sphere_searchlight(cvte, radius=3, space = 'voxel_indices')
        maps = []
        for ds_ in gen:
            print 'Here we are!'
            sl_map = sl(ds_)
            sl_map.samples *= -1
            sl_map.samples +=  1
            #sl_map.samples = sl_map.samples.mean(axis=len(sl_map.shape)-1)
            map = map2nifti(sl_map, imghdr=ds.a.imghdr)
            maps.append(map)
        
        res.append(maps)
            #err = cvte(ds_)

path_results = os.path.join(path,'0_results','220715_decision_searchlight')
os.system('mkdir '+path_results)

for i, subj in enumerate(subjects):
    path_subj = os.path.join(path_results, subj)
    os.system('mkdir '+path_subj)
    for j, task_ in enumerate(tasks):
        
        task_ = str.lower(task_)
        index_ = len(subjects)*i + j
        
        r = res[index_]
        
        mean_map = []
        
        for k, map_ in enumerate(r):
            
            name_ = task_+'_map_no_'+str(k)+'.nii.gz'
            
            # Print each balance map
            ni.save(map_, os.path.join(path_subj, name_))
            
            # Average across-fold
            map_data = map_.get_data().mean(axis=3)
            mean_map.append(map_data)
        
        # Mean map across balance
        mean_map = np.array(mean_map).mean(axis=3)
        fname_ = os.path.join(path_subj, task_+'_balance_avg.nii.gz')
        ni.save(ni.Nifti1Image(mean_map, map_.get_affine()), fname_)
        
        
        
    