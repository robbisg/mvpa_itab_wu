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
from mvpa_itab.wrapper.sklearn import SKLCrossValidation
from sklearn.cross_validation import StratifiedKFold

from sklearn.svm import SVC
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from mvpa2.suite import debug, sphere_searchlight
from mvpa2.suite import *

from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from numpy.random.mtrand import permutation

#path = '/DATA11/roberto/fmri/memory/'
path = '/home/robbis/fmri/memory/'


if __debug__:
    debug.active += ["SLC"]

evidences = [1,3,5]
subjects = ['110929angque',
 '110929anngio',
 '111004edolat',
 '111006giaman',
 '111006rossep',
 '111011marlam',
 '111011corbev',
 '111013fabgue',
 '111018montor',
 '111020adefer',
 '111020mardep',
 '111027ilepac',
 '111123marcai',
 '111123roblan',
 '111129mirgra',
 '111202cincal',
 '111206marant',
 '111214angfav',
 '111214cardin',
 '111220fraarg',
 '111220martes',
 '120119maulig',
 '120126andspo',
 '120112jaclau']  
tasks = ['memory', 'decision']
res = []
result_dict = dict()
 
conf = read_configuration(path, 'memory.conf', 'BETA_MVPA')

conf['analysis_type'] = 'searchlight'
conf['analysis_task'] = 'memory'

summarizers = [rs.SearchlightSummarizer()]
savers = [rs.SearchlightSaver()]
collection = rs.ResultsCollection(conf, path, summarizers)

subjects = subjects[:3]

for subj in subjects:
    
    data_type = 'BETA_MVPA'
    conf = read_configuration(path, 'memory.conf', data_type)
    data_path = conf['data_path']
    ds_original = load_dataset(data_path, subj, data_type, **conf)

    task_ = 'memory'
    ev = '3'
            
    print '---------------------------------'
    
    ev = str(ev)
    
    ds = ds_original.copy()
    '''
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
        count_ = 5

    ds.targets = np.core.defchararray.add(np.array(ds.sa[field_].value, dtype=np.str), 
                                          np.array(ds.sa.evidence,dtype= np.str))
    '''

    ds.targets = ds.sa.memory_status

    conf['label_dropped'] = 'None'
    conf['label_included'] = 'all'
    ds = preprocess_dataset(ds, data_type, **conf)
    count_ = 1
    field_ = 'memory'
    balanc = Balancer(count=count_, apply_selection=True, limit=None)
    gen = balanc.generate(ds)
    
    cv_storage = StoreResults()

    
    
    clf = LinearCSVMC(C=1)
                
    # This is used for the sklearn crossvalidation
    y = np.zeros_like(ds.targets, dtype=np.int_)
    y[ds.targets == ds.uniquetargets[0]] = 1
    
    # We needs to modify the chunks in order to use sklearn
    ds.chunks = np.arange(len(ds.chunks))
    
    permut_ = []
    
    i = 3
    
    partitioner = SKLCrossValidation(StratifiedKFold(y, n_folds=i))
    
    cvte = CrossValidation(clf,
                           partitioner,
                           enable_ca=['stats', 'probabilities'])
    
    sl = sphere_searchlight(cvte, radius=3, space = 'voxel_indices')
    
    maps = []
    
    for p_ in range(100):
        
        print '-------- '+str(p_+1)+' of 100 ------------'
        
        y_perm = permutation(range(len(ds.targets)))
        
        ds.targets = ds.targets[y_perm]
        
        sl_map = sl(ds)
        sl_map.samples *= -1
        sl_map.samples +=  1
    
        map_ = map2nifti(sl_map, imghdr=ds.a.imghdr)
        ni.save(map_, os.path.join(path, subj+'_permutation_'+str(p_+1)+'.nii.gz'))
        permut_.append(map_.get_data())
        
        
    permut_ = np.array(permut_).mean(4)
    permut_ = np.rollaxis(permut_, 0, 4)
    perm_map = ni.Nifti1Image(permut_, map_.get_affine())
    ni.save(perm_map, 
            os.path.join(path, subj+'_permutation_'+str(i)+'_'+task_+'_'+ev+'.nii.gz'))
        
    maps.append(permut_)
        
    name = "%s_%s_%s_evidence_%s_balance_ds_%s" %(subj, task_, data_type, str(ev), str(i+1))
    result_dict['radius'] = 3
    result_dict['map'] = perm_map
        
    subj_result = rs.SubjectResult(name, result_dict, savers)
    collection.add(subj_result)
        