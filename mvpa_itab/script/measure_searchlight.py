from mvpa_itab.io.base import load_dataset, read_configuration
from mvpa_itab.main_wu import detrend_dataset
from mvpa_itab.timewise import AverageLinearCSVM, ErrorPerTrial, StoreResults
import mvpa_itab.similarity.searchlight as slsim 
from mvpa2.measures.base import CrossValidation, Dataset
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner,\
    NGroupPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import mean_group_sample, map2nifti
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error
import nibabel as ni
import numpy as np
from mvpa2.clfs.base import Classifier
from mvpa2.generators.resampling import Balancer
import mvpa_itab.results as rs
from mvpa2.misc.neighborhood import Sphere, IndexQueryEngine
from mvpa2.measures.searchlight import Searchlight



#path = '/media/robbis/DATA/fmri/memory/'

conf = read_configuration(path, 'remote_memory.conf', 'BETA_MVPA')

conf['analysis_type'] = 'searchlight'
conf['analysis_task'] = 'memory_regression_sample_wise'
conf['mask_area'] = 'total'
task_ = 'BETA_MVPA'
subj = '110929anngio'

partitioners = [NGroupPartitioner(k) for k in np.arange(2, 5)]
result_dict = dict()

summarizers = [rs.SearchlightSummarizer()]
savers = [rs.SearchlightSaver()]
collection = rs.ResultsCollection(conf, path, summarizers)


for i, partitioner in enumerate(partitioners):
    ds = load_dataset(path, subj, task_, **conf)
    
    ds.sa['memory_evidence'] = np.ones_like(ds.targets, dtype=np.int)
    ds.sa.memory_evidence[ds.sa.stim == 'N'] = -1
    ds.sa.memory_evidence = ds.sa.memory_evidence * ds.sa.evidence
    
    ds.targets = [str(ii) for ii in ds.sa.memory_evidence]
    
    conf['label_dropped'] = '0'
    conf['label_included'] = ','.join([str(n) for n in np.array([-5,-3,-1,1,3,5])])
    
    ds = detrend_dataset(ds, task_, **conf)
    ds.targets = np.float_(ds.targets)
    ds.targets = (ds.targets - np.mean(ds.targets))/np.std(ds.targets)
    cv = CrossValidation(slsim.RegressionMeasure(), 
                            partitioner,
                            #NFoldPartitioner(cvtype=1),
                            errorfx=None
                             )
        
    
    kwa = dict(voxel_indices=Sphere(3))
    queryengine = IndexQueryEngine(**kwa)
    
    sl = Searchlight(cv, queryengine=queryengine)
    
    map_ = sl(ds)
    
    map_nii = map2nifti(map_, imghdr=ds.a.imghdr)  
    name = "%s_%s_regression_fold_%s" %(subj, task_, str(i))

    result_dict['radius'] = 3
    result_dict['map'] = map_nii
            
    subj_result = rs.SubjectResult(name, result_dict, savers)
    collection.add(subj_result) 
            
            
            
