from mvpa_itab.test_wu import _test_spatial
import os
import numpy as np
from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline,\
    StandardPreprocessingPipeline
from imblearn.under_sampling import RandomUnderSampler
from mvpa_itab.preprocessing.functions import SampleSlicer, TargetTransformer,\
    Node, Detrender, FeatureWiseNormalizer, SampleWiseNormalizer
from mvpa_itab.preprocessing.balancing import Balancer
from mvpa_itab.pipeline.analysis.roi_decoding import Decoding
from mvpa_itab.io import DataLoader
from sklearn.model_selection._split import *
from sklearn.svm.classes import SVC




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
    rois = os.listdir(os.path.join(path, roi_path))
    rois = [roi for roi in rois if roi.find('nii.gz') != -1]

    
    default_config = {
                      #'targets': ['decision', 'memory'],
                      'mask_label__evidence': [1, 3, 5],
                      'mask_area': rois,                 
                      'normalization': ['feature','sample','both'],
                      #'balance__count': [50],
                      'chunks':['adaptive', 5]
                      }
    
    default_config.update(kwargs)
    
    configuration_generator = get_configuration(default_config)
    
    subjects = np.loadtxt(os.path.join(path,'subjects.csv'),
                          dtype=np.str, 
                          delimiter=',')
    subjects_ = subjects[1:,0]
    
    
    config = {
              'target':'decision',
              'balancer__count':50,
              'chunk_number':5,
              'normalization':'feature',
              'saver__fields':['classifier', 'stats']
              }
    
    
    for conf in configuration_generator:

        config.update(conf)
        analysis_type = "%s_%s_%s" % (config['target'],
                                      config['mask_label__evidence'],
                                      config['normalization'])
        
        res = _test_spatial(path, 
                            subjects_, 
                            'memory.conf', 
                            'BETA_MVPA', 
                            analysis_type=analysis_type, 
                            **config)
        
    
###############
import cPickle as pickle

loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", task='OFP_NORES')
ds = loader.fetch()

decoding = Decoding(n_jobs=20, scoring=['accuracy'])

results = dict()
for subject in np.unique(ds.sa.subject):
    results[subject] = []
    for evidence in [1, 2, 3]:
        
        pipeline = PreprocessingPipeline(nodes=[ 
                                                TargetTransformer('decision'),
                                                SampleSlicer({'subject':[subject], 'evidence':[evidence]}),
                                                Balancer(balancer=RandomUnderSampler(return_indices=True), attr='chunks'),
                                                ])
        
        ds_ = pipeline.transform(ds)
        
        decoding.fit(ds_, roi=['lateral_ips'])

        results[subject].append(decoding.scores)

with open(os.path.join(loader._data_path, '0_results/lateral_ips_decoding.pickle'), 'wb') as output:
    pickle.dump(results, output)
    
    
    
###################
import cPickle as pickle

loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", task='OFP_NORES')

detrending_pipe =  PreprocessingPipeline(nodes=[Detrender(chunks_attr='file'), Detrender()])
zscore_pipe = PreprocessingPipeline(nodes=[FeatureWiseNormalizer(), SampleWiseNormalizer()])

ds = loader.fetch(prepro=detrending_pipe)

decoding = Decoding(n_jobs=20, 
                    scoring=['accuracy'], 
                    cv=StratifiedKFold(n_splits=5),
                    estimator=SVC(C=1, kernel='linear')
                    )

results = dict()
for subject in np.unique(ds.sa.subject):
    results[subject] = []
    for evidence in [1, 2, 3]:
        
        pipeline = PreprocessingPipeline(nodes=[ 
                                                TargetTransformer('decision'),
                                                SampleSlicer({'subject':[subject], 'evidence':[evidence]}),
                                                Balancer(balancer=RandomUnderSampler(return_indices=True), attr='all'),
                                                ])
        
        ds_ = pipeline.transform(ds)
        
        decoding.fit(ds_, roi=['lateral_ips'], cv_attr=None, prepro=zscore_pipe)

        results[subject].append(decoding.scores)

with open(os.path.join(loader._data_path, '0_results/within_detrending_total_zscore_roi_5_fold.pickle'), 'wb') as output:
    pickle.dump(results, output)
    
    
    
##################
decoding = Decoding(n_jobs=20, 
                    scoring=['accuracy'], 
                    cv=LeaveOneGroupOut(),
                    estimator=SVC(C=1, kernel='linear'))

results = []

for evidence in [1, 2, 3]:
    
    pipeline = PreprocessingPipeline(nodes=[ 
                                            TargetTransformer('decision'),
                                            SampleSlicer({'evidence':[evidence]}),
                                            Balancer(balancer=RandomUnderSampler(return_indices=True), attr='subject'),
                                            ])
    
    ds_ = pipeline.transform(ds)
    
    decoding.fit(ds_, roi=['lateral_ips'], cv_attr='subject')

    results.append(decoding.scores)
    
with open(os.path.join(loader._data_path, '0_results/across_lateral_ips_balance_subject.pickle'), 'wb') as output:
    pickle.dump(results, output)    
    