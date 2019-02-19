import _pickle as pickle

from sklearn.svm.classes import SVC
import numpy as np
from pyitab.io.loader import DataLoader
from scipy.io.matlab.mio import loadmat
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.model_selection import *

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.results import get_results, filter_dataframe
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, \
    SampleSlicer, TargetTransformer, Transformer
from pyitab.preprocessing.normalizers import SampleSigmaNormalizer, \
    FeatureZNormalizer, SampleZNormalizer
from pyitab.preprocessing import Node
from pyitab.analysis.decoding.roi_decoding import Decoding
from pyitab.analysis.searchlight import SearchLight
from pyitab.io.connectivity import load_mat_ds

import warnings
from pyitab.preprocessing.math import AbsoluteValueTransformer
warnings.filterwarnings("ignore")
   
######################################
# Only when running on permut1
from mvpa_itab.utils import enable_logging
root = enable_logging()
#####################################

conf_file =  "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"


loader = DataLoader(configuration_file=conf_file, 
                    #loader=load_mat_ds,
                    task='BETA_MVPA')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer()
                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)
    
_default_options = {
                        #'target_trans__target': ["decision"],
                        'sample_slicer__accuracy': [[1], [0]],           
                        }


_default_config = {
                    'prepro': ['sample_slicer', 'target_transformer', 'balancer'],
                    'sample_slicer__decision': ['NEW', 'OLD'],
                    'sample_slicer__evidence': [1],
                    'sample_slicer__accuracy': [0],
                    'target_transformer__target': "decision",
                    "balancer__attr": 'subject',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': DoubleGroupCrossValidator,
                    #'cv__n_splits': 50,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': SearchLight,
                    'analysis__n_jobs': 5,
                    'analysis__permutation': 0,
                    'analysis__radius': 9,
                    
                    'analysis__verbose': 1,

                    'kwargs__cv_attr': ['group', 'subject'],

                    }
 
 
iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="accuracy").fit(ds, **kwargs)
    a.save()



####################### Roi Analysis ##########################
conf_file =  "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"
loader = DataLoader(configuration_file=conf_file, 
                    #loader=load_mat_ds,
                    task='BETA_MVPA')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer()
                                      ])
ds = loader.fetch(prepro=prepro)
    
_default_options = {
                        'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                        'sample_slicer__accuracy': [[1], [0]],           
                        }


_default_config = {
                    'prepro': ['sample_slicer', 'target_transformer', 'balancer'],
                    'sample_slicer__decision': ['NEW', 'OLD'],
                    'sample_slicer__evidence': [1],
                    'sample_slicer__accuracy': [0],
                    'target_transformer__target': "decision",
                    "balancer__attr": 'all',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedKFold,
                    'cv__n_splits': 3,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': Decoding,
                    'analysis__n_jobs': 5,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 1,

                    'kwargs__roi' : ['omnibus'],
                    'kwargs__cv_attr': 'subject'

                    }
 
 
iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="omnibus_roi").fit(ds, **kwargs)
    a.save()


#################### Temporal decoding #################################
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from pyitab.io.base import load_subject_file

conf_file =  "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"
loader = DataLoader(configuration_file=conf_file, 
                    #loader=load_mat_ds,
                    task='RESIDUALS_MVPA')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer(),
                                      SampleSlicer(frame=[1,2,3,4,5,6,7]),
                                      TargetTransformer(attr='decision'),
                                      #Balancer(attr='frame'),
                                      ])

subject_file = "/media/robbis/DATA/fmri/carlo_mdm/subjects.csv"
subjects, extra_sa =  load_subject_file(subject_file)

for s in subjects:
    ds = loader.fetch(prepro=prepro, subject_names=[s])


    _default_options = {
                            'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                            'sample_slicer__evidence': [[1], [3], [5], [1, 3, 5]],           
                            }


    _default_config = {
                        'prepro': ['sample_slicer'],
                        #'sample_slicer__decision': ['NEW', 'OLD'],
                        #'sample_slicer__evidence': [1],
                        #'sample_slicer__frame': [1, 2, 3, 4, 5, 6, 7],
                        #'target_transformer__target': "decision",
                        #"balancer__attr": 'all',
                        
                        'estimator': [('clf', SVC(C=1, kernel='linear'))],
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel': 'linear',
                        
                        'cv': StratifiedKFold,
                        'cv__n_splits': 7,
                        #'cv__test_size': 0.25,
                        
                        'scores': ['accuracy'],
                        
                        'analysis': TemporalDecoding,
                        'analysis__n_jobs': 3,
                        
                        'analysis__permutation': 0,
                        
                        'analysis__verbose': 1,

                        'kwargs__roi' : ['omnibus'],
                        'kwargs__cv_attr': 'chunks'

                        }
    
    
    iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))
    for conf in iterator:
        kwargs = conf._get_kwargs()
        a = AnalysisPipeline(conf, name="temporal_omnibus_roi").fit(ds, **kwargs)
        a.save()



X = np.rollaxis(ds_.samples.reshape(-1, 7, ds_.shape[1]), 1, 3)
y = ds_.sa.decision.reshape(-1, 7)

labels = []
for yy in y:
    l, c = np.unique(yy, return_counts=True)
    labels.append(l[np.argmax(c)])

y = np.array(labels)
balancer = RandomUnderSampler(return_indices=True)
_, _, indices = balancer.fit(X[...,0], y)
indices = np.argsort(indices)

XX, yy = X[indices], y[indices]

estimator = Pipeline(steps=[('clf', SVC(C=1, kernel='linear'))])
time_gen = GeneralizingEstimator(estimator)

scores = cross_validate(time_gen, X[:,:200,:], y_, 
                        groups=None, 
                        scoring='accuracy',
                        cv=StratifiedKFold, 
                        n_jobs=1,
                        verbose=1, 
                        return_estimator=False, 
                        return_splits=False)



