#!/usr/bin/env python3
from sklearn.model_selection._split import  GroupShuffleSplit
from sklearn.svm.classes import SVC
import numpy as np
from pyitab.io.loader import DataLoader
import os

from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, Transformer
from pyitab.preprocessing.normalizers import FeatureZNormalizer, \
    SampleZNormalizer, SampleZNormalizer, SampleSigmaNormalizer, \
    FeatureSigmaNormalizer, DatasetFxNormalizer
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from joblib import Parallel, delayed

path = "/media/robbis/Seagate_Pt1/data/working_memory/"
conf_file = "%s/data/working_memory.conf" % (path)

task = 'PSI'
task = 'PSICORR'


loader = DataLoader(configuration_file=conf_file, 
                    loader='mat',
                    task=task,
                    data_path="%s/data/" % (path),
                    subjects="%s/data/participants.csv" % (path)
                    )

prepro = PreprocessingPipeline(nodes=[Transformer(),
                                      #SampleZNormalizer()
                                      ])


ds = loader.fetch(prepro=prepro)
    
_default_options = {
                       'sample_slicer__targets' : [['0back', '2back']],
                       'sample_slicer__band': [[c] for c in np.unique(ds.sa.band)],
                       'estimator__fsel__k':np.arange(1, 1200, 50),
                    }    
    
_default_config = {
               
                       'prepro':['sample_slicer'],
                       #'ds_normalizer__ds_fx': np.std,
                       'sample_slicer__band': ['gamma'], 
                       'sample_slicer__targets' : ['0back', '2back'],

                       'estimator': [('fsel', SelectKBest(k=150)),
                                     ('clf', SVC(C=1, kernel='linear'))],
                       'estimator__clf__C':1,
                       'estimator__clf__kernel':'linear',

                       'cv': GroupShuffleSplit,
                       'cv__n_splits': 75,
                       'cv__test_size': 0.25,

                       'scores' : ['accuracy'],

                       'analysis': RoiDecoding,
                       'analysis__n_jobs': -1,
                       'analysis__permutation': 0,
                       'analysis__verbose': 0,
                       'kwargs__roi': ['matrix_values'],
                       'kwargs__cv_attr':'subjects',

                    }
 
 
iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator,
                            config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name=task+"+review+singleband+plain").fit(ds, **kwargs)
    a.save()
    del a


################################################################
_default_options = {
                    'sample_slicer__targets' : [['0back', '2back']],
                    'estimator__fsel__k': np.arange(1, 1200, 20)
                    }    
    
_default_config = {
                    
                    'prepro': ['feature_stacker', 'sample_slicer'],
                    
                    'feature_stacker__stack_attr': ['band'],
                    'feature_stacker__keep_attr': ['targets', 'subjects'],
                    
                    'sample_slicer__targets' : ['0back', '2back'],

                    'estimator': [('fsel', SelectKBest(k=5)),
                                    ('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C':1,
                    'estimator__clf__kernel':'linear',

                    'cv': GroupShuffleSplit,
                    'cv__n_splits': 75,
                    'cv__test_size': 0.25,

                    'scores' : ['accuracy'],

                    'analysis': RoiDecoding,
                    'analysis__n_jobs': 4,
                    'analysis__permutation': 0,
                    'analysis__verbose': 0,
                    
                    'kwargs__roi': ['matrix_values'],
                    'kwargs__cv_attr':'subjects',

                    }


iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator,
                            config_kwargs=_default_config)

for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name=task+"multiband+channel+review").fit(ds, **kwargs)
    a.save()