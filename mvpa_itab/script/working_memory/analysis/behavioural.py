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
from pyitab.analysis.decoding.regression import RoiRegression
from joblib import Parallel, delayed
from sklearn.feature_selection import f_regression


def generate(configurator):
    loader = configurator._get_loader()
    fetch_kwargs = configurator._get_params('fetch')
    ds = loader.fetch(**fetch_kwargs)
    return ds



def run(n_jobs):
    path = "/media/robbis/Seagate_Pt1/data/working_memory/"

    conf_file = "%s/data/working_memory.conf" % (path)
    
    
    ### Load datasets ###
    
    iterator_kwargs = {
                        "loader__img_pattern":[
                                    #'power_parcel.mat',
                                    'power_normalized.mat', 
                                    #'connectivity_matrix.mat'
                                    'mpsi_normalized.mat'
                        ],
                        "fetch__prepro": [['none'], 
                                          ['none']],
                        "loader__task": ["POWER", "CONN"]
    }
    
    config_kwargs = {
                        'loader': DataLoader,
                        'loader__configuration_file': conf_file, 
                        'loader__loader':'mat',
                        'loader__task':'POWER',
                        #'fetch__n_subjects': 57,
                        "loader__data_path":"%s/data/" % (path),
                        "loader__subjects": "%s/data/participants.csv" % (path),
    }
    
    iterator = AnalysisIterator(iterator_kwargs, 
                                AnalysisConfigurator,
                                config_kwargs=config_kwargs, 
                                kind='list')
    
    ds_list = [generate(configurator) for configurator in iterator]
    
    
    for i, ds in enumerate(ds_list):
        ds_ = ds.copy()
        if i == 0:
            k = np.arange(1, 88, 10)
            ds_ = DatasetFxNormalizer(ds_fx=np.mean).transform(ds_)
        else:
            k = np.arange(1, 400, 50)
            #ds_ = DatasetFxNormalizer(ds_fx=np.mean).transform(ds_)
            
        _default_options = {
                            #'sample_slicer__targets' : [['0back', '2back'], ['0back', 'rest'], ['rest', '2back']],
                            #'kwargs__ds': ds_list,
                            'sample_slicer__targets' : [['0back'], ['2back']],
                            'target_transformer__attr':['accuracy_0back_both', 'accuracy_2back_both', 
                            'rt_0back_both', 'rt_2back_both'],
                            'sample_slicer__band': [['alpha'], ['beta'], ['theta'], ['gamma']],
                            'estimator__fsel__k': k,
                            'clf__C':[1, 10, 100],
                            'clf__kernel':['linear', 'rbf']
                            }    
            
        _default_config = {

                            
                            'prepro': ['sample_slicer', 'target_transformer'],
                            'sample_slicer__band': ['gamma'], 
                            'sample_slicer__targets' : ['0back', '2back'],

                            'estimator': [('fsel', SelectKBest(score_func=f_regression,
                                                               k=5)),
                                          ('clf', SVR(C=10, kernel='linear'))],
                            'estimator__clf__C': 1,
                            'estimator__clf__kernel':'linear',

                            'cv': GroupShuffleSplit,
                            'cv__n_splits': 75,
                            'cv__test_size': 0.25,

                            'analysis_scoring' : ['r2', 'neg_mean_squared_error'],

                            'analysis': RoiRegression,
                            'analysis__n_jobs': n_jobs,
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
            a = AnalysisPipeline(conf, name="triton+behavioural").fit(ds_, **kwargs)
            a.save()
            del a


if __name__ == '__main__':
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    from pyitab.utils import enable_logging
    root = enable_logging()
    n_jobs = int(sys.argv[1])
    run(n_jobs)