
from pyitab.io.loader import DataLoader
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.mapper import SampleZNormalizer, FeatureZNormalizer
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.searchlight import SearchLight
from sklearn.model_selection import *

from sklearn.svm import SVC
from pyitab.io.subjects import load_subject_file

import warnings
warnings.filterwarnings("ignore")
   
######################################
# Only when running on permut1
from pyitab.utils import enable_logging
root = enable_logging()
#####################################

subjects, _ = load_subject_file("/media/robbis/DATA/fmri/EGG/participants.tsv", delimiter="\t")

_default_options = {
                        'loader__task': ['smoothAROMAnonaggr', 'filtered'],
                        'fetch__subject_names':
                            [[s] for s in subjects[21:]]

                        }


_default_config = {
                    'prepro': ['sample_slicer', 'balancer'],

                    'sample_slicer__targets': ['coldstim', 'stim'],
                    "balancer__attr": 'all',
                    

                    'loader': DataLoader,
                    'loader__configuration_file': "/media/robbis/DATA/fmri/EGG/bids.conf",
                    'loader__loader':'bids',
                    #'loader__task':'simulations',
                    'fetch__prepro': ['feature_normalizer', 'sample_normalizer'],
                
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedKFold ,
                    'cv__n_splits': 4,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': SearchLight,
                    'analysis__n_jobs': 5,
                    'analysis__permutation': 0,
                    'analysis__radius': 9,
                    
                    'analysis__verbose': 0,

                    #'kwargs__cv_attr': ['group', 'subject'],

                    }
 
 
iterator = AnalysisIterator(_default_options, AnalysisConfigurator, config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="egg").fit(**kwargs)
    a.save()

##################################
# Analysis 26.11.2020
# 

subjects, _ = load_subject_file("/media/robbis/DATA/fmri/EGG/participants.tsv", delimiter="\t")

_default_options = {
                        'loader__task': #['smoothAROMAnonaggr', 'filtered', 
                                        ['plain'],
                        'fetch__subject_names':
                            [[s] for s in subjects[:]]

                        }


_default_config = {
                    'prepro': ['sample_slicer', 'balancer'],

                    'sample_slicer__targets': ['coldstim', 'stim'],
                    "balancer__attr": 'all',
                    

                    'loader': DataLoader,
                    'loader__configuration_file': "/media/robbis/DATA/fmri/EGG/bids.conf",
                    'loader__loader':'bids',
                    'loader__onset_offset': 1,
                    'loader__extra_duration': 2,
                    #'loader__task':'simulations',
                    'fetch__prepro': ['feature_normalizer', 'sample_normalizer'],
                
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedKFold ,
                    'cv__n_splits': 4,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': SearchLight,
                    'analysis__n_jobs': 5,
                    'analysis__permutation': 0,
                    'analysis__radius': 9,
                    
                    'analysis__verbose': 0,

                    #'kwargs__cv_attr': ['group', 'subject'],

                    }

import sentry_sdk
from sentry_sdk import capture_exception
sentry_sdk.init(
    "https://f2866916959e41bc81abdfaf580f3d26@o252224.ingest.sentry.io/1439199",
    traces_sample_rate=1.0,
)


errors = []
iterator = AnalysisIterator(_default_options, AnalysisConfigurator, config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name="egg+delay").fit(**kwargs)
        a.save()
    except Exception as err:
        capture_exception(err)
        errors.append([conf, err])


##################################
# Check - below-chance
# 

subjects, _ = load_subject_file("/media/robbis/DATA/fmri/EGG/participants.tsv", delimiter="\t")

_default_options = {
                        'loader__task': #['smoothAROMAnonaggr', 'filtered', 
                                        ['plain'],
                        'fetch__subject_names':
                            [[s] for s in subjects[:]]
                        
                        'estimator__clf': [SVC(C=1, kernel='linear'), 
                                           SVC(C=1, kernel='rbf', gamma=1)
                        ]

                        }


_default_config = {
                    'prepro': ['sample_slicer', 'balancer'],

                    'sample_slicer__targets': ['coldstim', 'stim'],
                    "balancer__attr": 'all',
                    

                    'loader': DataLoader,
                    'loader__configuration_file': "/media/robbis/DATA/fmri/EGG/bids.conf",
                    'loader__loader':'bids',
                    'loader__onset_offset': 1,
                    'loader__extra_duration': 2,
                    #'loader__task':'simulations',
                    'fetch__prepro': ['feature_normalizer', 'sample_normalizer'],
                
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    #'estimator__clf__C': 1,
                    #'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedShuffleSplit,
                    'cv__n_splits': 20,
                    'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': SearchLight,
                    'analysis__n_jobs': 4,
                    'analysis__permutation': 0,
                    'analysis__radius': 9,
                    
                    'analysis__verbose': 0,

                    #'kwargs__cv_attr': ['group', 'subject'],

                    }

import sentry_sdk
from sentry_sdk import capture_exception
sentry_sdk.init(
    "https://f2866916959e41bc81abdfaf580f3d26@o252224.ingest.sentry.io/1439199",
    traces_sample_rate=1.0,
)


errors = []
iterator = AnalysisIterator(_default_options, AnalysisConfigurator, config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name="egg+below+chance").fit(**kwargs)
        a.save()
    except Exception as err:
        capture_exception(err)
        errors.append([conf, err])