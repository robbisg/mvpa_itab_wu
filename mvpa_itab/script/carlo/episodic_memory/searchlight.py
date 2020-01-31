from pyitab.io.loader import DataLoader
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from sklearn.model_selection import *
from pyitab.analysis.searchlight import SearchLight
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVC
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.configurator import AnalysisConfigurator
import os


from pyitab.utils import enable_logging
root = enable_logging()

conf_file = "/home/carlos/fmri/carlo_ofp/ofp_new.conf"
#conf_file = "/media/robbis/DATA/fmri/carlo_ofp/ofp.conf"
loader = DataLoader(configuration_file=conf_file, task='OFP_NORES')
ds = loader.fetch()

import numpy as np


######################## Across Memory ##################################

_default_options = {
                       'target_trans__target': ["memory_status"],                         
                        }


_default_config = {
               
                        'prepro': ['sample_slicer', 'target_trans'],
                        'sample_slicer__memory_status': ['L', 'F'],
                        'sample_slicer__evidence': [1],
                        'target_trans__target': "memory_status",
                        
                        'estimator': [('clf', SVC(C=1, kernel='linear'))],
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel': 'linear',
                        
                        'cv': LeaveOneGroupOut,
                        #'cv__n_splits': 50,
                        #'cv__test_size': 0.25,
                        
                        'scores': ['accuracy'],
                        
                        'analysis': SearchLight,
                        'analysis__n_jobs': 15,
                        'analysis__permutation': 0,
                        'analysis__radius': 9,
                        
                        'analysis__verbose': 0,

                        'kwargs__cv_attr': 'subject',

                    }



iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))


for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="across_memory").fit(ds, **kwargs)
    a.save()


################# Across Decision ########################
_default_options = {
                       'target_trans__target': ["decision"],                         
                        }


_default_config = {
               
                        'prepro': ['sample_slicer', 'target_trans', 'balancer'],
                        'sample_slicer__decision': ['L', 'F'],
                        'sample_slicer__evidence': [1],
                        'target_trans__target': "decision",
                        "balancer__attr": 'subject',
                        
                        'estimator': [('clf', SVC(C=1, kernel='linear'))],
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel': 'linear',
                        
                        'cv': LeaveOneGroupOut,
                        #'cv__n_splits': 50,
                        #'cv__test_size': 0.25,
                        
                        'scores': ['accuracy'],
                        
                        'analysis': SearchLight,
                        'analysis__n_jobs': 15,
                        'analysis__permutation': 0,
                        'analysis__radius': 9,
                        
                        'analysis__verbose': 1,

                        'kwargs__cv_attr': 'subject',

                    }



iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))


for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="across_decision").fit(ds, **kwargs)
    a.save()

############   Within subjects decision ###############
_default_options = {
                       'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                       'cv__n_splits': [3, 5]                        
                        }


_default_config = {
               
                        'prepro': ['sample_slicer', 'target_trans', 'balancer'],
                        'sample_slicer__decision': ['L', 'F'],
                        'sample_slicer__evidence': [1],
                        'target_trans__target': "decision",
                        "balancer__attr": 'chunks',
                        
                        'estimator': [('clf', SVC(C=1, kernel='linear'))],
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel': 'linear',
                        
                        'cv': StratifiedKFold,
                        'cv__n_splits': 3,
                        #'cv__test_size': 0.25,
                        
                        'scores': ['accuracy'],
                        
                        'analysis': SearchLight,
                        'analysis__n_jobs': 15,
                        'analysis__permutation': 0,
                        'analysis__radius': 9,
                        
                        'analysis__verbose': 0,

                        'kwargs__cv_attr': 'subject',

                    }



iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))


for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="within_decision").fit(ds, **kwargs)
    a.save()


############   Within subjects memory ###############
_default_options = {
                       'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                       'cv__n_splits':[3, 5]                        
                        }


_default_config = {
               
                        'prepro': ['sample_slicer', 'target_trans', 'balancer'],
                        'sample_slicer__memory_status': ['L', 'F'],
                        'sample_slicer__evidence': [1],
                        'target_trans__target': "memory_status",
                        "balancer__attr": 'chunks',
                        
                        'estimator': [('clf', SVC(C=1, kernel='linear'))],
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel': 'linear',
                        
                        'cv': StratifiedKFold,
                        'cv__n_splits': 3,
                        #'cv__test_size': 0.25,
                        
                        'scores': ['accuracy'],
                        
                        'analysis': SearchLight,
                        'analysis__n_jobs': 15,
                        'analysis__permutation': 0,
                        'analysis__radius': 9,
                        
                        'analysis__verbose': 0,

                        'kwargs__cv_attr': 'chunks',

                    }



iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))


for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="within_memory").fit(ds, **kwargs)
    a.save()