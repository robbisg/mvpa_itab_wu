###################
import _pickle as pickle

from sklearn.model_selection._split import  GroupShuffleSplit
from sklearn.svm.classes import SVC
import numpy as np
from pyitab.io.loader import DataLoader
from scipy.io.matlab.mio import loadmat
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.results import get_results, filter_dataframe
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, SampleSlicer, \
    TargetTransformer, Transformer
from pyitab.preprocessing.normalizers import FeatureZNormalizer, \
    SampleZNormalizer, SampleZNormalizer, SampleSigmaNormalizer, \
    FeatureSigmaNormalizer
from pyitab.preprocessing import Node
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.io.connectivity import load_mat_ds


from pyitab.preprocessing.math import AbsoluteValueTransformer, SignTransformer

import warnings
warnings.filterwarnings("ignore")
 
######################################
# Only when running on permut1
from mvpa_itab.utils import enable_logging
root = enable_logging()
#####################################

#conf_file = "/home/carlos/mount/megmri03/working_memory/working_memory_remote.conf"
conf_file =  "/media/robbis/DATA/fmri/working_memory/working_memory.conf"


loader = DataLoader(configuration_file=conf_file, 
                    loader=load_mat_ds,
                    task='MPSI_NORM')

prepro = PreprocessingPipeline(nodes=[
                                      Transformer(), 
                                      #SignTransformer(),
                                      Detrender(),
                                      #AbsoluteValueTransformer(),
                                      SignTransformer(),
                                      #SampleSigmaNormalizer(),
                                      #FeatureSigmaNormalizer(),
                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)
    
_default_options = {
                       'sample_slicer__targets' : [['0back', '2back'], ['0back', 'rest'], ['rest', '2back']],
                       #'sample_slicer__targets' : [['0back', '2back']],
                       'sample_slicer__band': [[c] for c in np.unique(ds.sa.band)],
                       #'estimator__clf__C': [0.1, 1, 10],                          
                       #'cv__n_splits': [75, 150, 200, 250],
                       #'estimator__fsel__k': np.arange(50, 350, 50),
                       'estimator__fsel__k':np.arange(5, 410, 5),
                       #'estimator__fsel__k': np.arange(1, 88, 1)
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

                       'analysis': Decoding,
                       'analysis__n_jobs': 5,
                       'analysis__permutation': 0,
                       'analysis__verbose': 0,
                       'kwargs__roi': ['matrix_values'],
                       'kwargs__cv_attr':'subjects',

                    }
 
 
iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="wm_mpsi_norm_sign").fit(ds, **kwargs)
    a.save()
    del a


###############################################################
# 2019 #
conf_file =  "/media/robbis/DATA/fmri/working_memory/working_memory.conf"


loader = DataLoader(configuration_file=conf_file, 
                    loader=load_mat_ds,
                    task='MPSI_NORM')

prepro = PreprocessingPipeline(nodes=[
                                      Transformer(), 
                                      #SignTransformer(),
                                      Detrender(),
                                      AbsoluteValueTransformer(),
                                      #SignTransformer(),
                                      #SampleSigmaNormalizer(),
                                      #FeatureSigmaNormalizer(),
                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)
    
_default_options = {
                       'sample_slicer__targets' : [['0back', '2back'], ['0back', 'rest'], ['rest', '2back']],
                       #'sample_slicer__targets' : [['0back', '2back']],
                       'sample_slicer__band': [[c] for c in np.unique(ds.sa.band)],
                       #'estimator__clf__C': [0.1, 1, 10],                          
                       #'cv__n_splits': [75, 150, 200, 250],
                       #'estimator__fsel__k': np.arange(50, 350, 50),
                       'estimator__fsel__k':np.arange(1, 4851, 5),
                       #'estimator__fsel__k': np.arange(1, 88, 1)
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
 
 
iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="wm_mpsi_norm_abs").fit(ds, **kwargs)
    a.save(subdir="0_results/analysis_201901/")
    del a

