#import _pickle as pickle

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
from pyitab.analysis.results.base import get_results, filter_dataframe
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, \
    SampleSlicer, TargetTransformer
from pyitab.preprocessing.normalizers import SampleSigmaNormalizer, \
    FeatureZNormalizer, SampleZNormalizer
from pyitab.preprocessing.memory import MemoryReducer
from pyitab.base import Node, Transformer
from pyitab.analysis.decoding.roi_decoding import Decoding
from pyitab.analysis.searchlight import SearchLight
from pyitab.io.connectivity import load_mat_ds

from imblearn.under_sampling import RandomUnderSampler

import warnings
from pyitab.preprocessing.math import AbsoluteValueTransformer
warnings.filterwarnings("ignore")
   
######################################
# Only when running on permut1
from pyitab.utils import enable_logging
root = enable_logging()
#####################################


configuration_file = "/home/carlos/fmri/carlo_mdm/memory.conf"
#configuration_file = "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"

loader = DataLoader(configuration_file=configuration_file, 
                    data_path="/home/carlos/mount/meg_workstation/Carlo_MDM/",
                    task='BETA_MVPA', 
                    event_file="full", 
                    brain_mask="mask_intersection")

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer()
                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)

ds = MemoryReducer(dtype=np.float16).transform(ds)


_default_options =  [{
                      'target_transformer__attr': "image_type",
                      'sample_slicer__image_type': ["I", "O"],
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 60, "O": 60}, return_indices=True),
                     },
                      {
                      'target_transformer__attr': "decision",
                      'sample_slicer__image_type': ["N", "O"],
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 60, "O": 60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "evidence",
                      'sample_slicer__evidence': [1, 3, 5],
                      'sample_slicer__accuracy': ["C"],
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={1: 40, 3: 40, 5: 40}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "accuracy",
                      'sample_slicer__accuracy': ["I", "C"],
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 60, "C": 60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "memory_status",
                      'sample_slicer__memory_status': ["N", "O"],
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 60, "O": 60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__motor_resp': ["P", "S"],
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"P":60, "S":60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__target_side': ["L", "R"],
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"L":60, "R":60}, return_indices=True),
                     },   
                        ]


_default_config = {
                    'prepro': ['sample_slicer', 'target_transformer', 'balancer'],
                    "balancer__attr": 'subject',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    #'cv__n_splits': 50,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': SearchLight,
                    'analysis__n_jobs': -1,
                    'analysis__permutation': 0,
                    'analysis__radius': 9,
                    'analysis__save_partial': True,
                    
                    'analysis__verbose': 1,

                    'kwargs__cv_attr': 'subject',

                    }
 
errs = []
iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config), 
                            kind='configuration')
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name="fingerprint").fit(ds, **kwargs)
        a.save(path="/home/carlos/fmri/carlo_mdm/0_results/")
    except Exception as err:
        errs.append([conf._default_options, err])