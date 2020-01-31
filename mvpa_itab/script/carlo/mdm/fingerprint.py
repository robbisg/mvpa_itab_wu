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
from pyitab.base import Node
from pyitab.preprocessing.base import Transformer
from pyitab.analysis.decoding.roi_decoding import Decoding
from pyitab.analysis.searchlight import SearchLight
from pyitab.io.connectivity import load_mat_ds

from sentry_sdk import capture_exception

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
                                      FeatureZNormalizer(),

                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)

ds = MemoryReducer(dtype=np.float16).transform(ds)

import sentry_sdk  
sentry_sdk.init("https://f2866916959e41bc81abdfaf580f3d26@sentry.io/1439199")  

_default_options =  [{
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {
                          'image_type':["I", "O"],
                      },
                  
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 60, "O": 60}, return_indices=True),
                     },
               
                      {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr':{'decision': ["N", "O"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 60, "O": 60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "evidence",
                      'sample_slicer__attr':{'evidence': [1, 3, 5],
                                            #'accuracy': ["C"]
                                            },
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={1:40, 3:40, 5:40}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "accuracy",
                      'sample_slicer__attr':{'accuracy': ["I", "C"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 60, "C": 60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "memory_status",
                      'sample_slicer__attr':{'memory_status': ["N", "O"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 60, "O": 60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr':{'motor_resp': ["P", "S"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"P":60, "S":60}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr':{'target_side': ["L", "R"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"L":60, "R":60}, return_indices=True),
                     },   
                        ]


_default_config = {
                    'prepro': ['target_transformer', 'sample_slicer', 'balancer'],
                    "balancer__attr": 'subject',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    #'cv__n_splits': 50,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': SearchLight,
                    'analysis__n_jobs': 10,
                    'analysis__permutation': 0,
                    'analysis__radius': 9,
                    'analysis__save_partial': False,
                    
                    'analysis__verbose': 1,

                    'kwargs__cv_attr': 'subject',

                    }
 
errs = []

filtered_options = [_default_options[2]]

iterator = AnalysisIterator(filtered_options, 
                            AnalysisConfigurator(**_default_config), 
                            kind='configuration')
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name="fingerprint").fit(ds, **kwargs)
        a.save(path="/home/carlos/fmri/carlo_mdm/0_results/")
    except Exception as err:
        errs.append([conf._default_options, err])
        capture_exception(err)


##### Results #####
from pyitab.analysis.results.base import filter_dataframe
from pyitab.analysis.results.bids import get_searchlight_results_bids
from scipy.stats import zscore


dataframe = get_searchlight_results_bids('/media/robbis/DATA/fmri/carlo_mdm/0_results/derivatives/')
df = filter_dataframe(dataframe, id=['uv5oyc6s'], filetype=['full'])

mask = ni.load("/media/robbis/DATA/fmri/carlo_mdm/1_single_ROIs/mask_intersection.nii.gz").get_data()






image_dict = {}
image_list = []
for i, attr in enumerate(df['attr'].values):
    img = ni.load(df['filename'].values[i])
    data = img.get_data()[np.bool_(mask)]

    zdata = zscore(data, axis=0)

    image_dict[attr] = zdata
    image_list.append(zdata)

image_full = np.array(image_list)

from sklearn.cluster import KMeans
kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10)


for i in range(24):
    X = image_full[...,i].T
    kmeans.fit(X)
