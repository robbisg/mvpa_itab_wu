from sklearn.model_selection import *
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import f_regression

import numpy as np
from pyitab.io.loader import DataLoader
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.decoding.regression import RoiRegression
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, SampleSlicer, \
    TargetTransformer, Transformer
from pyitab.preprocessing.normalizers import FeatureZNormalizer, \
    SampleZNormalizer, SampleZNormalizer, SampleSigmaNormalizer, \
    FeatureSigmaNormalizer


import warnings
warnings.filterwarnings("ignore")


from pyitab.utils import enable_logging
root = enable_logging()

conf_file = "/media/robbis/DATA/meg/reftep/bids.conf"
    
_default_options = {                     
                       
                       'sample_slicer__band' : [
                           ['mu'], 
                           ['betalow'], 
                           ['betahigh']
                       ],

                       'target_transformer__attr': [
                           'mep-right', 'mep-left'
                       ],

                       'target_transformer__fx':[
                           ('log', lambda x: np.log(x))
                       ],
                       'loader__bids_win': ['700', '1490'],

                       #'estimator__fsel__k': np.arange(50, 15000, 1500),

                       'estimator__clf': [
                           MLPRegressor(),
                           SVR(C=1),
                           SVR(C=10),
                           Lasso(alpha=0.5),
                        ],                          
                    }    
    
_default_config = {    
                       'loader': DataLoader, 
                       'loader__configuration_file': conf_file, 
                       'loader__loader': 'bids-meg', 
                       'loader__bids_win': '700',
                       'loader__task': 'reftep',
                       'loader__load_fx': 'reftep-iplv',
                       
                       'fetch__subject_names': ['sub-1'],
                       'fetch__prepro': [Transformer()],
                     
                       
                       'prepro': ['sample_slicer', 'target_transformer'],
                       'target_transformer__fx': lambda x: np.log(x),

                       'balancer__attr':'all',

                       'estimator': [#('fsel', SelectKBest(k=50, score_func=f_regression)),
                                     ('clf', SVR(C=1, kernel='linear'))],

                       'cv': ShuffleSplit,
                       'cv__n_splits': 2,
                       #'cv__test_size': 0.25,

                       'analysis__scoring' : ['r2', 'explained_variance'],

                       'analysis': RoiRegression,
                       'analysis__n_jobs': 1,
                       'analysis__permutation': 0,
                       'analysis__verbose': 0,
                       'kwargs__roi': ['matrix_values'],
                       #'kwargs__cv_attr': 'mep-right',
                    }

from sentry_sdk import capture_exception
import sentry_sdk
sentry_sdk.init(
    "https://f2866916959e41bc81abdfaf580f3d26@o252224.ingest.sentry.io/1439199",
    traces_sample_rate=1.0,
)

errors = []
iterator = AnalysisIterator(_default_options, AnalysisConfigurator, config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name="reftep+iplv+singleregression").fit(**kwargs)
        a.save()
    except Exception as err:
        capture_exception(err)
        errors.append([conf, err])