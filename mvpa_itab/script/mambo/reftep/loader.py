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

conf_file = "/media/robbis/DATA/meg/reftep/bids.conf"
loader = DataLoader(configuration_file=conf_file, 
                    task='reftep', 
                    load_fx='reftep-sensor',
                    loader='bids-meg',
                    bids_space='sensor'
                    )

ds = loader.fetch(n_subjects=1)