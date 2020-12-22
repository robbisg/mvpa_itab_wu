###################
from sklearn.model_selection import *

from sklearn.svm.classes import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


import numpy as np
from pyitab.io.loader import DataLoader
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, SampleSlicer, \
    TargetTransformer, Transformer
from pyitab.preprocessing.normalizers import FeatureZNormalizer, \
    SampleZNormalizer, SampleZNormalizer, SampleSigmaNormalizer, \
    FeatureSigmaNormalizer
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding

import warnings
warnings.filterwarnings("ignore")

from pyitab.utils import make_analysis
path = "/media/robbis/DATA/fmri/"
analysis = 'bunch-of-things'
conf_file = make_analysis(path, analysis)

loader = DataLoader(configuration_file=conf_file, 
                    loader='simulations',
                    task='simulations')

ds = loader.fetch(prepro=Transformer())
    
_default_options = {
                       'sample_slicer__targets' : [
                           ['LH', 'RH'], 
                           #['LF', 'RF'], 
                           #['LH', 'RH', 'LF', 'RF']
                        ],

                       'estimator__clf': [
                           SVC(C=1, kernel='linear', probability=True),
                           SVC(C=1, gamma=1, kernel='rbf', probability=True),
                           LinearDiscriminantAnalysis(),
                           QuadraticDiscriminantAnalysis(),
                           GaussianProcessClassifier(1 * RBF(1.))

                       ],                          
                       #'estimator__fsel__k':np.arange(50, 100, 5),
                    }    
    
_default_config = {
               
                       'prepro':['sample_slicer', 'balancer'],
                       'balancer__attr':'all',

                       'estimator': [('fsel', SelectKBest(k=150)),
                                     ('clf', SVC(C=1, kernel='linear'))],

                       'cv': StratifiedKFold,
                       'cv__n_splits': 5,
                       #'cv__test_size': 0.25,

                       'scores' : ['accuracy'],

                       'analysis': TemporalDecoding,
                       'analysis__n_jobs': 8,
                       'analysis__permutation': 0,
                       'analysis__verbose': 0,
                       'kwargs__roi': ['matrix_values'],
                       #'kwargs__cv_attr':'subjects',

                    }
 
estimators = []
iterator = AnalysisIterator(_default_options, AnalysisConfigurator, config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="prova").fit(ds, **kwargs)
    a.save(save_estimator=True)
    est = a._estimator.scores['mask-matrix_values_value-1.0'][0]['estimator']
    estimators.append(est)
    del a


#########################
# Prediction timecourse #

# Load second session
ds_session = SampleSlicer(targets=['LH', 'RH', 'LF', 'RF']).transform(ds)

X = ds_session.samples
y = ds_session.targets

colormap = {'LH':'navy', 'RH':'firebrick', 'LF':'cornflowerblue', 'RF':'salmon'}
colors = [colormap[t] for t in y]


fig, axes = pl.subplots(len(estimators), 2)
for e, estimator in enumerate(estimators):
    avg_scores = [est.predict_proba(X) for est in estimator]
    avg_scores = np.mean(avg_scores, axis=0)
    for i in range(len(avg_scores.shape[-1])):
        score = avg_scores[...,i]
        for c, s in zip(colors, score):
            axes[e, i].set_title(str(estimator[0].estimators_[0]['clf'])+" class: "+str(i+1))
            axes[e, i].plot(np.diag(s), c=c, alpha=0.3)

