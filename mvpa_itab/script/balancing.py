from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVR, SVC

from pyitab.io.loader import DataLoader
from pyitab.io.bids import load_bids_dataset
from pyitab.preprocessing.pipelines import PreprocessingPipeline, \
    StandardPreprocessingPipeline
from pyitab.analysis.searchlight import SearchLight

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.configurator import AnalysisConfigurator

from imblearn.over_sampling import SMOTE
import numpy as np

from imblearn.under_sampling import *
from imblearn.over_sampling import *

conf_file = "/home/carlos/mount/megmri03/fmri/carlo_ofp/ofp.conf"
conf_file = "/media/robbis/DATA/fmri/carlo_ofp/ofp.conf"
#conf_file = "/home/carlos/fmri/carlo_ofp/ofp_new.conf"

if conf_file[1] == 'h':
    from mvpa_itab.utils import enable_logging
    root = enable_logging()


loader = DataLoader(configuration_file=conf_file, task='OFP')
ds = loader.fetch()


return_ = True
ratio = 'auto'

_default_options = {
                       'sample_slicer__evidence' : [[1]],
                       'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                       'balancer__balancer': [AllKNN(return_indices=return_, ratio=ratio),
                                              CondensedNearestNeighbour(return_indices=return_, ratio=ratio),
                                              EditedNearestNeighbours(return_indices=return_, ratio=ratio),
                                              InstanceHardnessThreshold(return_indices=return_, ratio=ratio),
                                              NearMiss(return_indices=return_, ratio=ratio),
                                              OneSidedSelection(return_indices=return_, ratio=ratio),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=5),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=15),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=25),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=35),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=45),
                                              RepeatedEditedNearestNeighbours(return_indices=return_, ratio=ratio),
                                              TomekLinks(return_indices=return_, ratio=ratio),
                                              #SMOTE(),
                                              #RandomOverSampler(),
                                            ],                          
                       'cv__n_splits': [7],
                       'analysis__radius':[9],
                        }


_default_config = {
               
                       'prepro':[
                                 'sample_slicer', 
                                 'feature_norm', 
                                 'target_trans', 
                                 'balancer'
                                 ],
                       'target_trans__target':"decision",
                       'balancer__attr':'subject',
                       'imbalancer__attr':'subject',
                       'estimator': [('clf', SVC(C=1, kernel='linear'))],
                       'estimator__clf__C':1,
                       'estimator__clf__kernel':'linear',

                       'cv': StratifiedKFold,
                       'cv__n_splits': 5,

                       'scores' : ['accuracy'],

                       'analysis': SearchLight,
                       'analysis__n_jobs': 15,
                       'cv_attr': 'subject'

                    }


iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
#conf = iterator.next()

for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="permut1_balancer_7").fit(ds, **kwargs)
    a.save()


#############################################################
# Haxby dataset #

conf_file = "/media/robbis/DATA/fmri/ds105/ds105.conf"


loader = DataLoader(configuration_file=conf_file, 
                    task='objectviewing',
                    loader=load_bids_dataset, 
                    prepro=StandardPreprocessingPipeline())

ds = loader.fetch()


return_ = True
ratio = 'auto'

_default_options = {
                       'imbalancer__sampling_strategy' : [0.1, 0.2, 0.3, 0.4, 0.5, 1.],
                       'balancer__balancer': [
                                              AllKNN(return_indices=return_, sampling_strategy=ratio),
                                              CondensedNearestNeighbour(return_indices=return_, sampling_strategy=ratio),
                                              EditedNearestNeighbours(return_indices=return_, sampling_strategy=ratio),
                                              InstanceHardnessThreshold(return_indices=return_, sampling_strategy=ratio),
                                              NearMiss(return_indices=return_, 
                                                       sampling_strategy=ratio,
                                                       n_neighbors=2,
                                                       ),
                                              NearMiss(return_indices=return_, 
                                                       sampling_strategy=ratio,
                                                       n_neighbors=2, 
                                                       version=2,),
                                              NearMiss(return_indices=return_, 
                                                       sampling_strategy=ratio,
                                                       n_neighbors_ver3=2, 
                                                       version=3),                                              
                                              OneSidedSelection(return_indices=return_, sampling_strategy=ratio),
                                              RandomUnderSampler(return_indices=return_, sampling_strategy=ratio, random_state=45),
                                              RepeatedEditedNearestNeighbours(return_indices=return_, sampling_strategy=ratio),
                                              NeighbourhoodCleaningRule(return_indices=return_, sampling_strategy=ratio, n_neighbors=2),
                                              TomekLinks(return_indices=return_, ratio=ratio),
                                              #SMOTE(),
                                              #RandomOverSampler(),
                                            ],                          
                       #'cv__n_splits': [5],
                       'sample_slicer__name': [[s] for s in np.unique(ds.sa.name)],
                       'analysis__radius':[9],
                        }



_default_config = {
               
                       'prepro':[
                                 'sample_slicer', 
                                 #'feature_normalizer', 
                                 'imbalancer', 
                                 'balancer'
                                 ],
                       'sample_slicer__targets':["face", "house"],
                       'balancer__attr':'chunks',
                       'imbalancer__attr': 'chunks',
                       'estimator': [('clf', SVC(C=1, kernel='linear'))],
                       'estimator__clf__C':1,
                       'estimator__clf__kernel':'linear',

                       'cv': GroupShuffleSplit,
                       #'cv__n_splits': 5,

                       'scores' : ['accuracy'],

                       'analysis': SearchLight,
                       'analysis__n_jobs': -1,
                       'cv_attr': 'chunks'

                    }

from pyitab.analysis import run_analysis
"""run_analysis(ds, _default_config, _default_options, 
             name="ds105_balancer", subdir="derivatives/mvpa")"""

name="ds105_balancer"
subdir="derivatives/mvpa"

iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config))

errs = [] 
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name=name).fit(ds, **kwargs)
        a.save(subdir=subdir)
    except Exception as err:
        errs.append([conf._default_options.copy(), err])
    

    
    

