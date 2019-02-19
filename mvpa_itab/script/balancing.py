from mvpa_itab.io.loader import DataLoader
from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline
from sklearn.model_selection import *
from mvpa_itab.pipeline.searchlight import SearchLight
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVR, SVC
from mvpa_itab.pipeline.iterator import AnalysisIterator
from mvpa_itab.pipeline.decoding import AnalysisPipeline
from mvpa_itab.pipeline.script import ScriptConfigurator
from imblearn.over_sampling.smote import SMOTE
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


iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))
#conf = iterator.next()

for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="permut1_balancer_7").fit(ds, **kwargs)
    a.save()


#############################################################
# Haxby dataset #

conf_file = "/media/robbis/DATA/fmri/haxby2001/ofp.conf"


loader = DataLoader(configuration_file="/media/robbis/DATA/fmri/haxby2001/ofp.conf", task='haxby', prepro=StandardPreprocessingPipeline())
ds = loader.fetch()


return_ = True
ratio = 'auto'

_default_options = {
                       'imbalancer__ratio' : [0.3, 0.4, 0.5, 0.6, 0.7],
                       'balancer__balancer': [
                                              #AllKNN(return_indices=return_, ratio=ratio),
                                              #CondensedNearestNeighbour(return_indices=return_, ratio=ratio),
                                              #EditedNearestNeighbours(return_indices=return_, ratio=ratio),
                                              #InstanceHardnessThreshold(return_indices=return_, ratio=ratio),
                                              #NearMiss(return_indices=return_, ratio=ratio),
                                              #OneSidedSelection(return_indices=return_, ratio=ratio),
                                              #RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=5),
                                              #RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=15),
                                              #RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=25),
                                              #RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=35),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=45),
                                              #RepeatedEditedNearestNeighbours(return_indices=return_, ratio=ratio),
                                              #TomekLinks(return_indices=return_, ratio=ratio),
                                              #SMOTE(),
                                              #RandomOverSampler(),
                                            ],                          
                       'cv__n_splits': [5],
                       'analysis__radius':[9],
                        }


_default_config = {
               
                       'prepro':[
                                 'sample_slicer', 
                                 'feature_norm', 
                                 'imbalancer', 
                                 'balancer'
                                 ],
                       'sample_slicer__targets':["face","house"],
                       'balancer__attr':'subject',
                       'estimator': [('clf', SVC(C=1, kernel='linear'))],
                       'estimator__clf__C':1,
                       'estimator__clf__kernel':'linear',

                       'cv': StratifiedKFold,
                       'cv__n_splits': 5,

                       'scores' : ['accuracy'],

                       'analysis': SearchLight,
                       'analysis__n_jobs': 1,
                       'cv_attr': 'subject'

                    }


iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))
conf = iterator.next()

for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="permut1_imbalancer_7").fit(ds, **kwargs)
    a.save()
