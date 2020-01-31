#cd /home/carlos/git/mvpa_itab_wu/

from mvpa_itab.io.loader import DataLoader
from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline
from sklearn.model_selection import *
from mvpa_itab.pipeline.searchlight import SearchLight
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVR
from mvpa_itab.pipeline.iterator import AnalysisIterator
from mvpa_itab.pipeline.analysis import AnalysisPipeline
from mvpa_itab.pipeline.script import ScriptConfigurator
import os


import logging

from mvpa_itab.utils import enable_logging
root = enable_logging()
#root.setLevel(logging.DEBUG)

loader = DataLoader(configuration_file="/home/carlos/mount/megmri03/monks/meditation_permut1.conf", task='meg')
ds = loader.fetch()

import numpy as np

_default_options = {
                       'sample_slicer__condition' : [[c] for c in np.unique(ds.sa.condition)],
                       'sample_slicer__band': [[c] for c in np.unique(ds.sa.band)],
                       'target_trans__target':["age"],
                       'estimator__clf__C': [1],                          
                       'cv__n_splits': [50],
                       'analysis__radius':[9.],
                        }


_default_config = {
               
                        'prepro':['sample_slicer', 'feature_norm', 'target_trans'],
                        'sample_slicer__band': ['alpha'], 
                        'sample_slicer__condition' : ['vipassana'],
                        'target_trans__target':"expertise_hours",
                        
                        'estimator': [('clf', SVR(C=1, kernel='linear'))],
                        'estimator__clf__C':1,
                        'estimator__clf__kernel':'linear',
                        
                        'cv': ShuffleSplit,
                        'cv__n_splits': 50,
                        'cv__test_size': 0.25,
                        
                        'scores' : ['neg_mean_squared_error','r2'],
                        
                        'analysis': SearchLight,
                        'analysis__n_jobs': 15,
                        'analysis__permutation':100,
                        'kwargs__cv_attr': 'subject',
                        'analysis__verbose':0,

                    }



iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))


for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="permut1_regression").fit(ds, **kwargs)
    a.save(save_cv=False)
    command = 'mv %s /home/carlos/fmri/monks/0_results/' % (a._path)
    os.system(command)


######################## Stats ###################################


    3dANOVA3 -type 5                            \
        -alevels 2                              \
        -blevels 3                              \
        -clevels 2                              \
        -dset 1 1 1 man1_houses+tlrc            \
        -dset 1 2 1 man1_faces+tlrc             \
        -dset 1 3 1 man1_donuts+tlrc            \
        -dset 1 1 2 man2_houses+tlrc            \
        -dset 1 2 2 man2_faces+tlrc             \
        -dset 1 3 2 man2_donuts+tlrc            \
        -dset 2 1 1 woman1_houses+tlrc          \
        -dset 2 2 1 woman1_faces+tlrc           \
        -dset 2 3 1 woman1_donuts+tlrc          \
        -dset 2 1 2 woman2_houses+tlrc          \
        -dset 2 2 2 woman2_faces+tlrc           \
        -dset 2 3 2 woman2_donuts+tlrc          \
        -adiff   1 2           MvsW             \
        -bdiff   2 3           FvsD             \
        -bcontr -0.5 1 -0.5    FvsHD            \
        -aBcontr 1 -1 : 1      MHvsWH           \
        -aBdiff  1  2 : 1      same_as_MHvsWH   \
        -Abcontr 2 : 0 1 -1    WFvsWD           \
        -Abdiff  2 : 2 3       same_as_WFvsWD   \
        -Abcontr 2 : 1 7 -4.2  goofy_example    \
        -bucket donut_anova




