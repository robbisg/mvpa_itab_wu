import glob
from pyitab.io.loader import DataLoader
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.normalizers import SampleZNormalizer
from pyitab.analysis.decoding.regression import RoiRegression
from pyitab.ext.sklearn.feature_selection import pearsonr_score
from pyitab.analysis import AnalysisConfigurator, AnalysisIterator, AnalysisPipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import *
from sklearn.metrics import make_scorer
from sklearn.svm import SVC, SVR

from sklearn.metrics import r2_score, mean_squared_error

def pearsonr_error(y, y_pred):
    from scipy.stats import pearsonr
    return pearsonr(y, y_pred)[0]

def mse_error(y, y_pred):
    import numpy as np
    error = np.power(y - y_pred, 2)
    return np.mean(error)




conf_file = "/media/robbis/DATA/fmri/monks/meditation.conf"

matrix_list = glob.glob("/media/robbis/DATA/fmri/monks/061102chrwoo/fcmri/*.mat")
matrix_list = [m.split("/")[-1] for m in matrix_list]


for m in matrix_list:
    m = '20151103_132009_connectivity_filtered_first_filtered_after_each_run_no_gsr_findlab_fmri.mat'
    loader = DataLoader(configuration_file=conf_file, 
                        loader='mat',
                        task='fcmri',
                        atlas='findlab',
                        event_file=m[:-4]+".txt",
                        img_pattern=m)

    prepro = PreprocessingPipeline(nodes=[
                                        #Transformer(), 
                                        #Detrender(), 
                                        SampleZNormalizer(),
                                        #FeatureZNormalizer()
                                        ])
    #prepro = PreprocessingPipeline()


    ds = loader.fetch(prepro=prepro)

    _default_options = [
                        {
                            'prepro':['sample_slicer', 'target_transformer'],
                            'target_transformer__attr': 'expertise',
                            'sample_slicer__targets': ['Samatha']
                        },
                        {
                            'prepro':['sample_slicer', 'target_transformer'],
                            'target_transformer__attr': 'age',
                            'sample_slicer__targets': ['Samatha']
                        },
                        {
                            'prepro':['sample_slicer', 'target_transformer', 'sample_residual'],
                            'target_transformer__attr': 'expertise',
                            'sample_residual__attr': ['age'],
                            'sample_slicer__targets': ['Samatha']
                        },
                        {
                            'prepro':['sample_slicer', 'target_transformer', 'sample_residual'],
                            'target_transformer__attr': 'age',
                            'sample_residual__attr': ['expertise'],
                            'sample_slicer__targets': ['Samatha']
                        },
                        {
                            'prepro':['sample_slicer', 'target_transformer'],
                            'target_transformer__attr': 'expertise',
                            'sample_slicer__targets': ['Vipassana']
                        },
                        {
                            'prepro':['sample_slicer', 'target_transformer'],
                            'target_transformer__attr': 'age',
                            'sample_slicer__targets': ['Vipassana']
                        },
                        {
                            'prepro':['sample_slicer', 'target_transformer', 'sample_residual'],
                            'target_transformer__attr': 'expertise',
                            'sample_residual__attr': ['age'],
                            'sample_slicer__targets': ['Vipassana']
                        },
                        {
                            'prepro':['sample_slicer', 'target_transformer', 'sample_residual'],
                            'target_transformer__attr': 'age',
                            'sample_residual__attr': ['expertise'],
                            'sample_slicer__targets': ['Vipassana']
                        },
    ]
    
    _default_config = {
                
                        'prepro':['sample_slicer', 'target_transformer'],
                        'target_transformer__attr': 'age',
                        'sample_slicer__group':['E'],
                        'estimator': [('fsel', SelectKBest(k=80, score_func=pearsonr_score)),
                                      ('clf', SVR(C=1, kernel='linear'))],
                        
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel':'linear',

                        'cv': ShuffleSplit,
                        'cv__n_splits': 50,
                        'cv__test_size': 0.25,

                        'analysis': RoiRegression,
                        'analysis__n_jobs': -1,
                        'analysis__permutation': 0,
                        'analysis__verbose': 1,
                        'analysis__scoring':{ 
                                             'mse': make_scorer(mse_error),
                                             'corr': make_scorer(pearsonr_error)
                                             },
                        'kwargs__roi': ['matrix_values'],
                        #'kwargs__cv_attr':'name',

                        }
    
    
    iterator = AnalysisIterator(_default_options, 
                                AnalysisConfigurator(**_default_config),
                                kind='configuration')
    for i, conf in enumerate(iterator):
        kwargs = conf._get_kwargs()
        a = AnalysisPipeline(conf, "rev1_2_regression").fit(ds, **kwargs)
        a.save()
        del a

###########################################
conf_file = "/media/robbis/DATA/fmri/monks/meditation.conf"

matrix_list = glob.glob("/media/robbis/DATA/fmri/monks/061102chrwoo/fcmri/*.mat")
matrix_list = [m.split("/")[-1] for m in matrix_list]

from sentry_sdk import capture_exception
import sentry_sdk  
sentry_sdk.init("https://f2866916959e41bc81abdfaf580f3d26@sentry.io/1439199") 

for m in matrix_list:
    m = '20151103_132009_connectivity_filtered_first_filtered_after_each_run_no_gsr_findlab_fmri.mat'
    loader = DataLoader(configuration_file=conf_file, 
                        loader='mat',
                        task='fcmri',
                        atlas='findlab',
                        event_file=m[:-4]+".txt",
                        img_pattern=m)

    prepro = PreprocessingPipeline(nodes=[
                                        #Transformer(), 
                                        #Detrender(), 
                                        SampleZNormalizer(),
                                        #FeatureZNormalizer()
                                        ])
    #prepro = PreprocessingPipeline()


    ds = loader.fetch(prepro=prepro)

    _default_options = {
                            'sample_slicer__targets': [['Samatha'], ['Vipassana']],
                            'target_transformer__attr': ['age', 'expertise_hours']
                        }
    

    _default_config = {
                
                        'prepro':['sample_slicer', 'target_transformer'],
                        'target_transformer__attr': 'age',
                        'sample_slicer__group':['N'],
                        'estimator': [('fsel', SelectKBest(k=80, score_func=pearsonr_score)),
                                      ('clf', SVR(C=1, kernel='linear'))],
                        
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel':'linear',

                        'cv': GroupShuffleSplit,
                        'cv__n_splits': 50,
                        'cv__test_size': 0.25,

                        'analysis': RoiRegression,
                        'analysis__n_jobs': -1,
                        'analysis__permutation': 500,
                        'analysis__verbose': 1,
                        'analysis__scoring':{ 
                                             'mse': make_scorer(mse_error),
                                             'corr': make_scorer(pearsonr_error)
                                             },
                        'kwargs__roi': ['matrix_values'],
                        'kwargs__cv_attr':'subject',

                        }
    
    
    iterator = AnalysisIterator(_default_options, 
                                AnalysisConfigurator(**_default_config),
                                #kind='configuration')
                                )

    for i, conf in enumerate(iterator):
        kwargs = conf._get_kwargs()
        a = AnalysisPipeline(conf, "rev2_novice").fit(ds, **kwargs)            
        try:
            a.save()
        except Exception as err:
            capture_exception(err)
        del a