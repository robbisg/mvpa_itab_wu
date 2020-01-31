from pyitab.io.loader import DataLoader
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from sklearn.svm.classes import SVC
from sklearn.model_selection import *
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, \
    SampleSlicer, TargetTransformer, Transformer, \
    TemporalTransformer
from pyitab.preprocessing.normalizers import SampleSigmaNormalizer, \
    FeatureZNormalizer, SampleZNormalizer
from imblearn.under_sampling import RandomUnderSampler
#from pyitab.utils.files import log_memory
from pyitab.preprocessing.memory import MemoryReducer
from sentry_sdk import capture_exception

import glob
import os
import numpy as np

######################################
# Only when running on permut1
from pyitab.utils import enable_logging
root = enable_logging()
#####################################

#conf_file =  "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"
conf_file = "/home/carlos/fmri/carlo_mdm/memory.conf"

roi_labels_fname = glob.glob('/home/carlos/fmri/carlo_mdm/1_single_ROIs/*mask.nii.gz')
#roi_labels_fname = glob.glob('/home/robbis/mount/permut1/fmri/carlo_mdm/1_single_ROIs/*mask.nii.gz')
roi_labels_fname = glob.glob('/media/robbis/DATA/fmri/carlo_mdm/1_single_ROIs/*mask.nii.gz')
roi_labels = {os.path.basename(fname).split('_')[0]:fname for fname in roi_labels_fname}

loader = DataLoader(configuration_file=conf_file, 
                    event_file='residuals_attributes_full',
                    roi_labels=roi_labels,
                    task='RESIDUALS_MVPA')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      #Detrender(attr='file'),
                                      Detrender(attr='chunks'), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer(),
                                      SampleSlicer(frame=[1,2,3,4,5,6,7]),
                                      #TargetTransformer(attr='decision'),
                                      MemoryReducer(dtype=np.float16),
                                      #Balancer(attr='frame'),
                                      ])

ds = loader.fetch(prepro=prepro, n_subjects=8)

ds = MemoryReducer(dtype=np.float16).transform(ds)

labels = list(roi_labels.keys())[:-1]

import sentry_sdk  
sentry_sdk.init("https://f2866916959e41bc81abdfaf580f3d26@sentry.io/1439199") 

_default_options =  [{
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"]},                  
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 40, "O": 40}, return_indices=True),
                     },
               
                      {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr':{'decision': ["N", "O"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 40, "O": 40}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr':{'motor_resp': ["P", "S"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 40, "S": 40}, return_indices=True),
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr':{'target_side': ["L", "R"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 40, "R": 40}, return_indices=True),
                     },   
                    ]


_default_config = {
                    'prepro': ['target_transformer', 'sample_slicer', 'balancer'],
                    "balancer__attr": ['subject', 'frame'],
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': TemporalDecoding,
                    'analysis__n_jobs': 4,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    'kwargs__roi': labels,
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    'kwargs__cv_attr': 'subject'

                    }


errs = []
import gc
filtered_options = _default_options[1:]

iterator = AnalysisIterator(filtered_options, 
                            AnalysisConfigurator(**_default_config), 
                            kind='configuration')
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name="temporal_decoding").fit(ds, **kwargs)
        a.save()
        gc.collect()
    except Exception as err:
        errs.append([conf._default_options, err])
        capture_exception(err)


########################## Results #######################################
from pyitab.analysis.results.bids import get_results_bids

path = '/media/robbis/DATA/fmri/carlo_mdm/derivatives/'
path = '/home/robbis/mount/permut1/fmri/carlo_mdm/derivatives/'

dataframe = get_results_bids(path=path,  
                             pipeline="temporal+decoding",
                             field_list=['sample_slicer'], 
                             )

tasks = np.unique(dataframe['attr'].values)
masks = np.unique(dataframe['mask'].values)

for task in tasks:
    for mask in masks:

        df = filter_dataframe(dataframe, attr=[task], mask=[mask])
        df_diagonal = df_fx_over_keys(dataframe=df, 
                                keys=['value'], 
                                attr='score_score', 
                                fx=lambda x: np.diagonal(np.mean(np.dstack(x), axis=2)))
        df_exploded = df_diagonal.explode('score_score')
        n_roi = len(np.unique(df_diagonal['value']))
        frames = np.hstack([np.arange(7)+1 for _ in range(n_roi)])

        df_exploded['value'] = np.int_(df_exploded['value'])
        df_exploded['frame'] = frames
        rois = [big_table[mask][value-1][3]+" "+str(value) for value in df_exploded['value'].values]
        df_exploded['roi'] = rois

        #pl.figure()
        grid = sns.FacetGrid(df_exploded, col="roi", hue="value", col_wrap=4, height=1.5)
        grid.map(pl.axhline, y=0.5, ls=":", c=".5")
        grid.map(pl.plot, "frame", "score_score", marker="o")
        grid.set(yticks=[.45, .5, .55, .6])
        figname = "/media/robbis/DATA/fmri/carlo_mdm/derivatives/temporal+decoding_task-%s_mask-%s.png" %(task, mask)
        grid.savefig(figname, dpi=100)

        

###################################################
###################################################
##### Within- 
###################################################

ds = loader.fetch(prepro=prepro)

ds = MemoryReducer(dtype=np.float16).transform(ds)

labels = list(roi_labels.keys())[:-1]
log_memory()
import sentry_sdk  
sentry_sdk.init("https://f2866916959e41bc81abdfaf580f3d26@sentry.io/1439199") 

_default_options =  {
                        'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                        'sample_transformer__attr': [
                                                     {'image_type':["I", "O"]}, 
                                                     {'decision': ["N", "O"]},
                                                     {'motor_resp': ["P", "S"]},
                                                     {'target_side': ["L", "R"]}
                                                     ],
                        #'prepro':[['sample_slicer', 'sample_transformer'],
                        #          ['sample_slicer', 'sample_transformer','balancer']],

                        
                        #'cv': [StratifiedKFold, LeaveOneGroupOut],
                        #'cv__n_splits':[5, 7]
                    }



_default_config = {
                    'prepro': ['sample_slicer', 
                               'sample_transformer', 
                               #'balancer'
                               ],
                    #"balancer__attr": ['chunks', 'frame'],
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedKFold,
                    'cv__n_splits': 7,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': TemporalDecoding,
                    'analysis__n_jobs': -1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    'kwargs__roi': ['omnibus', 'decision', 'image+type', 'motor+resp', 'target+side'],
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    'kwargs__cv_attr': 'chunks'

                    }

import gc

errs = []

#filtered_options = _default_options[1:]

iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()

    a = AnalysisPipeline(conf, name="temporal_decoding_mdm").fit(ds, **kwargs)
    a.save()
    gc.collect()


####################################################

def avg_plot(**kwargs):
    from scipy.stats import ttest_1samp
    data = kwargs['data']

    n_frames = np.max(data['frame'].values)
    frame_list = np.arange(n_frames)+1

    df_avg = df_fx_over_keys(dataframe=data, 
                             keys=['frame'], 
                             attr='score_score', 
                             fx=np.mean)
    test_values = []
    for i in frame_list:
        df_frame = filter_dataframe(data, frame=[i])
        t, p = ttest_1samp(df_frame['score_score'].values, 0.5)
        test_values.append([t, p])

    test_values = np.array(test_values)
    
    sign_values = np.logical_and(test_values[:,1] < 0.05/49.,
                                 test_values[:,0] > 0)


    pl.plot(frame_list, df_avg['score_score'], 'o-', color='k')
    pl.plot(frame_list[sign_values], 
            df_avg['score_score'].values[sign_values], 'o', 
            color='salmon', markersize=5)




path = '/home/robbis/mount/permut1/fmri/carlo_mdm/derivatives/'

dataframe = get_results_bids(path=path,  
                             pipeline="temporal+decoding+across",
                             field_list=['sample_slicer'], 
                             )

attr = np.zeros_like(dataframe['mask'].values, dtype='U24')
for k in ['resp', 'side', 'decision', 'type']:
    mask_ = np.logical_not([isinstance(v, float) for v in dataframe[k].values])
    attr[mask_] = k
    dataframe=dataframe.drop(k, axis=1)

dataframe['attr'] = attr

tasks = np.unique(dataframe['attr'].values)
masks = np.unique(dataframe['mask'].values)

for task in tasks:
    for mask in masks:

        df = filter_dataframe(dataframe, attr=[task], mask=[mask])
        if df.size == 0:
            continue

        df_diagonal = df_fx_over_keys(dataframe=df, 
                                      keys=['value', 'fold'], 
                                      attr='score_score', 
                                      fx=lambda x: np.diagonal(np.mean(np.dstack(x), axis=2)))



        df_exploded = df_diagonal.explode('score_score')
        n_roi = len(np.unique(df_diagonal['value'])) * len(np.unique(df_diagonal['fold']))
        frames = np.hstack([np.arange(7)+1 for _ in range(n_roi)])

        df_exploded['value'] = np.int_(df_exploded['value'])
        df_exploded['frame'] = frames
        rois = [big_table[mask][value-1][3]+" "+str(value) for value in df_exploded['value'].values]
        df_exploded['roi'] = rois

        #pl.figure()

        grid = sns.FacetGrid(df_exploded, col="roi", col_wrap=4, height=2.2, aspect=1.6)
        grid.map(pl.axhline, y=0.5, ls=":", c=".5")
        grid.map(pl.plot, "frame", "score_score", marker="o", alpha=0.1)
        grid.set(yticks=[.45, .5, .55, .6, .65])
        grid.map_dataframe(avg_plot)
        
        figname = "/media/robbis/DATA/fmri/carlo_mdm/derivatives/temporal+decoding+across+fsel_task-%s_mask-%s_sign-005.png" %(task, mask)
        grid.savefig(figname, dpi=100)




        df_matrix =  df_fx_over_keys(dataframe=df, 
                                      keys=['value', 'fold'], 
                                      attr='score_score', 
                                      fx=lambda x: np.mean(np.dstack(x), axis=2))
        df_matrix['value'] = np.int_(df_matrix['value'])
        rois = [big_table[mask][value-1][3]+" "+str(value) for value in df_matrix['value'].values]

        df_matrix['roi'] = rois
        grid = sns.FacetGrid(df_matrix, col="roi", col_wrap=4, height=2.2, aspect=1.6)
        grid.map_dataframe(imshow_plot)
        figname = "/media/robbis/DATA/fmri/carlo_mdm/derivatives/temporal+decoding+across+fsel_task-%s_mask-%s_sign-005_matrix.png" %(task, mask)
        grid.savefig(figname, dpi=100)


def imshow_plot(**kwargs):
    from scipy.stats import ttest_1samp
    dataframe = kwargs['data']
    #print(np.dstack(data['score_score'].values))
    data = np.dstack(dataframe['score_score'].values)

    matrix = np.mean(data, axis=2)
    t, p = ttest_1samp(data, 0.5, axis=2)

    sign_values = np.logical_and(p < 0.05/49.,
                                 t > 0)

    nonzero = np.nonzero(sign_values)
    
    pl.imshow(matrix, origin='lower', cmap=pl.cm.magma, vmin=0.5, vmax=0.6)
    pl.plot(nonzero[0], nonzero[1], 'o', color='green')
    pl.colorbar()



#############################################
# Across- time decoding

loader = DataLoader(configuration_file=conf_file, 
                    event_file='residuals_attributes_full',
                    roi_labels=roi_labels,
                    task='RESIDUALS_MVPA')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      #Detrender(attr='file'),
                                      Detrender(attr='chunks'), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer(),
                                      SampleSlicer(frame=[1,2,3,4,5,6,7]),
                                      TargetTransformer(attr='decision'),
                                      MemoryReducer(dtype=np.float16),
                                      TemporalTransformer(attr='frame'),
                                      ])

ds = loader.fetch(prepro=prepro)

ds = MemoryReducer(dtype=np.float16).transform(ds)

labels = list(roi_labels.keys())[:-1]

import sentry_sdk  
sentry_sdk.init("https://f2866916959e41bc81abdfaf580f3d26@sentry.io/1439199") 


_default_options =  [
                     #{
                     # 'target_transformer__attr': "image_type",
                     # 'sample_slicer__attr': {'image_type':["I", "O"]},                  
                     # 'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 40, "O": 40}, return_indices=True),
                     # 'kwargs__roi':['image+type']
                     #},
               
                      {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr':{'decision': ["N", "O"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 40, "O": 40}, return_indices=True),
                      'kwargs__roi':['decision', 'motor+resp']
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr':{'motor_resp': ["P", "S"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 40, "S": 40}, return_indices=True),
                      'kwargs__roi':['motor+resp', 'decision']
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr':{'target_side': ["L", "R"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 40, "R": 40}, return_indices=True),
                      'kwargs__roi':['target+side']
                     },   
                    ]



_default_config = {
                    'prepro': ['target_transformer', 'sample_slicer', 'balancer'],
                    'balancer__attr':'subject',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    #'cv__n_splits': 7,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': TemporalDecoding,
                    'analysis__n_jobs': -1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    #'kwargs__roi': [
                                     #'omnibus', 
                                     #'decision', 
                                     #'image+type', 
                                     #'motor+resp', 
                                     #'target+side'
                                     #],
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    'kwargs__cv_attr': 'subject'

                    }

import gc

errs = []

filtered_options = _default_options[:1]

iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config),
                            kind='configuration')
for conf in iterator:
    kwargs = conf._get_kwargs()

    a = AnalysisPipeline(conf, name="temporal_decoding_across").fit(ds, **kwargs)
    a.save()
    gc.collect()

###############################
# Resting state
from pyitab.analysis.connectivity.multivariate import TrajectoryConnectivity
from pyitab.preprocessing.connectivity import AverageEstimator 


loader = DataLoader(configuration_file=conf_file, 
                    #event_file='residuals_attributes_full',
                    roi_labels=roi_labels,
                    task='RESTING_STATE')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      #Detrender(attr='file'),
                                      Detrender(attr='chunks'), 
                                      #SampleZNormalizer(),
                                      #FeatureZNormalizer(),
                                      #SampleSlicer(frame=[1,2,3,4,5,6,7]),
                                      #TargetTransformer(attr='decision'),
                                      MemoryReducer(dtype=np.float16),
                                      #Balancer(attr='all'),
                                      ])

subjects = loader.get_subjects()


_default_options = {
                    
                    'fetch__subject_names':[[s] for s in subjects],
                    'sample_slicer__targets': [['task'], ['rest']],           
                    
                    
                    }


_default_config = { 
                    'loader__configuration_file':"/media/robbis/DATA/fmri/carlo_mdm/memory.conf",
                    #'loader__event_file':'residuals_attributes_full',
                    'loader__roi_labels':roi_labels,
                    'loader__task':'RESTING_STATE',
                    'fetch__prepro':prepro,



                    'prepro': ['sample_slicer'],
                    
                    'analysis': TrajectoryConnectivity,
                    'analysis__estimator': PreprocessingPipeline(nodes=[AverageEstimator(), 
                                                     FeatureZNormalizer()]),

                    "kwargs__roi":["decision"]

                    }


iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="connectivity").fit(**kwargs)
    a.save()