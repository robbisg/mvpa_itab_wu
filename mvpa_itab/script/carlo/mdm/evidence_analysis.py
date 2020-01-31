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
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.analysis.searchlight import SearchLight
from pyitab.io.connectivity import load_mat_ds
from sklearn.feature_selection.univariate_selection import SelectKBest
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

roi_labels_fname = glob.glob('/home/carlos/fmri/carlo_mdm/1_single_ROIs/*mask.nii.gz')
roi_labels = {os.path.basename(fname).split('_')[0]:fname for fname in roi_labels_fname}

configuration_file = "/home/carlos/fmri/carlo_mdm/memory.conf"
#configuration_file = "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"

loader = DataLoader(configuration_file=configuration_file, 
                    #data_path="/home/carlos/mount/meg_workstation/Carlo_MDM/",
                    task='BETA_MVPA',
                    roi_labels=roi_labels,
                    event_file="beta_attributes_full", 
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




_default_options =  [
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                             ('image+type', [4]), ('image+type', [5])],
                     },
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),        
                      'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                             ('image+type', [4]), ('image+type', [5])],
                     },
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[5]},                 
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                             ('image+type', [4]), ('image+type', [5])],
                     },
                     
                     #####################################

                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr': {'target_side':["L", "R"],'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 20, "R": 20}, return_indices=True),
                      'kwargs__roi_values': [('target+side', [1]), ('target+side', [2]), ('target+side', [3]),
                                             ('target+side', [4]), ('target+side', [5])],
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr': {'target_side':["L", "R"],'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 20, "R": 20}, return_indices=True),        
                      'kwargs__roi_values': [('target+side', [1]), ('target+side', [2]), ('target+side', [3]),
                                             ('target+side', [4]), ('target+side', [5])],
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr': {'target_side':["L", "R"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 20, "R": 20}, return_indices=True),         
                      'kwargs__roi_values': [('target+side', [1]), ('target+side', [2]), ('target+side', [3]),
                                             ('target+side', [4]), ('target+side', [5])],
                     },

                     ################################################################
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),       
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     #######################################################################
                     {         
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),              
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     }
                    ]




_default_config = {
                    'prepro': ['target_transformer', 'sample_slicer', 'balancer'],
                    "balancer__attr": 'subject',
                    
                    'estimator': [
                        ('fsel', SelectKBest(k=50)),
                        ('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': RoiDecoding,
                    'analysis__n_jobs': -1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    #'kwargs__roi': labels,
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    'kwargs__cv_attr': 'subject'
                    }

import gc 
iterator = AnalysisIterator(_default_options,  
                            AnalysisConfigurator(**_default_config), 
                            kind='configuration') 
for conf in iterator: 
    kwargs = conf._get_kwargs() 
 
    a = AnalysisPipeline(conf, name="roi_decoding_across_full").fit(ds, **kwargs) 
    a.save() 
    gc.collect()

################################
path = '/home/robbis/mount/permut1/fmri/carlo_mdm/derivatives/'

def imshow_plot(**kwargs):
    data = kwargs['data']
    print(data)

dataframe = get_results_bids(path=path,  
                             pipeline="roi+decoding+across+full",
                             field_list=['sample_slicer', 'ds.sa.evidence'], 
                             )

dataframe['evidence'] = np.int_([s[1:-1] for s in dataframe['ds.sa.evidence'].values])
grid = sns.FacetGrid(dataframe, row="attr", col="mask", hue="value",
                     height=3.5)
grid.map(pl.plot, "evidence", "score_score", marker="o")

grid.map_dataframe(avg_plot)

rushmore = ["#E1BD6D", "#EABE94", "#0B775E", "#35274A", "#F2300F"]
palette = sns.color_palette(rushmore)


fig = pl.figure(figsize=(15,15))

for i, task in enumerate(tasks):
    df = filter_dataframe(dataframe, attr=[task], mask=[task])
    ax = fig.add_subplot(2,2,i+1)

    data = df['score_score'].values
    evidences = df['evidence'].values
    colors = df['value'].values
    for j, mask in enumerate(np.unique(colors)):
        df_roi = filter_dataframe(df, value=[mask])
        df_avg = df_fx_over_keys(df_roi, attr="score_score", keys=['evidence'], fx=np.mean)
        #ax.scatter(evidences+(0.02*j), data, alpha=0.8, c=np.array([palette[j]]))
        ax.plot(df_avg['evidence'].values, 
                df_avg['score_score'].values,
                marker='o',
                markersize=12, 
                linewidth=3, 
                color=palette[j])
    ax.set_ylim(0.49)







for task in tasks:

    mask = task

    df = filter_dataframe(dataframe, attr=[task], mask=[mask])

    if df.size==0:
        continue
    
    for roi in np.unique(df['value'].values):
        df_roi = filter_dataframe(df, value=[roi])
        variables = np.array([df_roi[key].values for key in ['evidence', 'fold']]).T
        X, factors, n_factor = design_matrix(variables)  
        evidence = (np.int_(df_roi['evidence'].values[:,np.newaxis]) - 3)
        X_mixed = np.hstack((X, evidence))
        X_mixed = X_mixed[:,3:]
        y = (df_roi["score_score"].values - .5)
        res_omnibus = sm.OLS(y, X).fit()
        res_linear = sm.OLS(y, evidence.flatten()).fit()
        res_mixed = sm.OLS(y, X_mixed).fit()

        contrast_omnibus = build_contrast(n_factor, 0, const_value=0.)
        #n_factor_mixed = np.append(n_factor, 1)
        #contrast_mixed = build_contrast(n_factor_mixed, 0, const_value=0.5)


        t_contrast_mixed = np.zeros_like(X_mixed[0])
        t_contrast_mixed[-1] = 1

        t_contrast_linear = [1]


        test_omnibus = res_omnibus.f_test(contrast_omnibus)
        #test_mixed = res_mixed.f_test(contrast_mixed)


        test_linear_mixed = res_mixed.t_test(t_contrast_mixed)
        test_linear_linear = res_linear.t_test(t_contrast_linear)

        df_level_1 = filter_dataframe(df_roi, evidence=[1])
        t, p = ttest_1samp(df_level_1['score_score'].values, .5)

        r = {'roi':roi,
            'task':task,
            'label':big_table[task][int(roi)-1],
            'p_omnibus_plain':test_omnibus.pvalue,
            'objects':[test_omnibus, test_linear_linear, test_linear_mixed, t],
            #'p_omnibus_mixed':test_mixed.pvalue,
            'p_linear_plain' :test_linear_linear.pvalue,
            'p_linear_mixed' :test_linear_mixed.pvalue,
            'p_1x_vs_chance': p,
            }

        results.append(r)



######################################
loader = DataLoader(configuration_file=configuration_file, 
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

_default_options =  [
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 40, "O": 40}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                             ('image+type', [4]), ('image+type', [5])],
                     },
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"], 'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),        
                      'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                             ('image+type', [4]), ('image+type', [5])],
                     },
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"]},
                      'sample_slicer__attr': {'evidence':[5]},                 
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                             ('image+type', [4]), ('image+type', [5])],
                     },
                     
                     #####################################

                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr': {'target_side':["L", "R"],'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 20, "R": 20}, return_indices=True),
                      'kwargs__roi_values': [('target+side', [1]), ('target+side', [2]), ('target+side', [3]),
                                             ('target+side', [4]), ('target+side', [5])],
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr': {'target_side':["L", "R"],'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 20, "R": 20}, return_indices=True),        
                      'kwargs__roi_values': [('target+side', [1]), ('target+side', [2]), ('target+side', [3]),
                                             ('target+side', [4]), ('target+side', [5])],
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr': {'target_side':["L", "R"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 20, "R": 20}, return_indices=True),         
                      'kwargs__roi_values': [('target+side', [1]), ('target+side', [2]), ('target+side', [3]),
                                             ('target+side', [4]), ('target+side', [5])],
                     },

                     ################################################################
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),       
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 20, "O": 20}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     #######################################################################
                     {         
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[1]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[3]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),              
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"], 'evidence':[5]},
                      #'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 20, "S": 20}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     }
                    ]



_default_config = {
                    'prepro': [
                               'target_transformer',
                               'sample_slicer', 
                               #'balancer'
                               ],
                    #"balancer__attr": 'all',
                    
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

                    #'kwargs__roi': ['omnibus', 'decision', 'image+type', 'motor+resp', 'target+side'],
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    #'kwargs__cv_attr': 'chunks'

                    }



iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config),
                            kind='configuration')
for conf in iterator:
    kwargs = conf._get_kwargs()
    for s in np.unique(ds.sa.subject):


        conf._default_options['sample_slicer__attr'].update({'subject':[s]})
        #conf._default_config.update({'sample_slicer__attr': {'subject':[s]}})

        #print(conf._default_options)
        a = AnalysisPipeline(conf, name="temporal_decoding_mdm").fit(ds, **kwargs)
        a.save()
        gc.collect()



def imshow_plot(**kwargs):
    from scipy.stats import ttest_1samp
    dataframe = kwargs['data']
    #print(np.dstack(data['score_score'].values))
    data = np.dstack(dataframe['score_score'].values)

    matrix = np.mean(data, axis=2)
    #t, p = ttest_1samp(data, 0.5, axis=2)

    #sign_values = np.logical_and(p < 0.01/49.,
    #                             t > 0)

    #nonzero = np.nonzero(sign_values)
    
    pl.imshow(matrix, origin='lower', cmap=pl.cm.magma, vmin=0.5, vmax=0.6)
    #pl.plot(nonzero[0], nonzero[1], 'o', color='green')
    pl.colorbar()





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
    
    sign_values = np.logical_and(test_values[:,1] < 0.01/49.,
                                 test_values[:,0] > 0)


    pl.plot(frame_list, df_avg['score_score'], 'o-', color='k')
    """
    pl.plot(frame_list[sign_values], 
            df_avg['score_score'].values[sign_values], 'o', 
            color='salmon', markersize=5)
    """



path = '/home/robbis/mount/permut1/fmri/carlo_mdm/derivatives/'

dataframe = get_results_bids(path=path,  
                             pipeline="temporal+decoding+across",
                             field_list=['sample_slicer'], 
                             )


tasks = np.unique(dataframe['attr'].values)
masks = np.unique(dataframe['mask'].values)

for task in tasks:

    mask = task

    df = filter_dataframe(dataframe, attr=[task], mask=[mask])
    if df.size == 0:
        continue

    df_diagonal = df_fx_over_keys(dataframe=df, 
                                    keys=['value', 'evidence'], 
                                    attr='score_score', 
                                    fx=lambda x: np.diagonal(np.mean(np.dstack(x), axis=2)))



    df_exploded = df_diagonal.explode('score_score')
    n_roi = len(np.unique(df_diagonal['value'])) \
            * len(np.unique(df_diagonal['evidence']))
    frames = np.hstack([np.arange(7)+1 for _ in range(n_roi)])

    df_exploded['value'] = np.int_(df_exploded['value'])
    df_exploded['frame'] = frames
    rois = [big_table[mask][value-1][3]+" "+str(value) for value in df_exploded['value'].values]
    df_exploded['roi'] = rois

    #pl.figure()

    grid = sns.FacetGrid(df_exploded, 
                         col="roi", 
                         hue="evidence", 
                         palette="magma", 
                         col_wrap=3, 
                         height=2.2, 
                         aspect=1.6)
    grid.map(pl.axhline, y=0.5, ls=":", c=".5")
    grid.map(pl.plot, "frame", "score_score", marker="o").add_legend()
    grid.set(yticks=[.45, .5, .55, .6, .65, .7, .75, .8])
    #grid.map_dataframe(avg_plot)
    
    figname = "/media/robbis/DATA/fmri/carlo_mdm/derivatives/temporal+decoding+evidence_task-%s_mask-%s.png" %(task, mask)
    grid.savefig(figname, dpi=100)




    df_matrix =  df_fx_over_keys(dataframe=df, 
                                    keys=['value', 'evidence'], 
                                    attr='score_score', 
                                    fx=lambda x: np.mean(np.dstack(x), axis=2))
    df_matrix['value'] = np.int_(df_matrix['value'])
    rois = [big_table[mask][value-1][3]+" "+str(value) for value in df_matrix['value'].values]

    df_matrix['roi'] = rois
    grid = sns.FacetGrid(df_matrix, col="roi", row="evidence", height=2.2, aspect=1.6)
    grid.map_dataframe(imshow_plot)
    figname = "/media/robbis/DATA/fmri/carlo_mdm/derivatives/temporal+decoding+evidence_task-%s_mask-%s-matrix.png" %(task, mask)
    grid.savefig(figname, dpi=100)


#####################################################################################
_default_options =  [
                     {
                      'target_transformer__attr': "image_type",
                      'sample_slicer__attr': {'image_type':["I", "O"]},              
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"I": 40, "O": 40}, return_indices=True),
                      'kwargs__roi_values': [('image+type', [1]), ('image+type', [2]), ('image+type', [3]),
                                             ('image+type', [4]), ('image+type', [5])],
                     },
                     {
                      'target_transformer__attr': "target_side",
                      'sample_slicer__attr': {'target_side':["L", "R"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"L": 40, "R": 40}, return_indices=True),         
                      'kwargs__roi_values': [('target+side', [1]), ('target+side', [2]), ('target+side', [3]),
                                             ('target+side', [4]), ('target+side', [5])],
                     },
                     {
                      'target_transformer__attr': "decision",
                      'sample_slicer__attr': {'decision':["N", "O"],},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"N": 40, "O": 40}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     },
                     {
                      'target_transformer__attr': "motor_resp",
                      'sample_slicer__attr': {'motor_resp':["P", "S"]},
                      'balancer__balancer': RandomUnderSampler(sampling_strategy={"P": 40, "S": 40}, return_indices=True),          
                      'kwargs__roi_values': [('decision', [1]), ('decision', [2]), ('decision', [3]),
                                             ('decision', [4]), ('decision', [5]),
                                             ('motor+resp', [1]), ('motor+resp', [2]), ('motor+resp', [3]),
                                             ('motor+resp', [4]), ('motor+resp', [5])
                                             ],
                     }
                    ]



_default_config = {
                    'prepro': [
                               'target_transformer',
                               'sample_slicer', 
                               'balancer'
                               ],
                    "balancer__attr": 'subject',
                    
                    'estimator': [
                        ('fsel', SelectKBest(k=50)),
                        ('clf', SVC(C=1, kernel='linear'))
                        ],
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

                    #'kwargs__roi': ['omnibus', 'decision', 'image+type', 'motor+resp', 'target+side'],
                    #'kwargs__roi_values': [('image+type', [2])],
                    #'kwargs__prepro': ['feature_normalizer', 'sample_normalizer'],
                    'kwargs__cv_attr': 'subject'

                    }



iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config),
                            kind='configuration')
for conf in iterator:
    kwargs = conf._get_kwargs()

    a = AnalysisPipeline(conf, name="temporal_decoding_across_fsel").fit(ds, **kwargs)
    a.save()
    gc.collect()


####################