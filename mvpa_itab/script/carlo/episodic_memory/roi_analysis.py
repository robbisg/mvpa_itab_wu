import os
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

from pyitab.preprocessing.functions import SampleSlicer, TargetTransformer,\
    Detrender
from pyitab.preprocessing.normalizers import FeatureZNormalizer, SampleZNormalizer
from pyitab.preprocessing.balancing.base import Balancer
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.io.loader import DataLoader

from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator

from pyitab.analysis.pipeline import AnalysisPipeline
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.model_selection import *
from sklearn.svm.classes import SVC

import _pickle as pickle

loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", task='OFP_NORES')
ds = loader.fetch()

decoding = RoiDecoding(n_jobs=20, scoring=['accuracy'])

results = dict()
for subject in np.unique(ds.sa.subject):
    results[subject] = []
    for evidence in [1, 2, 3]:
        
        pipeline = PreprocessingPipeline(nodes=[ 
                                                TargetTransformer('decision'),
                                                SampleSlicer(**{'subject':[subject], 'evidence':[evidence]}),
                                                Balancer(balancer=RandomUnderSampler(return_indices=True), attr='chunks'),
                                                ])
        
        ds_ = pipeline.transform(ds)
        
        decoding.fit(ds_, roi=['lateral_ips'])

        results[subject].append(decoding.scores)

with open(os.path.join(loader._data_path, '0_results/lateral_ips_decoding.pickle'), 'wb') as output:
    pickle.dump(results, output)
    
    
    
###################
import cPickle as pickle

loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", task='OFP_NORES')

detrending_pipe =  PreprocessingPipeline(nodes=[Detrender(chunks_attr='file'), Detrender()])
zscore_pipe = PreprocessingPipeline(nodes=[FeatureWiseNormalizer(), SampleWiseNormalizer()])

ds = loader.fetch(prepro=detrending_pipe)

decoding = RoiDecoding(n_jobs=20, 
                    scoring=['accuracy'], 
                    cv=StratifiedKFold(n_splits=5),
                    estimator=SVC(C=1, kernel='linear')
                    )

results = dict()
for subject in np.unique(ds.sa.subject):
    results[subject] = []
    for evidence in [1, 2, 3]:
        
        pipeline = PreprocessingPipeline(nodes=[ 
                                                TargetTransformer('decision'),
                                                SampleSlicer({'subject':[subject], 'evidence':[evidence]}),
                                                Balancer(balancer=RandomUnderSampler(return_indices=True), attr='all'),
                                                ])
        
        ds_ = pipeline.transform(ds)
        
        decoding.fit(ds_, roi=['lateral_ips'], cv_attr=None, prepro=zscore_pipe)

        results[subject].append(decoding.scores)

with open(os.path.join(loader._data_path, '0_results/within_detrending_total_zscore_roi_5_fold.pickle'), 'wb') as output:
    pickle.dump(results, output)
    
    
    
##################


decoding = RoiDecoding(n_jobs=20, 
                    scoring=['accuracy'], 
                    cv=LeaveOneGroupOut(),
                    estimator=SVC(C=1, kernel='linear'))

results = []

for evidence in [1, 2, 3]:
    
    pipeline = PreprocessingPipeline(nodes=[ 
                                            TargetTransformer('decision'),
                                            SampleSlicer({'evidence':[evidence]}),
                                            Balancer(balancer=RandomUnderSampler(return_indices=True), attr='subject'),
                                            ])
    
    ds_ = pipeline.transform(ds)
    
    decoding.fit(ds_, roi=['lateral_ips'], cv_attr='subject')

    results.append(decoding.scores)
    
with open(os.path.join(loader._data_path, '0_results/across_lateral_ips_balance_subject.pickle'), 'wb') as output:
    pickle.dump(results, output)    


###############


_default_conf = {
       
                       'prepro':['targettrans', 'sampleslicer'],
                       'sampleslicer__evidence': ['1'], 
                       'targettrans__target':"decision",
                       
                       'estimator': [('svr', SVC(C=1, kernel='linear'))],
                       
                       'cv': StratifiedKFold,
                       'cv__n_splits': 5,

                       'scores': ['accuracy'],
                       
                       'analysis': Decoding,
                       'cv_attr': 'subject',
                       'roi':['lateral_ips'],
                       'kwargs__prepro':['featurenorm', 'samplenorm'],
                       'analysis__n_jobs':2
                   
                   }


_default_options = {
       
                       'sampleslicer__evidence' : [[1], [2], [3]],
                       'cv__n_splits': [3, 5],
                        }
 
 
iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="permut1_wm").fit(ds, **kwargs)
    a.save()


##################################################
# Review
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator

loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", 
                    task='OFP_NORES')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(),
                                      Detrender(chunks_attr='file'),
                                      SampleZNormalizer(),
                                      FeatureZNormalizer(),
                                      ])



ds = loader.fetch(prepro=prepro)


_default_options = {
                    'kwargs__roi': [['within_conjunction']],
                    #'sample_slicer__subject': [[s] for s in],
                    'sample_slicer__evidence': [[1], [2], [3]],           
                    }




_default_config = {
                    'prepro': [ 'target_transformer', 'sample_slicer', 'balancer'],
                    'target_transformer__attr':'decision',
                    'sample_slicer__decision': ['L', 'F'],
                    'balancer__attr': 'subject',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': LeaveOneGroupOut,
                    #'cv__n_splits': 7,
                    #'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': RoiDecoding,
                    'analysis__n_jobs': 5,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    'kwargs__roi' : ['omnibus'],
                    'kwargs__cv_attr': 'subject'

                    }


iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="review_across_roi_within").fit(ds, **kwargs)
    a.save(subdir="0_results/review")

#############################################
loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", 
                    task='OFP_NORES')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(),
                                      Detrender(chunks_attr='file'),
                                      SampleZNormalizer(),
                                      FeatureZNormalizer(),
                                      SampleSlicer(evidence=[1])
                                      ])



ds = loader.fetch(prepro=prepro)


_default_options = {
                    'kwargs__roi': [['within_conjunction']],
                    'sample_slicer__subject': [[s] for s in ds.sa.subject],
                    }




_default_config = {
                    'prepro': [ 'target_transformer', 'sample_slicer', 'balancer'],
                    'target_transformer__attr':'decision',
                    'sample_slicer__decision': ['L', 'F'],
                    'balancer__attr': 'all',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedShuffleSplit,
                    'cv__n_splits': 5,
                    'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': Decoding,
                    'analysis__n_jobs': 1,
                    
                    'analysis__permutation': 100,
                    
                    'analysis__verbose': 0,

                    'kwargs__roi' : ['omnibus'],
                    'kwargs__cv_attr': 'chunks'

                    }


iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="review_within_roi_decoding").fit(ds, **kwargs)
    a.save(subdir="0_results/review")

#############################################
loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", 
                    task='OFP_NORES')

prepro = PreprocessingPipeline(nodes=[
                                      Transformer(), 
                                      #Detrender(),
                                      #Detrender(chunks_attr='file'),

                                      #SampleSlicer(evidence=[1])
                                      ])



ds = loader.fetch(prepro=prepro)


_default_options = {
                    'kwargs__roi': [['within_conjunction']],
                    'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                    'sample_slicer__evidence': [[1], [2], [3]],           
                    }


kwargs_prepro = PreprocessingPipeline(nodes=[SampleZNormalizer(),
                                             FeatureZNormalizer()]) 

_default_config = {
                    'prepro': [ 'target_transformer', 'sample_slicer', 'balancer'],
                    'target_transformer__attr':'decision',
                    'sample_slicer__decision': ['L', 'F'],
                    'balancer__attr': 'all',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedShuffleSplit,
                    'cv__n_splits': 5,
                    'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': Decoding,
                    'analysis__n_jobs': 1,
                    
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 0,

                    'kwargs__roi' : ['omnibus'],
                    'kwargs__cv_attr': 'chunks',
                    'kwargs__prepro': kwargs_prepro

                    }


iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="review_within_good_roi_decoding").fit(ds, **kwargs)
    a.save(subdir="0_results/review")


##################### Correct vs Incorrect #############################
loader = DataLoader(configuration_file="/home/carlos/fmri/carlo_ofp/ofp_new.conf", 
                    task='OFP_NORES')

prepro = PreprocessingPipeline(nodes=[
                                      Transformer(), 
                                      Detrender(),
                                      Detrender(chunks_attr='file'),
                                      #SampleZNormalizer(),
                                      #FeatureZNormalizer(),
                                      #SampleSlicer(evidence=[1])
                                      ])



ds = loader.fetch(prepro=prepro)

kwargs_prepro = [SampleZNormalizer(), FeatureZNormalizer()]


_default_options = {
                    'kwargs__roi': [['within_conjunction']],
                    'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                    #'sample_slicer__subject': [[s] for s in ['s02_160216micbra', 's05_160118asspet', 's08_160301serpac', 
                    #                            's09_160304vindic', 's11_160318margia', 's12_160323edogog',
                    #                            's14_160405chrfas', 's06_160126ilagia']],
                    'sample_slicer__evidence': [[1]], 
                    #'target_transformer__attr':['decision', 'memory_status']     
                    }


_default_config = {
                    
                    'prepro': [ 'target_transformer', 'sample_slicer', 'balancer'],
                    'target_transformer__attr':'memory_status',
                    #'target_transformer__attr':'decision',
                    #'sample_slicer__decision': ['L', 'F'],
                    'sample_slicer__memory_status': ['L', 'F'],
                    #'sample_slicer__accuracy': ['I', 'C'],
                    'balancer__attr': 'all',
                    
                    'estimator': [('clf', SVC(C=1, kernel='linear'))],
                    'estimator__clf__C': 1,
                    'estimator__clf__kernel': 'linear',
                    
                    'cv': StratifiedShuffleSplit,
                    #'cv__cv_test':StratifiedKFold(n_splits=2),
                    'cv__n_splits': 50,
                    'cv__test_size': 0.25,
                    
                    'scores': ['accuracy'],
                    
                    'analysis': RoiDecoding,
                    'analysis__n_jobs': 5,
                    'analysis__permutation': 0,
                    
                    'analysis__verbose': 1,

                    'kwargs__cv_attr': 'chunks',
                    'kwargs__prepro': kwargs_prepro,

                    }

iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="review_mem_vs_dec_all").fit(ds, **kwargs)
    a.save(subdir="0_results/review")


###### Review #2 ##########
loader = DataLoader(configuration_file="/media/robbis/DATA/fmri/carlo_ofp/ofp.conf", 
                    task='OFP_NORES')

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(),
                                      Detrender(chunks_attr='file'),
                                      #SampleZNormalizer(),
                                      #FeatureZNormalizer(),
                                      ])



ds = loader.fetch(prepro=prepro)

import itertools

options = {#'evidence':[1,2,3],
            'subject':[s for s in np.unique(ds.sa.subject)],
            'kwargs__roi': ['within', 'across']}

combinations_ = list(itertools.product(*[options[arg] for arg in options]))

splits = []
#for ev, name, mask in combinations_:
for name, mask in combinations_:
    ds_ = SampleSlicer(#evidence=[ev], 
                       subject=[name], 
                       decision=['L', 'F']).transform(ds)
    ds_ = TargetTransformer(attr='decision').transform(ds_)
    balancer = Balancer(attr='all')
    ds__ = balancer.transform(ds_)
    split = np.zeros_like(ds__.targets, dtype=np.int)
    cv = StratifiedKFold(n_splits = 5)
    for j, (train, test) in enumerate(cv.split(ds__.samples, ds__.targets)):
        split[test] = j
        #print(Counter(ds__.targets[test]))

    splits.append([name, 
                   #ev, 
                   balancer._balancer._mask, 
                   split, 
                   mask])

###########
_default_options = {

                        'sample_slicer__subject': [],
                        'sample_slicer__evidence': [],
                        'dataset_masker__mask': [],
                        'cv__test_fold': [],
                        }

for s in splits:
    _default_options['sample_slicer__subject'].append(s[0])
    _default_options['sample_slicer__evidence'].append(s[1])
    _default_options['dataset_masker__mask'].append(s[2])
    _default_options['cv__test_fold'].append(s[3])


####################



kwargs_prepro = [SampleZNormalizer(), FeatureZNormalizer()]
for split in splits:

    subject = split[0]
    #evidence = split[1]
    mask = split[1]
    test = split[2]
    roi = split[3]

    _default_conf = {
                        
                        'prepro': [ 'target_transformer', 'sample_slicer', 'dataset_masker'],
                        
                        #'target_transformer__attr':'decision',
                        'sample_slicer__decision': ['L', 'F'],
                        'sample_slicer__subject': [subject],
                        #'sample_slicer__evidence': [evidence],
                        'target_transformer__attr':'decision',
                        'dataset_masker__mask': mask,
                        
                        'estimator': [('clf', SVC(C=1, kernel='linear'))],
                        'estimator__clf__C': 1,
                        'estimator__clf__kernel': 'linear',
                        
                        'cv': PredefinedSplit,
                        #'cv__cv_test':StratifiedKFold(n_splits=2),
                        #'cv__n_splits': 50,
                        'cv__test_fold': test,
                        
                        'scores': ['accuracy'],
                        
                        'analysis': RoiDecoding,
                        'analysis__n_jobs': 5,
                        'analysis__permutation': 0,
                        
                        'analysis__verbose': 1,
                        'kwargs__roi' : [roi],
                        'kwargs__cv_attr': 'chunks',
                        'kwargs__prepro': kwargs_prepro,
                        'kwargs__return_decisions': True,

                        }

    
    conf = AnalysisConfigurator(**_default_conf)
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="review__within_between_k5_all").fit(ds, **kwargs)
    a.save(subdir="0_results/review_2")





from nitime.timeseries import TimeSeries
from nitime.analysis import CorrelationAnalyzer


def connectivity(x):
    data = np.vstack(x)
    ts = TimeSeries(data, sampling_interval=1.)
    return CorrelationAnalyzer(ts).corrcoef

from scipy.stats import ttest_1samp
def ttest(x, index=0, p_value=0.005/15.):
    data = np.dstack(x)
    val = ttest_1samp(data, 0., axis=2)
    if index == 0:
        t = val[index]
        t[np.diag_indices_from(t)] = 0
        return t

    return val[index] < p_value



dir_id = "review__within_between_roi"

dataframe = get_results('/home/robbis/mount/permut1/fmri/carlo_ofp/0_results/review_2/', 
                          dir_id="review__within_between_k5_all", 
                          field_list=['sample_slicer'],
                          result_keys=['decisions'])
                           
df = df_fx_over_keys(dataframe,                         
                     keys=['subject', 'roi', 'roi_value'], 
                     attr='decisions',     
                     fx=lambda x:np.hstack(x))            
                         
df_corr = df_fx_over_keys(df, 
                          keys=['roi','subject'], 
                          attr='decisions', 
                          fx=connectivity)

df_avg = df_fx_over_keys(df_corr, 
                         keys=[ 'roi'], 
                         attr='decisions', 
                         fx=lambda x:np.mean(np.dstack(x), axis=2))

df_test = df_fx_over_keys(df_corr, 
                          keys=[ 'roi'], 
                          attr='decisions', 
                          fx=ttest,
                          index=1,
                          p_value=0.005/15.)

df_tvalue = df_fx_over_keys(df_corr, 
                          keys=[ 'roi'], 
                          attr='decisions', 
                          fx=ttest,
                          
                          )

labels = [['R Cerebellum', 
           'L Mid Front G', 
           'L AngG', 
           'L ParaHipp G',
           'R Insula',
           'L Fusiform G'],
          ['L AngG', 
          'R PCu', 
          'L Insula', 
          'L ParaHipp G',
          'L PCu',
          'L PreC G']]


fig, axes = pl.subplots(2, 3, sharey=True, figsize=(13,9))
for i, v in enumerate(df_test.values):
    ax = axes[np.int(i/3), i%3]
    ax.imshow(v[2])
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels[np.int(i/3)], rotation=30)
    ax.set_yticks(range(6))
    ax.set_yticklabels(labels[np.int(i/3)])
    ax.set_title("%s | Level of Evidence %s"%(v[0], v[1]))


fig, axes = pl.subplots(1,2, figsize=(13,9))
for i, v in enumerate(df_test.values):
    ax = axes[i]
    ax.imshow(v[1])
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels[np.int(i)], rotation=30)
    ax.set_yticks(range(6))
    ax.set_yticklabels(labels[np.int(i)])
    ax.set_title("%s"%(v[0]))

t = df_tvalue['decisions'][0]
p = np.logical_not(df_test['decisions'][0])
fig = pl.figure(figsize=(15,13))
#pl.title('conjunction_level_omnibus_control.nii.gz')
sign_t = masked_array(t, p)
pl.imshow(t, cmap=pl.cm.gray, alpha=0.4)
pl.imshow(sign_t, cmap=cmap_seaborn, vmax=6, vmin=-6)
pl.colorbar()
roi = labels[0]
_ = pl.xticks(np.arange(len(roi)), roi, rotation=90)
_ = pl.yticks(np.arange(len(roi)), roi)
#fig.savefig("matrix.eps", format='eps', dpi=300, figsize=(15,13))