###################
from sklearn.model_selection._split import GroupShuffleSplit
from sklearn.svm.classes import SVC
import numpy as np
from pyitab.io.loader import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, SampleSlicer, \
    Transformer, Resampler
from pyitab.preprocessing.normalizers import FeatureZNormalizer, \
    SampleZNormalizer, SampleZNormalizer, SampleSigmaNormalizer, \
    FeatureSigmaNormalizer
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from pyitab.io.connectivity import load_mat_ds

from pyitab.preprocessing.math import AbsoluteValueTransformer, SignTransformer

import warnings
warnings.filterwarnings("ignore")
 
######################################
# Only when running on permut1
from pyitab.utils import enable_logging
root = enable_logging()
#####################################

conf_file =  ""
conf_file = '/media/guidotr1/Seagate_Pt1/data/Viviana2018/meg/movie.conf' 
 
loader = DataLoader(configuration_file=conf_file,  
                    loader='mat', 
                    task='conn')


prepro = PreprocessingPipeline(nodes=[
                                      #SampleZNormalizer(),
                                      #FeatureZNormalizer(),
                                      Resampler(down=5)
                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)
    
_default_options = {
                    'sample_slicer__targets' : [['movie', 'rest'], 
                                                ['movie', 'scramble']],
                    'sample_slicer__band' : [['alpha'], ['beta']],
                    }    
    
_default_config = {
               
                       'prepro':['sample_slicer', 'feature_slicer'],
                       #'ds_normalizer__ds_fx': np.std,
                       'sample_slicer__band': ['alpha'], 
                       'sample_slicer__targets' : ['0back', '2back'],
                       'estimator': [
                           #('fsel', SelectKBest(k=50)),
                           ('clf', SVC(C=1, kernel='linear'))],
                       'estimator__clf__C': 1,
                       'estimator__clf__kernel':'linear',

                       'cv': GroupShuffleSplit,
                       'cv__n_splits': 75,
                       'cv__test_size': 0.25,

                       'scores' : ['accuracy'],

                       'analysis': TemporalDecoding,
                       'analysis__n_jobs': -1,
                       'analysis__permutation': 0,
                       'analysis__verbose': 0,
                       'kwargs__roi': ['matrix_values'],
                       'kwargs__cv_attr': 'subjects',

                    }
 
import gc
iterator = AnalysisIterator(_default_options, AnalysisConfigurator, config_kwargs=_default_config)
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="movie+revenge+nofsel").fit(ds, **kwargs)
    a.save()
    del a
    gc.collect()

################################# Results
from pyitab.analysis.results.bids import get_results_bids

path = '/media/guidotr1/Seagate_Pt1/data/Viviana2018/meg/derivatives/'

dataframe = get_results_bids(path=path,  
                             pipeline="movie+revenge",
                             field_list=['sample_slicer'],
                             result_keys=['features'] 
                             )



tasks = np.unique(dataframe['targets'].values)
bands = np.unique(dataframe['band'].values)

for task in tasks:
    for band in bands:

        df = filter_dataframe(dataframe, targets=[task], band=[band])
        
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