#######################################################
from pyitab.io.loader import DataLoader
from pyitab.io.connectivity import load_mat_ds
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import *
from pyitab.analysis.decoding.roi_decoding import Decoding
from pyitab.analysis.pipeline import AnalysisPipeline, Analyzer
from pyitab.analysis.configurator import ScriptConfigurator
from pyitab.analysis.iterator import AnalysisIterator
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import *
from sklearn.svm import SVC


conf_file = "/media/robbis/DATA/fmri/monks/meditation.conf"

matrix_list = glob.glob("/media/robbis/DATA/fmri/monks/061102chrwoo/fcmri/*.mat")
matrix_list = [m.split("/")[-1] for m in matrix_list]


for m in matrix_list:
    loader = DataLoader(configuration_file=conf_file, 
                        loader=load_mat_ds,
                        task='fcmri',
                        event_file=m[:-4]+".txt",
                        img_pattern=m)

    prepro = PreprocessingPipeline(nodes=[
                                        Transformer(), 
                                        #Detrender(), 
                                        SampleZNormalizer(),
                                        #FeatureZNormalizer()
                                        ])
    #prepro = PreprocessingPipeline()


    ds = loader.fetch(prepro=prepro)


    _default_options = {  
                        'sample_slicer__targets' : [['Vipassana'], ['Samatha']],

                        #'estimator__clf__C': [0.1, 1, 10],                          
                        #'cv__test_size': [0.1, 0.2, 0.25, 0.33],
                        'estimator__fsel__k': np.arange(50, 350, 50),
                        #'estimator__fsel__k': np.arange(10, 88, 15)
                            }    
        
    _default_config = {
                
                        'prepro':['feature_normalizer', 'target_transformer', 'sample_slicer'],
                        'target_transformer__target':'group',
                        'estimator': [('fsel', SelectKBest(k=400)),
                                        ('clf', SVC(C=1, kernel='linear'))],
                        'estimator__clf__C':1,
                        'estimator__clf__kernel':'linear',

                        'cv': StratifiedShuffleSplit,
                        'cv__n_splits': 100,
                        'cv__test_size': 0.25,

                        'scores': ['accuracy'],

                        'analysis': Decoding,
                        'analysis__n_jobs': 10,
                        'analysis__permutation': 0,
                        'analysis__verbose': 0,
                        'kwargs__roi': ['matrix_values'],
                        'kwargs__cv_attr':'name',

                        }
    
    
    iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))
    for i, conf in enumerate(iterator):
        kwargs = conf._get_kwargs()
        a = AnalysisPipeline(conf, name=m[:15]+"_EXPvsNOV").fit(ds, **kwargs)
        a.save()
        del a