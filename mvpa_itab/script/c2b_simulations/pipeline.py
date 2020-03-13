###################

import numpy as np
from pyitab.io.loader import DataLoader
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.io.connectivity import load_mat_ds
from pyitab.simulation.loader import load_simulations

from pyitab.preprocessing.math import AbsoluteValueTransformer, SignTransformer
from pyitab.preprocessing.base import Transformer

from pyitab.analysis.states.base import Clustering
from sklearn import cluster, mixture
from joblib import Parallel, delayed
conf_file = "/media/robbis/DATA/fmri/working_memory/working_memory.conf"
conf_file = '/m/home/home9/97/guidotr1/unix/data/simulations/meg/simulations.conf'


loader = DataLoader(configuration_file=conf_file, 
                    loader='simulations',
                    task='simulations')

ds = loader.fetch(prepro=Transformer())
    
_default_options = {
                    'estimator': [
                        [[('clf1', cluster.MiniBatchKMeans())]],
                        [[('clf1', cluster.KMeans())]], 
                        [[('clf1', cluster.SpectralClustering())]],
                        [[('clf1', cluster.AgglomerativeClustering())]],
                        [[('clf5', mixture.GaussianMixture())]], 
                    ],

                    'sample_slicer__subject':[[trial] for trial in np.unique(ds.sa.subject)],

                    'estimator__clf1__n_clusters': range(2, 10),
                    'estimator__clf5__n_components': range(2, 10),
                    }    
    
_default_config = { 
                    'prepro': ['sample_slicer'],
                    #'estimator': [[('MinBatchKMeans', cluster.MiniBatchKMeans)]],
                    'analysis': Clustering,}
 
 
iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config),
                            kind='combined'
                            )
def analysis(ds, conf):
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="c2b_01").fit(ds, **kwargs)
    a.save()
    del a


Parallel(n_jobs=10, verbose=1)(delayed(analysis)(ds, conf) for conf in iterator)

#################################################
from pyitab.simulation.loader import SimulationLoader

n_repetitions = 20

loader_configuration = {

                        'loader': SimulationLoader, 
                        'loader__conf_file': "/m/home/home9/97/guidotr1/unix/data/simulations/meg/simulations.conf", 
                        'loader__loader': 'simulations', 
                        'loader__task': 'simulations', 
                        
                        'fetch__pipeline':['connectivity_state_simulator', 
                                           'phase_delayed_model', 
                                           'butter_filter', 
                                           'sliding_window_connectivity'], 
                        
                        'fetch__n_subjects': n_repetitions,

                        'fetch__connectivity_state_simulator__min_time': 2.5,
                        'fetch__connectivity_state_simulator__max_time': 3,
                        'fetch__connectivity_state_simulator__length_states': 10,

                        'fetch__phase_delayed_model__snr': 10,
                        'fetch__phase_delayed_model__delay': .5 * np.pi,

                        'fetch__butter_filter__order': 8,
                        'fetch__butter_filter__min_freq': 6,
                        'fetch__butter_filter__max_freq': 20,

                        'fetch__sliding_window__window_length': 1,  

}

_default_config = { 
                    'prepro': ['sample_slicer'],
                    #'estimator': [[('MinBatchKMeans', cluster.MiniBatchKMeans)]],
                    'analysis': Clustering
                    }

_default_config.update(loader_configuration)


_default_options = {

                    'fetch__connectivity_state_simulator__min_time': [0.5, 1, 1.5, 2.],
                    'fetch__phase_delayed_model__snr': [10, 100, 1000, 10000],
                    
                    'estimator': [
                        [[('clf1', cluster.MiniBatchKMeans())]],
                        [[('clf1', cluster.KMeans())]], 
                        [[('clf1', cluster.SpectralClustering())]],
                        [[('clf1', cluster.AgglomerativeClustering())]],
                        [[('clf5', mixture.GaussianMixture())]], 
                    ],

                    'estimator__clf1__n_clusters': range(4, 8),
                    'estimator__clf5__n_components': range(4, 8),

                    'foo': [[1] for i in range(3)],
                    }


iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config),
                            kind='combined'
                            )
def analysis(ds, conf):
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="c2b_01").fit(ds=ds, **kwargs)
    a.save()
    del a


#Parallel(n_jobs=10, verbose=1)(delayed(analysis)(None, conf) for conf in iterator)

for conf in iterator:
    _ = analysis(None, conf)


#############################
# Build datasets

loader_configuration = {

                        'loader': SimulationLoader, 
                        'loader__conf_file': "/m/home/home9/97/guidotr1/unix/data/simulations/meg/simulations.conf", 
                        'loader__loader': 'simulations', 
                        'loader__task': 'simulations',
                        
                        
                        'fetch__pipeline':['connectivity_state_simulator', 
                                           'phase_delayed_model', 
                                           'butter_filter', 
                                           'sliding_window_connectivity'], 
                        
                        'fetch__n_subjects': n_repetitions,

                        'fetch__connectivity_state_simulator__min_time': 2.5,
                        'fetch__connectivity_state_simulator__max_time': 3,
                        'fetch__connectivity_state_simulator__length_states': 100,

                        'fetch__phase_delayed_model__snr': 10,
                        'fetch__phase_delayed_model__delay': .5 * np.pi,

                        'fetch__butter_filter__order': 8,
                        'fetch__butter_filter__min_freq': 6,
                        'fetch__butter_filter__max_freq': 20,

                        'fetch__sliding_window__window_length': 1,  

}

loader_options = {
    'fetch__connectivity_state_simulator__min_time': [0.5, 1., 1.5, 2.],
    'fetch__phase_delayed_model__snr': [10, 100, 1000, 10000]
    }



iterator = AnalysisIterator(loader_options, 
                            AnalysisConfigurator,
                            config_kwargs=loader_configuration,
                            #kind='combined'
                            )

ds_list = Parallel(n_jobs=-1)(delayed(generate)(configurator) for configurator in iterator)

def generate(configurator):
    loader = configurator._get_loader()
    fetch_kwargs = configurator._get_params('fetch')
    ds = loader.fetch(**fetch_kwargs)
    fname = 'ds_c2b-'
    for k in loader_options.keys():
        key = k.split('__')[-1]
        value = configurator._default_options[k]
        fname += "%s_%s-" % (key, str(value))
    fname = fname[:-1]
    loader.save(fname)
    return ds


###
from mvpa2.base.hdf5 import h5load
import os
path = '/m/home/home9/97/guidotr1/unix/data/simulations/meg/'
files = os.listdir(path)
files.sort()
files = [f for f in files if f.find("ds_c2b") != -1]
ds_list = [h5load(os.path.join(path, f)) for f in files]


_default_config = { 
                    'prepro': ['sample_slicer'],
                    #'estimator': [[('MinBatchKMeans', cluster.MiniBatchKMeans)]],
                    'analysis': Clustering,
                    'butter_filter__order': 4,
                    'butter_filter__btype': 'lowpass',
                    'butter_filter__max_freq': 2,
                }

_default_options = {

                    'kwargs__ds': ds_list,
                    
                    'estimator': [
                        [[('clf1', cluster.MiniBatchKMeans())]],
                        [[('clf1', cluster.KMeans())]], 
                        [[('clf1', cluster.SpectralClustering())]],
                        [[('clf1', cluster.AgglomerativeClustering())]],
                        [[('clf5', mixture.GaussianMixture())]], 
                    ],

                    'estimator__clf1__n_clusters': range(3, 9),
                    'estimator__clf5__n_components': range(3, 9),

                    'sample_slicer__subject': [[i+1] for i in range(20)],
                    }

iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator,
                            config_kwargs=_default_config,
                            kind='combined'
                            )
def analysis(conf, name):
    #print(conf._default_options)
    kwargs = conf._get_kwargs()
    #print(kwargs)
    a = AnalysisPipeline(conf, name=name).fit(**kwargs)
    a.save(path="/media/guidotr1/Seagate_Pt1/data/simulations/")
    return


results = Parallel(n_jobs=-1, verbose=1)(delayed(analysis)(conf, "c2b_real") for conf in iterator)


errors = 0
for conf in iterator:
    try:
        _ = analysis(conf, "c2b_easy")
    except Exception as _:
        errors += 1
        continue