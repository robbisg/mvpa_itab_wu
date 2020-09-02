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
from pyitab.simulation.loader import SimulationLoader

n_repetitions = 2
conf_file = '/media/robbis/Seagate_Pt1/data/simulations/simulations.conf'
conf_file = "/home/robbis/mount/simulations.conf"


loader_configuration = {

                        'loader': SimulationLoader, 
                        'loader__conf_file': conf_file, 
                        'loader__loader': 'simulations', 
                        'loader__task': 'simulations',
                        
                        
                        'fetch__pipeline':['connectivity_state_simulator', 
                                           'phase_delayed_model', 
                                           'butter_filter', 
                                           'sliding_window_connectivity'], 
                        
                        'fetch__n_subjects': n_repetitions,

                        'fetch__connectivity_state_simulator__state_duration': {
                            'distribution': np.random.normal,
                            'params': {'loc':1.5, 'scale': 0.5}
                        },
                        'fetch__connectivity_state_simulator__length_dynamics': 100,

                        'fetch__phase_delayed_model__snr': 10,
                        'fetch__phase_delayed_model__delay': .5 * np.pi,

                        'fetch__butter_filter__order': 8,
                        'fetch__butter_filter__min_freq': 6,
                        'fetch__butter_filter__max_freq': 20,

                        'fetch__sliding_window__window_length': 1,  

}

loader_options = {
    'fetch__connectivity_state_simulator__state_duration': [
        {'distribution': np.random.normal, 'params': {'loc': 1.5, 'scale': 0.5}},
        {'distribution': np.random.normal, 'params': {'loc': 2.,  'scale': 0.5}},
        {'distribution': np.random.normal, 'params': {'loc': 2.5, 'scale': 0.5}},
        {'distribution': np.random.normal, 'params': {'loc': 3.,  'scale': 0.5}},
    ],
    'fetch__phase_delayed_model__snr': [3, 5, 10]
}



iterator = AnalysisIterator(loader_options, 
                            AnalysisConfigurator,
                            config_kwargs=loader_configuration,
                            #kind='combined'
                            )

ds_list = Parallel(n_jobs=-1)(delayed(generate)(configurator) for configurator in iterator)

for configurator in iterator:
    ds = generate(configurator)

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
    #loader.save(fname)
    return ds
