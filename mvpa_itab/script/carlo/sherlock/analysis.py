from pyitab.io.loader import DataLoader
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from sklearn.model_selection import *
from pyitab.analysis.searchlight import SearchLight
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVC
from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.configurator import AnalysisConfigurator
import os


conf_file = path = "/home/robbis/mount/permut1/sherlock/bids/bids.conf"
loader = DataLoader(configuration_file=conf_file, 
                    loader='bids',
                    task='preproc', bids_task=['day1', 'day2'], bids_run=['01'])
ds = loader.fetch(n_subjects=1)