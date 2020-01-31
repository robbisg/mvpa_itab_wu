###################
import _pickle as pickle

from sklearn.model_selection._split import GroupShuffleSplit
from sklearn.svm.classes import SVC
import numpy as np
from pyitab.io.loader import DataLoader
from scipy.io import loadmat, savemat
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import ScriptConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.results import get_results, filter_dataframe
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, FeatureZNormalizer,\
    SampleZNormalizer, SampleSlicer, TargetTransformer, SampleSigmaNormalizer, \
    Transformer
from pyitab.preprocessing import Node
from pyitab.analysis.decoding.roi_decoding import Decoding
from pyitab.io.connectivity import load_mat_ds

import warnings
from pyitab.preprocessing.math import AbsoluteValueTransformer
warnings.filterwarnings("ignore")


conf_file = "/media/robbis/DATA/fmri/working_memory/working_memory.conf"

loader = DataLoader(configuration_file=conf_file, 
                    loader=load_mat_ds,
                    task='CONN')

prepro = PreprocessingPipeline(nodes=[
                                      Transformer(), 
                                      #Detrender(), 
                                      #SampleSigmaNormalizer()
                                      #FeatureZNormalizer()
                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)

prepro = PreprocessingPipeline(nodes=[Transformer()])
detr = PreprocessingPipeline(nodes=[Detrender()])

ds_plain = loader.fetch(prepro=prepro)
ds_detr = loader.fetch(prepro=detr)

pl.plot(ds_plain.samples[:12,:5])
pl.plot(ds_detr.samples[:12,:5])

################################################

# Correction of connectivity data
path = "/media/robbis/DATA/fmri/working_memory/"
subject_list = glob.glob(path+"sub*")
subject_list.sort()

for i, subj in enumerate(subject_list):
    
    # Load attributes
    labels = np.loadtxt(os.path.join(subj, "meg_attributes.txt"), dtype=np.str, skiprows=1)
    data = loadmat(os.path.join(subj, "meg", "connectivity_matrix.mat"))
    data = data['data']

    norm_ds = []
    baseline_ds = []

    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]

        size_mat = loadmat(os.path.join(path, "parcelsizes_%s.mat" % (band)))
        size_key = "%s%s" % (band, condition)

        size = np.expand_dims(size_mat[size_key][i], axis=0)
        
        size_matrix = np.dot(size.T, size)

        norm_matrix = matrix / np.float_(size_matrix)
        norm_matrix[np.isnan(norm_matrix)] = 0.

        norm_ds.append(norm_matrix)

    norm_ds = {"data": np.array(norm_ds)}

    savemat(os.path.join(subj, "meg", "mpsi_normalized.mat"), norm_ds)


####################################################
# Correction based on rest #

path = "/media/robbis/DATA/fmri/working_memory/"
subject_list = glob.glob(path+"sub*")
subject_list.sort()

for i, subj in enumerate(subject_list):
    
    # Load attributes
    labels = np.loadtxt(os.path.join(subj, "meg_attributes.txt"), dtype=np.str, skiprows=1)
    data = loadmat(os.path.join(subj, "meg", "mpsi_normalized.mat"))
    data = data['data']

    norm_ds = []
    baseline_ds = []

    baseline_rest = dict()

    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]
        if condition == 'rest':
            baseline_rest[band] = matrix


    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]

        norm_matrix = matrix / baseline_rest[band]
        norm_matrix[np.isnan(norm_matrix)] = 0.

        norm_ds.append(norm_matrix)

    norm_ds = {"data": np.array(norm_ds)}

    savemat(os.path.join(subj, "meg", "mpsi_rest_normalized.mat"), norm_ds)


####################################################
# Correction of power #

path = "/media/robbis/DATA/fmri/working_memory/"
subject_list = glob.glob(path+"sub*")
subject_list.sort()

for i, subj in enumerate(subject_list):
    
    # Load attributes
    labels = np.loadtxt(os.path.join(subj, "power_attributes.txt"), dtype=np.str, skiprows=1)
    data = loadmat(os.path.join(subj, "meg", "power_parcel.mat"))
    data = data['data']

    norm_ds = []
    baseline_ds = []

    for j, matrix in enumerate(data):
        condition, band, _ = labels[j]

        size_mat = loadmat(os.path.join(path, "parcelsizes_%s.mat" % (band)))
        size_key = "%s%s" % (band, condition)

        size = size_mat[size_key][i]

        norm_matrix = data[j] * size
        norm_matrix[np.isnan(norm_matrix)] = 0.

        norm_ds.append(norm_matrix)

    norm_ds = {"data": np.array(norm_ds)}

    savemat(os.path.join(subj, "meg", "power_normalized.mat"), norm_ds)









