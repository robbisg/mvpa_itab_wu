from mvpa_itab.similarity.connectivity import pattern_connectivity,\
    subject_pattern_connectivity
from mvpa_itab.pipeline.deprecated.searchlight import LeaveOneSubjectOutSL
import nibabel as ni
from scipy.stats.stats import zscore
from mvpa_itab.similarity.trajectory import get_speed_timecourse,\
    partial_correlation, get_partial_correlation, trajectory_connectivity
from scipy.spatial.distance import euclidean
import numpy as np
import os
import cPickle as pickle

conf = {
        'condition_names': ['evidence', 'task'],
        'configuration_file': 'ofp.conf',
        'data_type': 'OFP_EXE',
        'evidence': 3,
        'mask_area': 'glm_atlas_mask_333.nii',
        'n_balanced_ds': 1,
        'n_folds': 3,
        'normalization': 'both',
        'partecipants': 'subjects.csv',
        'path': '/home/carlos/fmri/carlo_ofp/',
        'project': 'carlo_ofp',
        'split_attr': 'subject_ofp',
        'task': 'decision'}

path_roi = "/home/carlos/mount/megmri03/fmri/carlo_ofp/0_results/control_analysis/"
path = "/home/carlos/fmri/carlo_ofp/"
mask_list = ['cluster_level_effect_p0-0033_q0-05_mask.nii',
             'cluster_omnibus_p0-0014_q0-01_mask.nii',
             'cluster_level_effect_p0-0033_q0-05_mask.nii',
             'cluster_conjunction_level-0-003_omnibus-0-001_filled.nii.gz']

mask_list = ['conjunction_level_omnibus_control.nii.gz',
             'clusterin_level_effect_z_p-0.033_q-0.05_control_mask.nii.gz',
             'clusterin_omnibus_chi_p-0.015_q-0.01_control_mask.nii.gz'
             ]

mask_list = ['within_conjunction_mask.nii.gz',
             #'within_omnibus_q0.01_p0.003_30vx_mask.nii.gz',
             #'within_level_effect_q0.05_p0.005_20vx_mask.nii.gz'
              ]



result_dict = {}
for mask in mask_list:
    mask_img = ni.load(os.path.join(path_roi, mask))
    sl_analysis = LeaveOneSubjectOutSL(roi_labels={mask: mask_img}, **conf)
    ds = sl_analysis.load_dataset()
    for condition in ["L", "F"]:
        r = pattern_connectivity(ds, condition={'decision': condition}, cv_attribute='subject')
        m = subject_pattern_connectivity(r)
        key = "%s_%s" % (mask, condition)
        result_dict[key] = [r, m]
        
with open(os.path.join(path, 'result_dict_ctrl.pickle'), 'wb') as output:
    pickle.dump(result_dict, output)
        
#########################################

result_dict = {}

roi_labels = {mask:ni.load(os.path.join(path_roi, mask)) for mask in mask_list}


sl_analysis = LeaveOneSubjectOutSL(roi_labels=roi_labels, **conf)
ds = sl_analysis.load_dataset()



for subject in ds.sa.subject:
    results = trajectory_connectivity(ds, {'subject': subject, 'decision':['L', 'F']})




within = results['within_conjunction_mask.nii.gz']

subj_tc = [t[0] for t in within]
subj_bc = results['full_brain']

subject_partial_correlation = get_partial_correlation(subj_tc, subj_bc)


with open(os.path.join(path, 'result_trajectory_partial_correlation_within.pickle'), 'wb') as output:
    pickle.dump(subject_partial_correlation, output)

with open(os.path.join(path, 'result_trajectory_full_brain_within.pickle'), 'wb') as output:
    pickle.dump(results['full_brain'], output)

with open(os.path.join(path, 'result_trajectory_within.pickle'), 'wb') as output:
    pickle.dump(result_dict, output)

##############################################
import _pickle as pickle

from sklearn.model_selection._split import  GroupShuffleSplit
from sklearn.svm.classes import SVC
import numpy as np
from pyitab.io.loader import DataLoader
from scipy.io.matlab.mio import loadmat
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.configurator import ScriptConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.results import get_results, filter_dataframe
from pyitab.analysis.connectivity.multivariate import TrajectoryConnectivity
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.preprocessing.functions import Detrender, FeatureZNormalizer,\
    SampleZNormalizer, SampleSlicer, TargetTransformer, SampleSigmaNormalizer, \
    Transformer
from pyitab.preprocessing import Node
from pyitab.analysis.decoding.roi_decoding import Decoding
from pyitab.io.connectivity import load_mat_ds


from pyitab.preprocessing.math import AbsoluteValueTransformer

import warnings
warnings.filterwarnings("ignore")


conf_file =  "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"


loader = DataLoader(configuration_file=conf_file, 
                    #loader=load_mat_ds,
                    task='BETA_MVPA',
                    roi_labels={'conjunction':"/media/robbis/DATA/fmri/carlo_mdm/1_single_ROIs/conjunction_map_mask.nii.gz"})

prepro = PreprocessingPipeline(nodes=[
                                      #Transformer(), 
                                      Detrender(), 
                                      SampleZNormalizer(),
                                      FeatureZNormalizer()
                                      ])
#prepro = PreprocessingPipeline()


ds = loader.fetch(prepro=prepro)
    
_default_options = {
                        'kwargs__use_partialcorr': [True, False],
                        'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],           
                        }


_default_config = {
                    'prepro': ['sample_slicer'],
                    
                    'analysis': TrajectoryConnectivity,
                    "kwargs__roi":["conjunction"]

                    }
 
 
iterator = AnalysisIterator(_default_options, ScriptConfigurator(**_default_config))
for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="trajectory_connectivity").fit(ds, **kwargs)
    a.save()