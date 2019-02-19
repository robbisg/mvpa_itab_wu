from mvpa_itab.io.base import load_subject_ds
from mvpa_itab.preprocessing.pipelines import MonksPreprocessingPipeline
import nibabel as ni
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm.classes import LinearSVC
from mne.decoding import GeneralizingEstimator
from mvpa_itab.preprocessing.functions import FeatureSlicer, SampleSlicer
from mne.decoding.base import cross_val_multiscore



atlas_dict = {}
#path_templates = '/media/robbis/DATA/fmri/templates_fcmri/1_MNI_3mm'
path_templates = '/home/carlos/mount/megmri03/templates_fcmri/1_MNI_3mm'
for network in os.listdir(path_templates):
    atlas_dict[network[:-21]] = ni.load(os.path.join(path_templates, network))
    
#path = '/media/robbis/DATA/fmri/monks'
path = '/home/carlos/mount/megmri03/monks'
subjects = os.listdir(path)
subjects = [s for s in subjects if s.find('.') == -1 and s.find('_') == -1]

# Load monk data in the form of n_samples x n_voxels x n_time
ds, _, _ = load_subject_ds(path,
                           subjects[:1],
                           #os.path.join(path, 'subjects.csv'), 
                           'meditation_permut1.conf',
                           'fmri',
                           prepro=MonksPreprocessingPipeline(),
                           roi_labels=atlas_dict
                           )



clf = make_pipeline(StandardScaler(), LinearSVC(C=1))
time_gen = GeneralizingEstimator(clf, scoring='accuracy', n_jobs=20)

ds = SampleSlicer({'group': ['E']}).transform(ds)

scores_dict = {}
# Generalization of time
for network in os.listdir(path_templates):
    
    network = network[:-21]
    ds_network = FeatureSlicer({network:['!0']}).transform(ds)
    
    n_samples, n_voxels = ds_network.shape
    data = ds_network.samples.reshape(-1, 135, n_voxels)
    X = np.rollaxis(data, 1, 3)
    y = np.arange(data.shape[0]) % 2
    
    scores = cross_val_multiscore(time_gen, X, y, cv=12, n_jobs=20)

    scores_dict[network] = scores


######## Pattern connectivity ################
results = trajectory_connectivity(ds)