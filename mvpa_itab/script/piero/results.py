from pyitab.analysis.results.bids import get_searchlight_results_bids
dataframe = get_searchlight_results_bids('/media/robbis/DATA/fmri/EGG/derivatives/', pipeline=['egg+delay'])

# Plain
filtered = filter_dataframe(dataframe, task=['plain'], filetype=['mean'])

command = "3dttest++ -singletonA 0.5 -setB task %s -prefix %s"
setB = ""
avg = []
for i, df in filtered.iterrows():

    #setB += "sub%02d %s'[0]' " % (i+1, df['filename'].values[0])
    img = ni.load(df['filename'])
    avg.append(img.data)

save_map(path+"average-plain.nii.gz", np.mean(avg, axis=0), img.affine)

######################################################################

from pyitab.analysis.results.base import filter_dataframe
from pyitab.utils.image import save_map
import nibabel as ni

dataframe = get_searchlight_results_bids('/media/robbis/DATA/fmri/EGG/derivatives/', pipeline=['egg+delay'])
dataframe = get_searchlight_results_bids('/media/robbis/DATA/fmri/EGG/derivatives/', pipeline=['egg+below+chance'])

path = '/media/robbis/DATA/fmri/EGG/derivatives/'

# Plain
filtered = filter_dataframe(dataframe, task=['plain'], filetype=['mean'])

command = "3dttest++ -singletonA 0.5 -setB task %s -prefix %s"
setB = ""
avg = []
for i, df in filtered.iterrows():

    setB += " sub%02d %s'[0]' " % (i+1, df['filename'])
    img = ni.load(df['filename'])
    avg.append(img.data)

save_map(path+"pipeline-type-average_task-plain.nii.gz", np.mean(avg, axis=0), img.affine)


command = command %(setB, path+"egg+below+chance")