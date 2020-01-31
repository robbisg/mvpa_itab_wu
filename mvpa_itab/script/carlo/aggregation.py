from pyitab.analysis.results import *
from pyitab.utils.image import *
import nibabel as ni

dir_id = "within_memory"

dataframe = get_searchlight_results(path="/home/carlos/fmri/carlo_ofp/0_results/", 
                                    dir_id=dir_id, 
                                    field_list=['sample_slicer', 'cv'], 
                                    load_cv=False)

for cv in [3, 5]:
    df_ = filter_dataframe(dataframe, selection_dict={'n_splits':[cv]})

    aggregated = []
    for map_ in df_['map'].values:
        img = ni.load(map_)
        aggregated.append(img.get_data())

    aggregated = np.rollaxis(np.array(aggregated), 0, 4)

    fname = "/home/carlos/fmri/carlo_ofp/0_results/%s_evidence_1_cv_%d.nii.gz" % (dir_id, cv)
    #ni.save(ni.Nifti1Image(aggregated, img.affine), fname)

    

filelist = glob.glob("/home/carlos/fmri/carlo_ofp/0_results/*_evidence_1_cv_*.nii.gz")
mask_fname = "/home/carlos/fmri/carlo_ofp/1_single_ROIs/glm_atlas_mask_333.nii.gz"
for fl in filelist:
    out_fname = fl[:-7]+"_minus_chance.nii.gz"
    out_demeaned = fl[:-7]+"_demeaned.nii.gz"

    remove_value_nifti(fl, value=0.5, output_fname=out_fname, mask_fname=mask_fname)
    remove_mean_brick(fl, output_fname=out_demeaned, mask_fname=mask_fname)