from mvpa_itab.test_wu import *
import numpy as np
from mvpa2.clfs.svm import LinearCSVMC
from mvpa_itab.washu import load

#img = load('/media/robbis/DATA/fmri/memory/nibabel__/angque_task_4RES_GOonly_res_b1.4dfp.ifh')
#data = img.get_data()
root = logging.getLogger()
root.setLevel(logging.INFO)
path = '/media/robbis/DATA/fmri/learning/'
conf = read_configuration(path, 'learning.conf', 'task')

#conf['mask_area'] = 'll'

subjects = ['andant']

ds_merged = get_merged_ds(path, subjects, 'learning.conf', 'task', dim=4)
ds = ds_merged[0]


if __debug__:
    debug.active += ["SLC"]



cv = CrossValidation(CorrelationMeasure(), 
                     TargetCombinationPartitioner(attr='targets'),
                     splitter=Splitter(attr='partitions', attr_values=(3,2)),
                     errorfx=None)


kwa = dict(voxel_indices=Sphere(3), 
            event_offsetidx=Sphere(7))
mask = load_mask(path, subjects[0], **conf)



sl = Searchlight(cv, 
                 IndexQueryEngine(**kwa), 
                 roi_ids=np.arange(0, np.count_nonzero(mask.get_data() != 0), 5))

sl_map = sl(ds)

############################
### Partitioner debug ######
'''
nfold = TargetCombinationPartitioner(attr='targets')
gen = nfold.generate(ds)
ds_part = gen.next()

print 'hello'
'''
ds.fa.voxel_indices[np.arange(0, np.count_nonzero(mask.get_data() != 0),5)]
ds_ = fmri_dataset('/media/robbis/DATA/fmri/learning/andant/rest/2860_b_rest6_rmsp_faln_dbnd_xr3d_atl.4dfp_crop_brain_flirt.nii.gz', 
                   add_fa={'voxel_indices':ds.fa.voxel_indices[
                                                               np.arange(0, 
                                                                         np.count_nonzero(mask.get_data() != 0),
                                                                         5)]})