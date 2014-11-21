from mvpa_itab.test_wu import *
import numpy as np
'''
path = '/media/robbis/DATA/fmri/learning/'
conf = read_configuration(path, 'learning.conf', 'task')

conf['mask_area'] = 'll'

subjects = ['andant']

ds_merged = get_merged_ds(path, subjects, 'learning.conf', 'task')
ds = ds_merged[0]

if __debug__:
    debug.active += ["SLC"]

h_task = HalfPartitioner(attr='task')
h_cond = HalfPartitioner(attr='target')

ds = ds[np.logical_or((ds.targets == 'trained'),(ds.targets == 'RestPre')) ]
'''
sl = sphere_searchlight(MahalanobisMeasure(), 3, space= 'voxel_indices')
sl_map = sl(ds)

############################
### Partitioner debug ######

nfold = NFoldPartitioner(attr='chunks')
gen = nfold.generate(ds)
ds_part = gen.next()



