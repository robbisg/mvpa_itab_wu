from mvpa_itab.test_wu import *
import numpy as np

path = '/media/robbis/DATA/fmri/learning/'
conf = read_configuration(path, 'learning.conf', 'task')

conf['mask_area'] = 'll'

subjects = ['andant']

ds_merged = get_merged_ds(path, subjects, 'learning.conf', 'task')
ds = ds_merged[0]

if __debug__:
    debug.active += ["SLC"]


cv = CrossValidation(MahalanobisMeasure(), 
                     TargetCombinationPartitioner(attr='targets'),
                     splitter=Splitter(attr='partitions', attr_values=(3,2)))

sl = sphere_searchlight(cv, 3, space= 'voxel_indices')
sl_map = sl(ds)

############################
### Partitioner debug ######
'''
nfold = TargetCombinationPartitioner(attr='targets')
gen = nfold.generate(ds)
ds_part = gen.next()


