from mvpa_itab.test_wu import *
import numpy as np
from mvpa2.clfs.svm import LinearCSVMC

#img = ni.load('/media/robbis/DATA/fmri/group_searchlight_task_task_rad_5_analyze.img')

path = '/media/robbis/DATA/fmri/learning/'
conf = read_configuration(path, 'learning.conf', 'task')

conf['mask_area'] = 'll'

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
                 roi_ids=np.arange(0, 
                                   np.count_nonzero(mask.get_data() != 0)))

sl_map = sl(ds)

############################
### Partitioner debug ######
'''
nfold = TargetCombinationPartitioner(attr='targets')
gen = nfold.generate(ds)
ds_part = gen.next()

print 'hello'
'''

