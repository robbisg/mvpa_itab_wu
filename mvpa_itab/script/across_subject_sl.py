import numpy as np
print np.__version__
from mvpa_itab.test_wu import *

path = '/media/robbis/DATA/fmri/memory/'

subjects = ['110929angque',
            '110929anngio',
            '111004edolat',
            '111006giaman',
            ]

if __debug__:
        debug.active += ["SLC"]
        
ds_, _, conf_ = load_subjectwise_ds(path, 
                                   subjects, 
                                   'memory.conf', 
                                   'BETA_MVPA', 
                                   extra_sa={'group':['group1',
                                                      'group1',
                                                      'group2',
                                                      'group2'
                                            ]},
                                   mask_area='PCC')

ds_.targets = ds_.sa.memory_status


cv = CrossValidation(LinearCSVMC(C=1),
                    HalfPartitioner(attr='group')
                    )

err_ = cv(ds_)

#sl = sphere_searchlight(cv, 3, space = 'voxel_indices')
#map_ = sl(ds_)