import os
from mvpa2.suite import fmri_dataset, SampleAttributes
from mvpa2.suite import eventrelated_dataset
import logging
import numpy as np
import nibabel as ni
from mvpa_itab.utils import fidl_convert
from mvpa2.datasets.eventrelated import find_events
from mvpa_itab.main_wu import detrend_dataset, normalize_dataset
from mvpa2.base.dataset import vstack
from mvpa_itab.preprocessing.functions import Node
from mvpa_itab.conn.connectivity import z_fisher



class ZFisherTransformer(Node):
    
    
    def __init__(self, name='zfisher', **kwargs):
        Node.__init__(self, name=name, **kwargs)
    
    
    def transform(self, ds):
        
        samples = z_fisher(ds.samples)
        samples[np.isinf(samples)] = 1
        
        ds.samples = samples
        
        return ds