import numpy as np
from mvpa_itab.preprocessing.functions import Node
from mvpa_itab.conn.connectivity import z_fisher


import logging
logger = logging.getLogger(__name__)



class ZFisherTransformer(Node):
    
    
    def __init__(self, name='zfisher', **kwargs):
        Node.__init__(self, name=name, **kwargs)
    
    
    def transform(self, ds):
        
        logger.info("Transforming values to zfisher.")
        samples = z_fisher(ds.samples)
        samples[np.isinf(samples)] = 1
        
        ds.samples = samples
        
        return ds