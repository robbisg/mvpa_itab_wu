import numpy as np
from mvpa_itab.pipeline import Transformer
from mvpa_itab.conn.connectivity import z_fisher


import logging
logger = logging.getLogger(__name__)



class ZFisherTransformer(Transformer):
    
    
    def __init__(self, name='zfisher', **kwargs):
        Transformer.__init__(self, name=name, **kwargs)
    
    
    def transform(self, ds):
        
        logger.info("Transforming values to zfisher.")
        samples = z_fisher(ds.samples)
        samples[np.isinf(samples)] = 1
        
        ds.samples = samples
        
        return ds