import numpy as np
from mvpa_itab.conn.connectivity import z_fisher


def get_filter(filter_type):
    
    mapping = {
               'none':      NoneFilter(),
               'normalize': NormalizerFilter(),
               'matrix':    MatrixNormalizationFilter(),
               'subject':   SubjectWiseNormalizationFilter(),
               'zfisher':   ZFisherFilter()                 
               }
    
    return mapping[filter_type]


class Filter(object):
    
    def transform(self, data):
        raise NotImplementedError()
    
    

class NormalizerFilter(Filter):
    
    def transform(self, data):
        """
        Implement a sample-wise demeaning.
        """                
        return data - data.mean(2)[...,None]
        

class MatrixNormalizationFilter(Filter):
    
    def transform(self, data):
        return data - data.mean(1).mean(0)
    

class SubjectWiseNormalizationFilter(Filter):
    
    def transform(self, data):
        return data - data.mean(1)[:, None, :]


class ZFisherFilter(Filter):
    
    def transform(self, data):
        return z_fisher(data)



class NoneFilter(Filter):
    
    def transform(self, data):
        return data
    
    
    

        