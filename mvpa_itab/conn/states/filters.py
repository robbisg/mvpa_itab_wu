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
    
    def fit(self, data):
        raise NotImplementedError()
    
    

class NormalizerFilter(Filter):
    
    def fit(self, data):
        """
        Implement a sample-wise demeaning.
        """                
        return data - data.mean(2)[...,None]
        

class MatrixNormalizationFilter(Filter):
    
    def fit(self, data):
        return data - data.mean(1).mean(0)
    

class SubjectWiseNormalizationFilter(Filter):
    
    def fit(self, data):
        return data - data.mean(1)[:, None, :]


class ZFisherFilter(Filter):
    
    def fit(self, data):
        return z_fisher(data)



class NoneFilter(Filter):
    
    def fit(self, data):
        return data
    
    
    

        