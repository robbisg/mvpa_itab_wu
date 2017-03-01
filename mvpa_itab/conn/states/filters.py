import numpy as np


def get_filter(filter_type):
    
    mapping = {'none': NoneFilter(),
               'normalize': NormalizerFilter()                            
               }
    
    return mapping[filter_type]


class Filter(object):
    
    def fit(self, data):
        raise NotImplementedError()
    
    

class NormalizerFilter(Filter):
    
    def fit(self, data):                
        return data - data.mean(2)[...,None]
        
        

class NoneFilter(Filter):
    
    def fit(self, data):
        return data
    
    
    

        