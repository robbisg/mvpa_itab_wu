import numpy as np
from mvpa2.mappers.detrend import PolyDetrendMapper
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.mappers.zscore import ZScoreMapper

   
def function_mapper(name):
    
    mapper = {
              'detrending':Detrender,
              'class-modifier': ClassModifier,
              'feature-wise': FeatureWiseNormalization,
              'sample-cleaner': SampleCleaner,
              'sample-wise': SampleWiseNormalization
              }
    
    return mapper[name]
    
    
    
class Node(object):
    
    def __init__(self, name='node', **kwargs):
        self.name = name
    
    
    def run(self, ds):
        # Return dataset
        raise NotImplementedError()
    
    

class NodeBuilder(object):
    
    def __init__(self, **node_dict):
        name = node_dict['name']
        self.node = function_mapper(name)(**node_dict)
        
    
    def get_node(self):
        return self.node


  

class Detrender(Node):
    
    def __init__(self, degree=1, chunks_attr='chunks', **kwargs):
        self.node = PolyDetrendMapper(chunks_attr=chunks_attr, polyord=degree)
        Node.__init__(self, **kwargs)
    
    
    def run(self, ds):
        # logger
        return self.node.forward(ds)
        



class SampleCleaner(Node):
    
    def __init__(self, modality='remove', labels=[], *kwargs):
        
        if modality == 'remove':
            self.node = np.logical_and
            self.mask_type = np.ones_like
            self.compare = np.not_equal
        else:
            self.node = np.logical_or
            self.mask_type = np.zeros_like
            self.compare = np.equal
        
        self.labels = labels    
        Node.__init__(self, **kwargs)
    
    def run(self, ds):
        
        mask = self.mask(ds.sa.targets, dtype=np.bool)
        
        for label in self.labels:
            mask = self.node(mask, self.compare(ds.sa.targets, label))
            
        return ds[mask]
           


class SampleAverager(Node):
    
    
    def __init__(self, attributes, **kwargs):
        self.node = mean_group_sample(attributes)
        Node.__init__(self, **kwargs)
        
        
    def run(self, ds):
        #logger.info('Dataset preprocessing: Averaging samples...')
        return ds.get_mapped(self.node)  




class FeatureWiseNormalization(Node):
    
    def __init__(self, chunks_attr='chunks', param_est=None, **kwargs):
        self.node = ZScoreMapper(chunks_attr=chunks_attr, param_est=param_est)
        Node.__init__(self, **kwargs)
        
    
    def run(self, ds):
        #logger.info('Dataset preprocessing: Normalization feature-wise...')
        return self.node.forward(ds)
    


    
class SampleWiseNormalization(Node):       

    def run(self, ds):
        #logger.info('Dataset preprocessing: Normalization sample-wise...')
        ds.samples -= np.mean(ds, axis=1)[:, None]
        ds.samples /= np.std(ds, axis=1)[:, None]
        
        ds.samples[np.isnan(ds.samples)] = 0
        
        return ds



class ClassModifier(Node):
    
    def __init__(self, attribute, **kwargs):
        self._attribute = attribute
        Node.__init__(self, **kwargs)
    
    def run(self, ds):
        ds.targets = ds.sa[self._attribute]



class DatasetMasker(Node):
    
    def run(self, ds):
        Node.run(self, ds)

