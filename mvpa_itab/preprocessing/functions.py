import numpy as np
from mvpa2.mappers.detrend import PolyDetrendMapper
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.mappers.zscore import ZScoreMapper

   
def function_mapper(name):
    
    mapper = {
              'detrending':Detrender,
              'class-modifier': TargetTransformer,
              'feature-wise': FeatureWiseNormalizer,
              'dataset-slicer': DatasetSlicer,
              'sample-wise': SampleWiseNormalizer
              }
    
    return mapper[name]
    
    
    
class Node(object):
    
    def __init__(self, name='node', **kwargs):
        self.name = name
    
    
    def transform(self, ds):
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
    
    
    def transform(self, ds):
        # logger
        return self.node.forward(ds)
                   


class SampleAverager(Node):
    
    
    def __init__(self, attributes, **kwargs):
        self.node = mean_group_sample(attributes)
        Node.__init__(self, **kwargs)
        
        
    def transform(self, ds):
        #logger.info('Dataset preprocessing: Averaging samples...')
        return ds.get_mapped(self.node)  




class FeatureWiseNormalizer(Node):
    
    def __init__(self, chunks_attr='chunks', param_est=None, **kwargs):
        self.node = ZScoreMapper(chunks_attr=chunks_attr, param_est=param_est)
        Node.__init__(self, **kwargs)
        
    
    def transform(self, ds):
        #logger.info('Dataset preprocessing: Normalization feature-wise...')
        return self.node.forward(ds)
    


    
class SampleWiseNormalizer(Node):       

    def transform(self, ds):
        #logger.info('Dataset preprocessing: Normalization sample-wise...')
        ds.samples -= np.mean(ds, axis=1)[:, None]
        ds.samples /= np.std(ds, axis=1)[:, None]
        
        ds.samples[np.isnan(ds.samples)] = 0
        
        return ds



class TargetTransformer(Node):
    
    def __init__(self, attribute, **kwargs):
        self._attribute = attribute
        Node.__init__(self, **kwargs)
    
    def transform(self, ds):
        ds.targets = ds.sa[self._attribute]



class DatasetSlicer(Node):
    """
    Selects only portions of the dataset based on a dictionary
    The dictionary indicates the sample attributes to be used as key and a list
    with conditions to be selected:
    
    selection_dict = {
                        'accuracy': ['I'],
                        'frame':[1,2,3]
                        }
                        
    This dictionary means that we will select all samples with frame attribute
    equal to 1 OR 2 OR 3 AND all samples with accuracy equal to 'I'.
    
    """

    def __init__(self, selection_dictionary=None, **kwargs):
        self._selection = selection_dictionary
        Node.__init__(self, **kwargs)    


    def transform(self, ds):
        
        selection_dict = self._selection
    
        selection_mask = np.ones_like(ds.targets, dtype=np.bool)
        for key, values in selection_dict.iteritems():
            
            #logger.info("Selected %s from %s attribute." %(str(values), key))
            
            ds_values = ds.sa[key].value
            condition_mask = np.zeros_like(ds_values, dtype=np.bool)
            
            for value in values:        
                condition_mask = np.logical_or(condition_mask, ds_values == value)
                
            selection_mask = np.logical_and(selection_mask, condition_mask)
            
        
        return ds[selection_mask]
    


