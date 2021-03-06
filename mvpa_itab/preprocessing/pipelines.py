from mvpa_itab.preprocessing.functions import Detrender, FeatureWiseNormalizer, \
                SampleSlicer, SampleWiseNormalizer

from mvpa_itab.pipeline import Transformer

import logging
from mvpa_itab.preprocessing.mapper import function_mapper
logger = logging.getLogger(__name__)


class PreprocessingPipeline(Transformer):
    
    
    def __init__(self, name='pipeline', nodes=[Transformer()]):
        
        self.nodes = []
        
        if nodes != None:
            self.nodes = nodes
        
        if isinstance(nodes[0], str):
            self.nodes = [function_mapper(node)() for node in nodes]
                    
        Transformer.__init__(self, name)
    
    
    def add(self, node):
        
        self.nodes.append(node)
        return self
    
    
    def transform(self, ds):
        logger.info("%s is performing..." %(self.name))
        for node in self.nodes:
            ds = node.transform(ds)
        
        
        return ds
            
    
    
class StandardPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureWiseNormalizer(),            
                      SampleWiseNormalizer(),
                      ]
        
        PreprocessingPipeline.__init__(self, nodes=self.nodes)
        


class MonksPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureWiseNormalizer(),
                      SampleSlicer(selection_dictionary={'events_number':range(1, 13)})                 
                      
                      ]
        
        PreprocessingPipeline.__init__(self, nodes=self.nodes)
        
        

class MonksConnectivityPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureWiseNormalizer(),
                      SampleSlicer(selection_dictionary={'events_number':range(1,13)})                 
                      
                      ]
        
        PreprocessingPipeline.__init__(self, nodes=self.nodes)              