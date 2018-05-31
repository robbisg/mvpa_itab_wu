from mvpa_itab.preprocessing.functions import Node, Detrender,\
    FeatureWiseNormalizer, SampleSlicer, SampleWiseNormalizer

import logging
logger = logging.getLogger(__name__)

class PreprocessingPipeline(Node):
    
    
    def __init__(self, name='pipeline', nodes=[Node()]):
        self.nodes = []
        if nodes != None:
            self.nodes = nodes
            
        Node.__init__(self, name)
    
    
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
                      SampleSlicer(selection_dictionary={'events_number':range(1,13)})                 
                      
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