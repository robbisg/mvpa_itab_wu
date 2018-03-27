from mvpa_itab.preprocessing.functions import *


class PreprocessingPipeline(Node):
    
    
    def __init__(self, nodes=None):
        self.nodes = []
        if nodes != None:
            self.nodes = nodes
    
    
    def add(self, node):
        
        self.nodes.append(node)
        return self
    
    
    def transform(self, ds):
        for node in self.nodes:
            ds = node.transform(ds)
            
        return ds
            
    
    
class StandardPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureWiseNormalizer(),                   
                      
                      ]
        
        PreprocessingPipeline.__init__(self, self.nodes)
        


class MonksPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureWiseNormalizer(),
                      SampleSlicer(selection_dictionary={'events_number':range(1,13)})                 
                      
                      ]
        
        PreprocessingPipeline.__init__(self, self.nodes)            