from mvpa_itab.preprocessing.functions import *


class PreprocessingPipeline(Node):
    
    
    def __init__(self, nodes=None):
        self.nodes = []
        if nodes != None:
            self.nodes = nodes
    
    
    def add(self, node):
        
        self.nodes.append(node)
        return self
    
    
    def run(self, ds):
        for node in self.nodes:
            ds = node.run(ds)
            
        return ds
            
    
    
class CarloMemoryPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      
                      
                      
                      
                      ]
        
        
        
        
            