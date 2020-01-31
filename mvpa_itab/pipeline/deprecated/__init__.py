import logging
logger = logging.getLogger(__name__)


class ScriptIterator(object):
    
    
    
    def __init__(self, options):
        return
        
    
    
    
    def setup_analysis(self, **kwargs):
        
        import itertools
            
        args = [arg for arg in kwargs]
        logger.info(kwargs)
        combinations_ = list(itertools.product(*[kwargs[arg] for arg in kwargs]))
        self.configurations = [dict(zip(args, elem)) for elem in combinations_]
        self.i = 0
        self.n = len(self.configurations)
    
    
    def __iter__(self):
        return self
    
        
    
    def next(self):
        
        if self.i < self.n:
            value = self.configurations[self.i]
            self.i += 1
            return value
        else:
            raise StopIteration()
        
    
    
    def transform(self, pipeline):
        results = []

        for conf in tqdm(self, desc='configuration_iterator'):
            pipeline.update_configuration(**conf)
            res = pipeline.transform()
            
            results.append([conf, res])
        
        self.results = results

        return results