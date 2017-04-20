from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline
from mvpa_itab.preprocessing.functions import NodeBuilder





def get_preprocessing(**conf):
    
    import collections
    
    conf_pp = conf['preprocessing']
    pipeline_conf = collections.OrderedDict(sorted(conf_pp.items()))
    
    pipeline = PreprocessingPipeline()
    
    for step in pipeline_conf:
        node_dict = pipeline_conf[step]
        node = NodeBuilder(**node_dict).get_node()    
        pipeline.add(node)
        
    
    return pipeline



