from mvpa_itab.preprocessing.functions import Detrender, TargetTransformer,\
    FeatureWiseNormalizer, FeatureSlicer, SampleSlicer, SampleWiseNormalizer,\
    FeatureStacker
from mvpa_itab.preprocessing.balancing.base import Balancer



def function_mapper(name):
    # TODO: Balancer mappers!
    mapper = {
              'detrending': Detrender,
              'target_trans': TargetTransformer,
              'feature_norm': FeatureWiseNormalizer,
              'feature_slicer': FeatureSlicer,
              'sample_slicer': SampleSlicer,
              'sample_norm': SampleWiseNormalizer,
              'sample_stacker': FeatureStacker,
              'balancer': Balancer
              }
    
    return mapper[name]