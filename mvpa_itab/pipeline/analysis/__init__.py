from mvpa_itab.preprocessing.functions import Node
from mvpa_itab.io.base import get_ds_data
from sklearn.model_selection._validation import cross_val_score



class Analysis(object):
    
    def __init__(self, pipeline, cross_validation, transformer=Node(), **cv_kwargs):
        
        self._transformer = transformer
        self._pipeline = pipeline
        
        # Should we use an ad-hoc cross_validator
        self._cross_validation = cross_validation
        
        
        object.__init__(self, **cv_kwargs)
        
    
    def fit(self, ds):
        
        ds_ = self._transformer.transform(ds)
        
        # Is y available for sklearn
        X, y = get_ds_data(ds_)
        
        self._scores = cross_val_score(self._pipeline, X, y, cv=self._cross_validation)
        
        return self
    