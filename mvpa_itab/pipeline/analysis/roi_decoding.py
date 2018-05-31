from mvpa_itab.preprocessing.functions import Node, FeatureSlicer
from sklearn.metrics.scorer import _check_multimetric_scoring
import numpy as np
from sklearn.svm import SVC
from mvpa_itab.io.base import get_ds_data
from sklearn.preprocessing.label import LabelEncoder
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.model_selection._validation import cross_validate


import logging
logger = logging.getLogger(__name__)


class Decoding(Node):
    """Implement decoding analysis using an arbitrary type of classifier.

    Parameters
    -----------

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.
        
    permutation : int. Default is 0.
        The number of permutation to be performed.
        If the number is 0, no permutation is performed.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False
        
    
    Attributes
    -----------

    scores : dict.
            The dictionary of results for each roi selected.
            The key is the union of the name of the roi and the value(s).
            The value is a list of values, the number is equal to the permutations.
            
    """

    def __init__(self, 
                 estimator=SVC(C=1, kernel='linear', gamma='auto'),
                 n_jobs=1, 
                 scoring='accuracy', 
                 cv=LeaveOneGroupOut(),
                 permutation=0,
                 verbose=1):
        
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.permutation = permutation
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        


    def fit(self, ds, cv_attr='chunks', roi='all', roi_values=None, prepro=Node()):
        """Fits the decoding of the dataset.

        Parameters
        -----------
    
        ds : PyMVPA dataset
            The dataset to be used to fit the data
    
        cv_attr : string. Default is 'chunks'.
            The attribute to be used to separate data in the cross validation.
            If cv attribute is specified this parameter is ignored.
            
    
        roi : list of strings. Default is 'all'
            The list of rois to be selected for the analysis. 
            Each string must correspond to a key in the dataset feature attributes.

            
        roi_values : list of tuple, optional. Default is None
            The list of tuple must have as first element the name of roi to be used,
            which should be in the feature attribute of the dataset.
            The second element of the tuple must be a list of values, corresponding to
            the value of the specific roi 
            (e.g. roi_values = [('lateral_ips', [2,4,6]), ('left_precuneus', [10,12])] 
             performs two analysis on lateral_ips and left_precuneus with the
             union of rois with values of 2,4,6 and 10,12 )
             
             
        prepro : Node or PreprocessingPipeline implementing transform, optional.
            A transformation of series of transformation to be performed
            before the decoding analysis is performed.
        
        """

        
        if roi_values == None:
            roi_values = self._get_rois(ds, roi)
        
        self.scores = dict()
        
        # TODO: How to use multiple ROIs
        for r, value in roi_values:
            
            ds_ = FeatureSlicer({r:value}).transform(ds)
            ds_ = prepro.transform(ds_)
            
            logger.info("Dataset shape %s" % (str(ds_.shape)))
            logger.info(ds_.summary(chunks_attr=cv_attr))
            
            scores = self._fit(ds_, cv_attr)
            
            string_value = "_".join([str(v) for v in value])
            self.scores["%s_%s" % (r, string_value)] = scores
        
        return self
    
    
    
    def _get_rois(self, ds, roi):
        """Gets the roi list if the attribute is all"""
        
        rois = [r for r in ds.fa.keys() if r != 'voxel_indices']
        
        if roi != 'all':
            rois = roi
        
        rois_values = []
        
        for r in rois:
            value = [(r, [v]) for v in np.unique(ds.fa[r].value) if v != 0]
            rois_values.append(value)
            
        return list(*rois_values)    
    
    
    def _get_permutation_indices(self, n_samples):
        
        """Permutes the indices of the dataset"""
        
        from numpy.random.mtrand import permutation
        
        if self.permutation == 0:
            return [range(n_samples)]
        
        
        # reset random state
        indices = [range(n_samples)]
        for _ in range(self.permutation):
            idx = permutation(indices[0])
            indices.append(idx)
        
        return indices
            
    
    
    
    
    def _fit(self, ds, cv_attr=None):
        """General method to fit data"""
                   
        self.scoring, _ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)
        
        X, y = get_ds_data(ds)
        y = LabelEncoder().fit_transform(y)
        indices = self._get_permutation_indices(len(y))
                
        values = []
        
        groups = None
        if cv_attr != None:
            groups = LabelEncoder().fit_transform(ds.sa[cv_attr].value)
        
        
        for idx in indices:
            
            y_ = y[idx]

            scores = cross_validate(self.estimator, X, y_, groups,
                                  self.scoring, self.cv, self.n_jobs,
                                  self.verbose, return_estimator=True)
            
            values.append(scores)
        
        return values
    
        
        