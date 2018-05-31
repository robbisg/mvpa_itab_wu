from mvpa_itab.preprocessing.functions import Node, FeatureSlicer
from sklearn.metrics.scorer import _check_multimetric_scoring
import numpy as np
from sklearn.svm import SVC
from mvpa_itab.io.base import get_ds_data
from sklearn.preprocessing.label import LabelEncoder
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.model_selection._validation import cross_validate,\
    permutation_test_score

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

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False
    """

    def __init__(self, 
                 estimator=SVC(C=1),
                 n_jobs=1, 
                 scoring='accuracy', 
                 cv=LeaveOneGroupOut(),
                 feature_selection=None,
                 permutation=0,
                 verbose=1):
        
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.permutation = permutation
        self.scoring = scoring
        self.fsel = feature_selection
        self.cv = cv
        self.verbose = verbose
        


    def fit(self, ds, cv_attr='chunks', roi='all', roi_values=None):
        """Fit the decoding
        
        all: fit all rois within fa and for each value
        """
        rois = [r for r in ds.fa.keys() if r != 'voxel_indices']
        if roi != 'all':
            rois = roi
      
        roi_values = self._get_rois(ds, rois)
        
        self.scores = dict()
        for r, value in roi_values:
            
            ds_ = FeatureSlicer({r:[value]}).transform(ds)
            logger.info("Dataset shape %s" % (str(ds_.shape)))
            scores = self._fit(ds_, cv_attr)
            self.scores["%s_%02d" % (r, int(value))] = scores
        
        return self
    
    
    def _get_rois(self, ds, rois):
        
        rois_values = []
        
        for r in rois:
            value = [(r, v) for v in np.unique(ds.fa[r].value) if v != 0]
            rois_values.append(value)
            
        return list(*rois_values)    
    
    
    def _get_permutation_indices(self, n_samples):
        
        from numpy.random.mtrand import permutation
        
        if self.permutation == 0:
            return [range(n_samples)]
        
        
        # random state bla bla
        indices = [range(n_samples)]
        for i in range(self.permutation):
            idx = 
            
    
    
    
    
    def _fit(self, ds, cv_attr=None):
                           
        self.scoring, _ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)
        
        
        values = []
        
        for 
        
        X, y = get_ds_data(ds)
        
        y = LabelEncoder().fit_transform(y)
        
        #groups = None
        groups = LabelEncoder().fit_transform(ds.sa[cv_attr].value)
        
        if self.permutation != 0:
            scores = {score:permutation_test_score(self.estimator, X, y, groups, cv=self.cv, 
                                            n_permutations=100, n_jobs=self.n_jobs, 
                                            scoring=self.scoring[score]) for score in self.scoring.keys()}
        else:
            scores = cross_validate(self.estimator, X, y, groups,
                                  self.scoring, self.cv, self.n_jobs,
                                  self.verbose, return_estimator=True)
        
        return scores
    
        
        