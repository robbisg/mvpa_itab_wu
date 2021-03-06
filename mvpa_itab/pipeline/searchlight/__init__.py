from nilearn.image.resampling import coord_transform
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from sklearn.metrics.scorer import _check_multimetric_scoring
from nilearn.decoding.searchlight import search_light
import numpy as np
from nilearn import masking
from sklearn.svm import SVC
from mvpa_itab.pipeline.searchlight.utils import _get_affinity, check_proximity,\
    load_proximity, save_proximity
from mvpa_itab.io.utils import get_ds_data
from sklearn.preprocessing.label import LabelEncoder
from sklearn.model_selection._split import LeaveOneGroupOut
import os

import logging
from mvpa_itab.pipeline import Analyzer
from mvpa_itab.io.utils import save_map
from mvpa_itab.results import make_dir
logger = logging.getLogger(__name__)



def get_seeds(ds, radius):
    
    if check_proximity(ds, radius):
        return load_proximity(ds, radius)
    
    
    # Get the seeds
    process_mask_coords = ds.fa.voxel_indices.T
    process_mask_coords = coord_transform(
        process_mask_coords[0], process_mask_coords[1],
        process_mask_coords[2], ds.a.imgaffine)
    process_mask_coords = np.asarray(process_mask_coords).T
    
    seeds = process_mask_coords
    coords = ds.fa.voxel_indices
    logger.info("Building proximity matrix...")
    A = _get_affinity(seeds, coords, radius, allow_overlap=True, affine=ds.a.imgaffine)
    
    save_proximity(ds, radius, A)
    
    return A





class SearchLight(Analyzer):
    """Implement search_light analysis using an arbitrary type of classifier.
    This is a wrapper of the nilearn algorithm to work with pymvpa dataset.

    Parameters
    -----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

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
                 radius=9.,
                 estimator=SVC(C=1, kernel='linear'),
                 n_jobs=1, 
                 scoring='accuracy', 
                 cv=LeaveOneGroupOut(), 
                 permutation=0,
                 verbose=1):
        
        self.radius = radius
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.permutation = permutation
        
        self.scoring = scoring
        
        self.cv = cv
        self.verbose = verbose
        
        Analyzer.__init__(self, name='searchlight')



    def _get_permutation_indices(self, n_samples):
        
        from numpy.random.mtrand import permutation
        
        if self.permutation == 0:
            return [range(n_samples)]
        
        
        # reset random state
        indices = [range(n_samples)]
        for _ in range(self.permutation):
            idx = permutation(indices[0])
            indices.append(idx)
        
        return indices

    


    def fit(self, ds, cv_attr='chunks'):
        """
        Fit the searchlight

        """
        
        A = get_seeds(ds, self.radius)
        
        estimator = self.estimator
            
        self.scoring, _ = _check_multimetric_scoring(estimator, scoring=self.scoring)
        
        X, y = get_ds_data(ds)
        y = LabelEncoder().fit_transform(y)
        groups = LabelEncoder().fit_transform(ds.sa[cv_attr].value)
        
        
        values = []
        indices = self._get_permutation_indices(len(y))
        
        for idx in indices:
            y_ = y[idx]        
            scores = search_light(X, y_, estimator, A, groups,
                                  self.scoring, self.cv, self.n_jobs,
                                  self.verbose)
            
            values.append(scores)
        
        self.scores = values
        
        self._info = self._store_ds_info(ds, cv_attr=cv_attr)

        return self
    
    
    
    
    
    def save(self, path=None, save_cv=True):
        
        map_type = ['avg', 'cv']
        
        path = Analyzer.save(self, path)
                    
        
        fx = self._info['a'].mapper.reverse1
        affine = self._info['a'].imgaffine
        
        
        
        for i in range(len(self.scores)):
            for j, img_dict in enumerate(self.scores[i]):
                for score, image in img_dict.items():
                    
                    # TODO: Better use of cv and attributes for leave-one-subject-out
                    if map_type[j] == 'cv' and save_cv == False:
                        continue
                    
                    filename = "%s_perm_%04d_%s.nii.gz" %(score, i, map_type[j])
                    logger.info("Saving %s" %(filename))
                    filename = os.path.join(path, filename)
                    save_map(filename, fx(image), affine)
                    
    
    
    
    def _get_analysis_info(self):
        
        info = Analyzer._get_analysis_info(self)
        info['radius'] = self.radius
        
        return info
        
    
        
        
        
        
        
        
        
            
            
            