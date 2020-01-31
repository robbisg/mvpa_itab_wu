import numpy as np
from scipy.stats.stats import zscore
from mvpa_itab.stats.base import Correlation
from mvpa_itab.measure import ranking_correlation
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm.classes import SVR
from sklearn.metrics.regression import mean_squared_error
from mvpa_itab.measure import correlation
from mvpa_itab.regression.base import ConnectivityDataLoader, FeatureSelectionIterator, \
                                        RegressionAnalysis, PermutationAnalysis

import logging
logger = logging.getLogger(__name__)


class Pipeline(object):
    
    def update_configuration(self, **kwargs):
        
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            
            if arg in self.filter_.keys():
                self.filter_[arg] = kwargs[arg]
                
        return self
    
    
    def save(self):
        import json
        import os
        
        js_dict = dict()
        js_dict['conf'] = str(self._configuration)
        js_dict['data'] = np.array(self.results).tolist()
        
        fname = os.path.join(self.path, 
                            '0_results', 
                            self.directory,
                            'result.json')
        
        with open(fname, 'w') as fp:
            json.dump(js_dict, fp)
            
            
    def __str__(self, *args, **kwargs):
        self.__name__







class RegressionAnalysisPipeline(Pipeline):
    
    _default_fields = {'path':'/media/robbis/DATA/fmri/monks/', 
                       'roi_list':None, 
                       'directory':None, 
                       'condition_list':None,
                       'subjects':None,
                       'filter_':None,
                       'fs_algorithm':Correlation,
                       'fs_ranking_fx':ranking_correlation,
                       'cv_schema':ShuffleSplit(12, 
                                                n_iter=250, 
                                                test_size=0.25),
                       'learner':SVR(kernel='linear', C=1),
                       'error_fx':[mean_squared_error, correlation],
                       'y_field':None,
                       'n_permutations':2000
                       }
    
    
    def setup_analysis(self, **kwargs):
        """
        Fields needed:
        
        path: (e.g. '/media/robbis/DATA/fmri/monks/')
        roi_list: (e.g. load '/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt')
        directory (e.g. '20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri')
        condition_list (e.g. ['Samatha', 'Vipassana'])
        subjects  (e.g. '/media/robbis/DATA/fmri/monks/attributes_struct.txt')
        filter  (e.g. {'meditation':'Samatha', 'groups':'E'})
            -- filter_keywords were used to build the filter
                e.g. meditation: Vipassana will overwrite the filter
        fs_algorithm  (e.g. mvpa_itab.stats.base.Correlation)
        fs_ranking_fx (e.g. mvpa_itab.measures.ranking_correlation)
        cv_schema (e.g. ShuffleSplit(num_exp_subjects, n_iter=cv_repetitions, test_size=cv_fraction)
        learner (e.g. SVR(kernel='linear', C=1))
        error_fx (e.g. [mean_squared_error, correlation])
        y_field (e.g. expertise)
        n_permutations (e.g. 1000)
        """
        self._configuration = dict()
        
        for arg in self._default_fields:
            setattr(self, arg, self._default_fields[arg])
            self._configuration[arg] = self._default_fields[arg]
        
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            self._configuration[arg] = kwargs[arg]
            
        logger.debug(self.y_field)
        
        return self


    def update_configuration(self, **kwargs):
        
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            self._configuration[arg] = kwargs[arg]
            
            if arg in self.filter_.keys():
                self.filter_[arg] = kwargs[arg]
                self._configuration['filter_'][arg] = kwargs[arg]
                
        return self
    

            
    def transform(self):
        
        self.results = []
        
        # Data loading or not maybe not!
        self.loader = ConnectivityDataLoader()
        logger.debug(self.y_field)
        self.X, self.y = self.loader.setup_analysis(self.path, 
                                                    self.roi_list, 
                                                    self.directory, 
                                                    self.condition_list, 
                                                    self.subjects).filter(self.filter_).get_data(y_field=self.y_field)

        
        X = self.X
        y = self.y
        
        logger.debug(X.shape, y.shape)
        
        # This is a preprocessing step
        X = zscore(X, axis=1) # Sample-wise
        y = zscore(np.float_(y))
        
        # This is another pipeline to be included
        self.fs = FeatureSelectionIterator()
        self.fs.setup_analysis(self.fs_algorithm, self.fs_ranking_fx).transform(X, y).select_first(80)
        
        
        # This is the analysis
        self.reg = RegressionAnalysis().setup_analysis(self.cv_schema, 
                                                       self.learner, 
                                                       self.error_fx)
        # Speedup stuff
        schema = ShuffleSplit(12, 
                              n_iter=1, 
                              test_size=0.25)
        
        self.perm_reg = RegressionAnalysis().setup_analysis(schema, 
                                                  self.learner, 
                                                  self.error_fx)
        
        # Permutator the Analysis object
        self.perm = PermutationAnalysis().setup_analysis(self.reg, 
                                                    n_permutation=self.n_permutations,
                                                    dimension='features')
        

        
        for i, set_ in enumerate(self.fs):
            
            if i > 78:
                X_ = X[:,set_]
                y_ = y
                            
                reg_res = self.reg.transform(X_, y_) # To be selected
                n_dist = self.perm.transform(X_, y_)
                
                p_res = self.perm.pvalues(reg_res)
                
                self.results.append([reg_res, n_dist, p_res])

        #self.save()
        return self.results

        

class NoPermutationPipeline(RegressionAnalysisPipeline):
    
    
    def transform(self):
        
        self.loader = ConnectivityDataLoader()
        self.X, self.y = self.loader.setup_analysis(self.path, 
                              self.roi_list, 
                              self.directory, 
                              self.condition_list, 
                              self.subjects).filter(self.filter_).get_data()
        
        X = self.X
        y = self.y
                             
        X = zscore(X, axis=1) # Sample-wise
        y = zscore(np.float_(y))       
        
        self.fs = FeatureSelectionIterator()
        self.fs.setup_analysis(self.fs_algorithm, 
                          self.fs_ranking_fx).transform(X, y).select_first(80)
        
        
        
        self.reg = RegressionAnalysis().setup_analysis(self.cv_schema, 
                                                  self.learner, 
                                                  self.error_fx)
        self.results = []
        for set_ in self.fs:
            X_ = X[:,set_]
            y_ = y
                        
            reg_res = self.reg.transform(X_, y_) # To be selected
                       
            self.results.append([reg_res])
        
        self.save()
        
        return self.results


class FeaturePermutationRegression(RegressionAnalysisPipeline):
    
    def transform(self):
        
        logger.debug(self.filter_)
        
        # 
        self.loader = ConnectivityDataLoader()
        self.X, self.y = self.loader.setup_analysis(self.path, 
                              self.roi_list, 
                              self.directory, 
                              self.condition_list, 
                              self.subjects).filter(self.filter_).get_data(y_field=self.y_field)
                              
        
        logger.debug(self.X.shape)
        logger.debug(self.y)
                
        X = self.X
        y = self.y
                             
        X = zscore(X, axis=1) # Sample-wise
        y = zscore(np.float_(y))       
        
        self.fs = FeatureSelectionIterator().setup_analysis(self.fs_algorithm, 
                                                            self.fs_ranking_fx)
        
        
        self.reg = RegressionAnalysis().setup_analysis(self.cv_schema, 
                                                       self.learner, 
                                                       self.error_fx,
                                                       feature_selection=self.fs)
        
        self.perm = PermutationAnalysis().setup_analysis(self.reg, 
                                                         n_permutation=self.n_permutations,
                                                         dimension='labels')
        
        self.results = []

                
        reg_res = self.reg.transform(X, y) # To be selected
        
        perm_res = self.perm.transform(X, y)
                       
        self.results.append([reg_res, perm_res])
        
        #self.save()
        
        return self.results
            

   