import numpy as np
import os
from mvpa_itab.stats.base import Correlation
from mvpa_itab.measure import ranking_correlation
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm.classes import SVR
from sklearn.metrics.regression import mean_squared_error
from mvpa_itab.measure import correlation
from mvpa_itab.utils import progress
from tqdm import *


import logging
logger = logging.getLogger(__name__)


class Analysis(object):
    
    def setup_analysis(self):
        return self

    
    def transform(self):
        return
    
    def __str__(self, *args, **kwargs):
        self.__name__

            


class RegressionAnalysis(Analysis):      
  
       
    def setup_analysis(self, 
                       cv_schema, 
                       learner, 
                       error_fx, 
                       feature_selection=None
                       ):
        
        self.cross_validation = cv_schema
        self.learner = learner
        self.error_fx = error_fx
        self.feature_selection = feature_selection
        
        return self
    
    
    def transform(self, X, y, sets=None):
        """
        The output is a vector r x n where r is the number
        of repetitions of the splitting method
        """
        
        if sets==None:
            cv = self.cross_validation
            
        mse_ = []
        set_ = []
        weights_ = []
        samples_ = []
        i = 0
        n = cv.n_iter
        
        for train_index, test_index in tqdm(cv, desc="regression cross-validation"):           
            i += 1
            #progress(i, n, suffix=' -- regression')
            
            logger.debug(X.shape)
            logger.debug(y.shape)
            
            X_train = X[train_index]
            y_train = y[train_index]
            
            # Feature selection
            self.feature_selection.transform(X_train, y_train).select_first(80)
            fset_ = self.feature_selection.ranking
            X_train = X_train[:, fset_]
            
            
            X_test = X[test_index][:,fset_]
            #print X_test.shape
            
            # We suppose only scikit-learn transform algorithms are passed!
            y_predict = self.learner.fit(X_train, y_train).predict(X_test)
            
            errors = []
            for error_ in self.error_fx:
                err_ = error_(y[test_index], y_predict)
                errors.append(err_)
        
            mse_.append(errors)
            set_.append(fset_)
            weights_.append(self.learner.coef_)
            samples_.append(train_index)
        
        self.result = [mse_, set_, weights_, samples_]
        
        return self.result


        
class PermutationAnalysis(Analysis):
    
    
    def transform(self, X, y):
        """
        Runs permutations shuffiling stuff indicated in dimension variable
        """
        
        self.null_dist = []
        
        for i in tqdm(range(self.n_permutation), desc="permutation"):
            
            X_, y_ = self.shuffle(X, y)
            permut_value = self.analysis.transform(X_, y_)
            self.null_dist.append(permut_value)
            progress(i, self.n_permutation, suffix='')
            
        self.null_dist = np.array(self.null_dist)
        return self.null_dist
    
    
    def setup_analysis(self, main_analysis, n_permutation=2000, dimension='labels'):
        
        self.n_permutation = n_permutation
        self.analysis = main_analysis
        self.dimension = dimension
        
        return self
        
        
    def shuffle(self, X, y):
        
        from numpy.random.mtrand import permutation
        
        if self.dimension == 'labels':
            arg_x = range(X.shape[1])
            arg_y = permutation(range(len(y)))
        else:
            arg_x = permutation(range(X.shape[1]))
            arg_y = range(len(y))
        
        
        #print arg_x, arg_y
        
        return X[:, arg_x], y[arg_y]
    
    
    def pvalues(self, true_values, null_dist=None, tails=0):
        """tails = [0, two-tailed; 1, upper; -1, lower]"""

        if null_dist == None:
            null_dist = self.null_dist
        
        if tails == 0:
            count_ = np.abs(null_dist) > np.abs(true_values)
        else:
            count_ = (tails * null_dist) > (tails * true_values)
  
        p_values = np.sum(count_, axis=0) / np.float(self.n_permutation)
        
        self.p_values = p_values
        return p_values

        
        
class FeatureSelectionIterator(object):
    
    def __init__(self, i=0):
        self.i = i
    
    
    def setup_analysis(self, measure, ranking_fx):
        
        self.measure = measure
        self.ranker = ranking_fx
        
        return self
    
    
    def __iter__(self):
        return self
    
    
    def transform(self, X, y):
        
        """
        Algorithm should implement a transform(X, y) method and the first value should be
        the ranking criterion
        """
        a = self.measure(X)
        values, _ = a.transform(X, y)
        
        self.ranking = self.ranker(values)
        self.i = 0
        self.n = len(values)
        
        return self
    
    
    def select_first(self, n):
        "If n<1 it indicates a percentage"
        
        if n<1:
            number = np.rint(len(self.ranking) * n)
        else:
            number = n
            
        self.ranking = self.ranking[:number]
        self.n = number
        

        return self
        

    def next(self):
        "Returns indexes of features to be selected"
        
        if self.i < self.n:
            self.i += 1
            return self.ranking[:self.i]
        else:
            raise StopIteration()






    
          

class RegressionSaver(object):
    
    
    def setup_analysis(self, pipeline):
        self.pipeline = pipeline
    
    
    def save(self):
        
        path = os.path.join(self.pipeline.path, 'regression')
        os.system('mkdir '+path)
        
        for array_ in self.pipeline.result:
            
            results_ = array_[0]
            null_dist = array_[1]
            p_values = array_[2]
            

            
            
###### Script ######
'''
path = "/media/robbis/DATA/fmri/monks"
#path = "/home/robbis/fmri/monks"
dir_ = ""

r = '20151103_132009_connectivity_filtered_first_filtered_after_each_run_no_gsr_findlab_fmri'
roi_list = np.loadtxt(os.path.join(path, dir_, 'findlab_rois.txt'), 
                      delimiter=',',
                      dtype=np.str)

iterator_setup = {'directory':['20151103_132009_connectivity_filtered_first_filtered_after_each_run_no_gsr_findlab_fmri'],
                  'conditions': ["Samatha", "Vipassana"],
                  'learner': [SVR(kernel='linear', C=1)],
                  "y_field": ['age']                  
                  }

subjects = np.loadtxt(os.path.join(path, 'attributes_struct.csv'),
                      delimiter=",",
                      dtype=np.str)

_fields = {'path': os.path.join(path, '0_results'), 
               'roi_list':np.loadtxt(os.path.join(path, 'findlab_rois.txt'), 
                      delimiter=',',
                      dtype=np.str), 
               #'directory':'20150427_124039_connectivity_fmri', 
               'condition_list':['Samatha', 'Vipassana'],
               'directory':'20151103_132009_connectivity_filtered_first_filtered_after_each_run_no_gsr_findlab_fmri',
               #'condition_list':['Rest'],
               'subjects':subjects,
               'filter_':{'groups':'E'},
               'fs_algorithm':Correlation,
               'fs_ranking_fx':ranking_correlation,
               'cv_schema':ShuffleSplit(12, 
                                        n_iter=250, 
                                        test_size=0.25),
               'learner':SVR(kernel='linear', C=1),
               'error_fx':[mean_squared_error, correlation],
               'y_field':'expertise',
               'n_permutations':1000
               }




from mvpa_itab.regression.base import *
from mvpa_itab.regression.pipelines import FeaturePermutationRegression
pipeline = FeaturePermutationRegression()
pipeline.setup_analysis(**_fields)
iter_ = ScriptIterator()
iter_.setup_analysis(**iterator_setup)
results = iter_.transform(pipeline)
pickle.dump(results, file("/home/robbis/fmri/monks/results_regression_age_perm_1000_cv_250.pyobj", "w"))
'''

