import numpy as np
from mvpa_itab.stats.base import CrossValidation
from scipy.stats.stats import zscore
from mvpa_itab.conn.io import ConnectivityLoader
import os
from mvpa_itab.stats.base import Correlation
from mvpa_itab.measure import ranking_correlation
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm.classes import SVR
from sklearn.metrics.regression import mean_squared_error, r2_score
from mvpa_itab.measure import correlation
from mvpa_itab.utils import progress


class Analysis(object):
    
    def setup_analysis(self):
        return self

    
    def run(self):
        return
    
    def __str__(self, *args, **kwargs):
        self.__name__

   
    
class ConnectivityDataLoader(Analysis):
    
    def setup_analysis(self, path, roi_list, directory, conditions, subjects):
        
        self.directory = directory
        self.conditions = conditions
        self.subjects = subjects
        
        conn = ConnectivityLoader(path, self.subjects, self.directory, roi_list)
        conn.get_results(self.conditions)
        self.ds = conn.get_dataset()
        
        return self

    
    def filter(self, filter_, operation='and'):
        """Filter is a dictionary the key is the sample attribute the value is the value
        to filter, if the dictionary has more than one item a logical and is performed"""
        
        if operation == 'and':
            func = np.logical_and
            mask = np.ones_like(self.ds.targets, dtype=np.bool)
        else:
            func = np.logical_or
            mask = np.zeros_like(self.ds.targets, dtype=np.bool)
            
        for k, v in filter_.items():
            mask = func(mask, self.ds.sa[k].value == v)
            
        self.ds = self.ds[mask]
        
        return self
        
    def get_data(self, y_field='expertise'):
        """
        Get X matrix of samples and y array of outcomes
        """
        return self.ds.samples, self.ds.sa[y_field].value    
            


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
    
    
    def run(self, X, y):
        """
        The output is a vector r x n where r is the number
        of repetitions of the splitting method
        """
        
        cv = self.cross_validation
        
        mse_ = []
        set_ = []
        weights_ = []
        samples_ = []
        i = 0
        n = cv.n_iter
        for train_index, test_index in cv:           
            i += 1
            progress(i, n, suffix=' -- regression')
            X_train = X[train_index]
            y_train = y[train_index]
            
            # Feature selection
            self.feature_selection.run(X_train, y_train).select_first(80)
            fset_ = self.feature_selection.ranking
            X_train = X_train[:, fset_]
            
            
            X_test = X[test_index][:,fset_]
            #print X_test.shape
            
            # We suppose only scikit-learn transform algorithms are passed!
            y_predict = self.learner.transform(X_train, y_train).predict(X_test)
            
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
    
    
    def run(self, X, y):
        """
        Runs permutations shuffiling stuff indicated in dimension variable
        """
        
        self.null_dist = []
        
        for i in range(self.n_permutation):
            
            X_, y_ = self.shuffle(X, y)
            permut_value = self.analysis.run(X_, y_)
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
    
    
    def run(self, X, y):
        
        """
        Algorithm should implement a run(X, y) method and the first value should be
        the ranking criterion
        """
        a = self.measure(X)
        values, _ = a.run(X, y)
        
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





class ScriptIterator(object):
    
    
    def setup_analysis(self, **kwargs):
        
        import itertools
            
        args = [arg for arg in kwargs]
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
        
    
    def run(self, pipeline):
        results = []

        for conf in self:
            progress(self.i, self.n)
            print str(np.float(self.i)/self.n)
            pipeline.update_configuration(**conf)
            res = pipeline.run()
            
            results.append([conf, res])
        
        self.results = results

        return results
    
          

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

r = '20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri'
roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str)

iterator_setup = {'directory':['20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri'],
                  'conditions': ['Rest'],
                  'learner': [SVR(kernel='linear', C=1)]
                  
                  }

subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)

_fields = {'path':'/media/robbis/DATA/fmri/monks/0_results', 
               'roi_list':np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str), 
               #'directory':'20150427_124039_connectivity_fmri', 
               #'conditions':['Samatha', 'Vipassana'],
               'directory':'20150427_124039_connectivity_fmri',
               'condition_list':['Rest'],
               'subjects':subjects,
               'filter_':{'meditation':'Rest', 'groups':'E'},
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



'''
from mvpa_itab.regression.base import *
from mvpa_itab.regression.pipelines import FeaturePermutationRegression
pipeline = FeaturePermutationRegression()
pipeline.setup_analysis(**_fields)
iter_ = ScriptIterator()
iter_.setup_analysis(**iterator_setup)
results = iter_.run(pipeline)
'''

