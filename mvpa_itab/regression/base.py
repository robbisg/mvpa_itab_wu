import numpy as np
from mvpa_itab.stats import CrossValidation
from scipy.stats.stats import zscore
from mvpa_itab.conn.utils import ConnectivityTest
import os
from mvpa_itab.stats import Correlation
from mvpa_itab.measure import ranking_correlation
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm.classes import SVR
from sklearn.metrics.regression import mean_squared_error
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
        
        conn = ConnectivityTest(path, self.subjects, self.directory, roi_list)
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
  
       
    def setup_analysis(self, cv_schema, learner, error_fx):
        self.analysis = CrossValidation(cv_schema, learner, error_fx=error_fx)
        return self
    
    
    def run(self, X=None, y=None):
        
        '''
        if X == y == None:
            X = self.X
            y = self.y
        '''
        self.results = self.analysis.run(X, y)
        return self.results


        
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
        self.dimension = 'labels'
        
        return self
        
        
    def shuffle(self, X, y):
        
        from numpy.random.mtrand import permutation
        
        if self.dimension == 'labels':
            arg_x = range(X.shape[1])
            arg_y = permutation(range(len(y)))
        else:
            arg_x = permutation(range(X.shape[1]))
            arg_y = range(len(y))
            
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
    
    def __init__(self):
        self.i = 0
    
    
    def setup_analysis(self, algorithm, ranking_fx):
        
        self.analysis = algorithm
        self.ranker = ranking_fx
        
        return self
    
    
    def __iter__(self):
        return self
    
    
    def run(self, X, y):
        
        """
        Algorithm should implement a run(X, y) method and the first value should be
        the ranking criterion
        """
        a = self.analysis(X)
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

iterator_setup = {'directory':['20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri',
                            '20150427_124039_connectivity_fmri'],
                  'conditions': ['Samatha', 'Vipassana'],
                  'learner': [SVR(kernel='linear', C=1), 
                              SVR(kernel='rbf', C=1)]
                  
                  }

subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)

_test_fields = {'path':'/media/robbis/DATA/fmri/monks/', 
               'roi_list':np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str), 
               'directory':'20150427_124039_connectivity_fmri', 
               'conditions':['Samatha', 'Vipassana'],
               'subjects':subjects,
               'filter_':{'meditation':'Samatha', 'groups':'E'},
               'fs_algorithm':Correlation,
               'fs_ranking_fx':ranking_correlation,
               'cv_schema':ShuffleSplit(12, 
                                        n_iter=250, 
                                        test_size=0.25),
               'learner':SVR(kernel='linear', C=1),
               'error_fx':[mean_squared_error, correlation],
               'y_field':'expertise',
               'n_permutations':200
               }



'''
loader = ConnectivityDataLoader()
conditions = ['Samatha', 'Vipassana']
result_dir = ['dir1', 'dir2']

kwargs = {'cond':conditions, 'dir':result_dir}

loader.setup_analysis(path, roi_list, directory, conditions, subjects)

filter_ = {'meditation':'Samatha', 'group':'E'}

loader.filter(filter_)

X = loader.ds.samples
y = np.float_(loader.ds.sa.expertise)*0.01
'''

