from mvpa2.suite import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.svm import LinearCSVMC
import numpy as np
from collections import Counter
from scipy.stats.stats import ttest_1samp, ttest_ind, pearsonr
from numpy.random.mtrand import permutation
from sklearn.metrics import mean_squared_error
from mvpa_itab.utils import progress

def cross_validate(ds, clf, partitioner, permuted_labels):
    
    partitions = partitioner.generate(ds)
    
    accuracies = []
    true_labels = ds.targets.copy()
    
    for p in partitions:
        
        training_mask = p.sa.partitions == 1
        
        ds.targets[training_mask] = permuted_labels[training_mask]
        
        c = Counter(ds.targets[training_mask])
        
        assert len(np.unique(np.array(c.values()))) == 1
        assert (ds.targets[~training_mask] == true_labels[~training_mask]).any()
        
        
        clf.train(ds[training_mask])
        predictions = clf.predict(ds[~training_mask])

        good_p = np.count_nonzero(np.array(predictions) == ds.targets[~training_mask])

        acc = good_p/np.float(len(ds.targets[~training_mask]))
        
        accuracies.append(acc)
        
        ds.targets = true_labels
    
    return np.array(accuracies)


def randomize_labels(ds):
    '''
    Procedure to randomize labels in each chunk.
    
    ------------------------
    Parameters
        ds: The dataset with chunks and targets to be shuffled
        
        out: list of shuffled labels 
    
    '''
    
    labels = ds.targets.copy()
    
    for fold in np.unique(ds.chunks):
        
        mask_chunk = ds.chunks == fold
        
        labels[mask_chunk] = np.random.permutation(ds.targets[mask_chunk])
        
    return labels
        
class PermutationTest(object):
    
    def __init__(self, 
                 analysis, 
                 permutation_axis=0, 
                 n_permutation=1000):
        
        """permutation dimension indicates the axis to permute on ds"""
        
        self.analysis = analysis   # check to be done
        self.n_permutation = n_permutation
        self._axis = permutation_axis
    
    
    
    def shuffle(self, ds, labels):
        # Temporary function
        fp = np.memmap('/media/robbis/DATA/perm.dat',
                       dtype='float32', 
                       mode='w+',
                       shape=(self.n_permutation, 
                              ds.shape[1],
                              ds.shape[2])
                       )
        #print fp.shape
        for i in range(self.n_permutation):
            p_labels = permutation(labels)
            #print p_labels
            fp[i,:] = self.analysis.transform(p_labels)
            #fp[i,:] = self.analysis.transform(ds_p)
        
            #null_dist.append(value_)
        
        #fp = np.array(null_dist)
        self.null_dist = fp
    
        return fp
    
    
    def transform(self, ds, labels):
        # What the fuck is labels???
        #null_dist = []
        
        fp = np.memmap('/media/robbis/DATA/perm.dat',
                       dtype='float32', 
                       mode='w+',
                       shape=(self.n_permutation, 
                              len(labels),
                              len(labels))
                       )
        
        for i in range(self.n_permutation):
            ds_p = self.ds_simple_permutation(ds)
            #fp[i,:] = self.analysis.transform(p_labels)
            fp[i,:] = self.analysis.transform(ds_p)
        
            #null_dist.append(value_)
        
        #fp = np.array(null_dist)
        self.null_dist = fp
    
        return fp
    
    
    
    def p_values(self, true_values, null_dist=None, tails=0):
        """tails = [0, two-tailed; 1, upper; -1, lower]"""
        
        #check stuff
                    
        if null_dist == None:
            null_dist = self._null_dist
        
        
        if tails == 0:
            count_ = np.abs(null_dist) > np.abs(true_values)
        else:
            count_ = (tails * null_dist) > (tails * true_values)

            
        p_values = np.sum(count_, axis=0) / np.float(self.n_permutation)
        
        self._p_values = p_values
        
        return p_values    
    
    
    
    def ds_permutation(self, ds):
        
        from datetime import datetime
        start = datetime.now()

        dim = self._axis
        
        #check if dimension is coherent with ds shape
        new_indexing = []
        for i in range(len(ds.shape)):
            ind = range(ds.shape[i])
            if i == dim:
                ind = list(permutation(ind))
            
            new_indexing.append(ind)
        
        ds_ = ds[np.ix_(*new_indexing)]
        
        finish = datetime.now()
        print (finish - start)
        
        return ds_
    
    
    
    def ds_simple_permutation(self, ds):
        
        from datetime import datetime
        start = datetime.now()
        
        dim = self._axis
        ind = range(ds.shape[dim])
        
        if dim == 0:
            ds_ = ds[ind]
        elif dim == 1:
            ds_ = ds[:,ind]
        elif dim == 2:
            ds_ = ds[:,:,ind]
             
        finish = datetime.now()
        print (finish - start)
        
        return ds_




class TTest(object):
    
    
    def __init__(self, ds, conditions=None, sample_value=None):
        self.dataset = ds
        self.conditions=conditions
        self.sample_value=sample_value
        print 'version new'
        print len(conditions)

        
    def transform(self, labels):
        
        ds = self.dataset
        conditions = self.conditions
        single_value = self.sample_value
        
        if conditions == single_value == None:
            raise ValueError()
        elif len(conditions)>2:
            raise ValueError()
    
        if single_value != None:
            t, p = ttest_1samp(ds, single_value, axis=0)
            return t, p
    
    
        t, p = ttest_ind(ds[labels == conditions[0]],
                     ds[labels == conditions[1]],
                     axis=0
                     )
        #print ds.shape
        #print t.shape
        t[np.isnan(t)] = 1
        return t



class Correlation(object):
    
    
    def __init__(self, ds):
        
        self._dataset = ds
        
    def transform(self, ds, seed):
        """
        
        """
        
        #ds = self._dataset.T
        ds = ds.T
        y = seed
        
        corr = []
        for x in ds:

            r_, _ = pearsonr(x, y)
            
            corr.append(r_)
            
        corr = np.array(corr)
        
        return corr, _
    
    def __str__(self, *args, **kwargs):
        self.__name__
            
            
            
class SKLRegressionWrapper(object):
    
    def __init__(self, algorithm, error_fx=mean_squared_error):
        self.is_trained = False
        self.algorithm = algorithm
        self.error_fx = error_fx
        self._y_pred = None
        
    def train(self, X, y):
        
        self.algorithm.transform(X, y)
        self.is_trained = True
        return
        
    def predict(self, X):
        
        if self.is_trained == False:
            raise ValueError()
        
        self._y_pred = self.algorithm.predict(X)
        
        return self._y_pred
        
    def evaluate(self, y_true, y_pred=None):
        
        if y_pred != None:
            y_pred = self.y_pred
        
        return self.error_fx(y_true, y_pred)
        
        
    def transform(self, X, y, error_function=mean_squared_error, **kwargs):
        
        
        return
        
        
        
class CrossValidation(object):
    
    def __init__(self, method, algorithm, error_fx=[mean_squared_error]):
        
        self.method = method
        self.algorithm = algorithm
        
        # List of error functions
        self.errorfx = error_fx
        
        
    def transform(self, X, y):
        """
        The output is a vector r x n where r is the number
        of repetitions of the splitting method
        """
        
        cv = self.method
        
        # Check if all elements could be selected
        if cv.n != len(y):
            cv.n = len(y)
        
        mse_ = []
        for train_index, test_index in cv:           
                
            X_train = X[train_index]
            y_train = y[train_index]
            
            # Feature selection
            
            
            X_test = X[test_index]
            
            # We suppose only scikit-learn transform algorithms are passed!
            y_predict = self.algorithm.transform(X_train, y_train).predict(X_test)
            
            errors = []
            for error_ in self.errorfx:
                err_ = error_(y[test_index], y_predict)
                errors.append(err_)
        
            mse_.append(errors)
    
        self.result = np.array(mse_)
        
        return self.result




class RegressionPermutation(object):
    
    def __init__(self, 
                 analysis,
                 n_permutation=1000,
                 print_progress=True):
        
        """permutation dimension indicates the axis to permute on ds"""
        
        self.analysis = analysis   # check to be done
        self.n_permutation = n_permutation
        self.print_progress = print_progress
    
    def shuffle(self, y):
        return permutation(y)
    
    def transform(self, X, y):

        null_dist = []

        for i in range(self.n_permutation):
            if self.print_progress:
                progress(i, self.n_permutation)
            y_perm = self.shuffle(y)
            value = self.analysis.transform(X, y_perm)
        
            null_dist.append(value)
        
        null_dist = np.array(null_dist)
        self.null_dist = null_dist
    
    
    def p_values(self, true_values, null_dist=None, tails=0):
        """tails = [0, two-tailed; 1, upper; -1, lower]"""
        
        #check stuff
                    
        if null_dist == None:
            null_dist = np.mean(self.null_dist, axis=1)
        
        
        if tails == 0:
            count_ = np.abs(null_dist) > np.abs(true_values)
        else:
            count_ = (tails * null_dist) > (tails * true_values)

            
        p_values = np.sum(count_, axis=0) / np.float(self.n_permutation)
        
        self._p_values = p_values
        
        return p_values    
    
    
    
    def ds_permutation(self, ds):
        
        from datetime import datetime
        start = datetime.now()

        dim = self._axis
        
        #check if dimension is coherent with ds shape
        new_indexing = []
        for i in range(len(ds.shape)):
            ind = range(ds.shape[i])
            if i == dim:
                ind = list(permutation(ind))
            
            new_indexing.append(ind)
        
        ds_ = ds[np.ix_(*new_indexing)]
        
        finish = datetime.now()
        print (finish - start)
        
        return ds_
    
    
    
    def ds_simple_permutation(self, ds):
        
        from datetime import datetime
        start = datetime.now()
        
        dim = self._axis
        ind = range(ds.shape[dim])
        
        if dim == 0:
            ds_ = ds[ind]
        elif dim == 1:
            ds_ = ds[:,ind]
        elif dim == 2:
            ds_ = ds[:,:,ind]
             
        finish = datetime.now()
        print (finish - start)
        
        return ds_
        
        
                
def permutation_test(ds, labels, analysis, n_permutation=1000):
    
    null_dist = []
    
    for _ in range(n_permutation):
        p_labels = permutation(labels)
        t_, _ = analysis.transform(ds, p_labels)
        
        null_dist.append(t_) 
    
    return np.array(null_dist)   
        

        
   