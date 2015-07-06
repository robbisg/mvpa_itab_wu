from mvpa2.suite import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.svm import LinearCSVMC
import numpy as np
from collections import Counter
from scipy.stats.stats import ttest_1samp, ttest_ind, pearsonr
from numpy.random.mtrand import permutation


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
            fp[i,:] = self.analysis.run(p_labels)
            #fp[i,:] = self.analysis.run(ds_p)
        
            #null_dist.append(value_)
        
        #fp = np.array(null_dist)
        self.null_dist = fp
    
        return fp
    
    def run(self, ds, labels):
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
            #fp[i,:] = self.analysis.run(p_labels)
            fp[i,:] = self.analysis.run(ds_p)
        
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

        
    def run(self, labels):
        
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
        
    def run(self, ds, seed):
        
        ds = self._dataset.T
        y = seed
        
        corr = []
        for x in ds:
            r_, _ = pearsonr(x, y)
            
            corr.append(r_)
            
        corr = np.array(corr)
        
        return corr, _
            
            
    
    
def permutation_test(ds, labels, analysis, n_permutation=1000):
    
    null_dist = []
    
    for i in range(n_permutation):
        p_labels = permutation(labels)
        t_, _ = analysis.run(ds, p_labels)
        
        null_dist.append(t_) 
    
    return np.array(null_dist)   
        
        
        
   