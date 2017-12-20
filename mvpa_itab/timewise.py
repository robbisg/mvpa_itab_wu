from mvpa2.measures.base import CrossValidation, Dataset
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import mean_group_sample
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error
import nibabel as ni
import numpy as np
from mvpa2.clfs.base import Classifier
from mvpa2.generators.resampling import Balancer


class TrialAverager(Classifier):
    
    # Wrapper class, obviously better!
    
    def __init__(self, clf):
        self._clf = clf
        
    def _train(self, ds):
        avg_mapper = mean_group_sample(['trial']) 
        ds = ds.get_mapped(avg_mapper)
        return self._clf._train(ds)
    
    def _call(self, ds):
        # Function overrided to let the results have
        # some dataset attributes
        
        res = self._clf._call(ds)

        if isinstance(res, Dataset):
            for k in ds.sa.keys():
                res.sa[k] = ds.sa[k]
            return res
        else:
            return Dataset(res, sa=ds.sa)


class AverageLinearCSVM(LinearCSVMC):
    """Extension of the classical linear SVM

    Classes inherited from this class gain ability to access
    collections and their items as simple attributes. Access to
    collection items "internals" is done via <collection_name> attribute
    and interface of a corresponding `Collection`.
    """
    
    def __init__(self, C=1, attr='trial'):
        LinearCSVMC.__init__(self, C=1)
        self._attribute = attr
        
    def _train(self, ds):
        avg_mapper = mean_group_sample([self._attribute]) 
        ds = ds.get_mapped(avg_mapper)
        return LinearCSVMC._train(self, ds)
    
    
    def _call(self, ds):
        # Function overrided to let the results have
        # some dataset attributes
        
        res = LinearCSVMC._call(self, ds)

        if isinstance(res, Dataset):
            for k in ds.sa.keys():
                res.sa[k] = ds.sa[k]
            return res
        else:
            return Dataset(res, sa=ds.sa)


class StoreResults(object):
    
    def __init__(self):
        self.storage = []
            
    def __call__(self, data, node, result):
        self.storage.append(node.measure.ca.predictions),
        
    def _post_process(self, ds):
        predictions = np.array(self.storage)
        

class ErrorPerTrial(BinaryFxNode):
    
    def __init__(self, **kwargs):

        BinaryFxNode.__init__(self, fx=mean_mismatch_error, space='targets', **kwargs)


    def _call(self, ds):
        # extract samples and targets and pass them to the errorfx
        targets = ds.sa[self.get_space()].value
        # squeeze to remove bogus dimensions and prevent problems during
        # comparision later on
        values = np.atleast_1d(ds.samples.squeeze())
        if not values.shape == targets.shape:
            # if they have different shape numpy's broadcasting might introduce
            # pointless stuff (compare individual features or yield a single
            # boolean
            raise ValueError("Trying to compute an error between data of "
                             "different shape (%s vs. %s)."
                             % (values.shape, targets.shape))
        err = [self.fx(values[ds.sa.frame == i], targets[ds.sa.frame == i]) 
               for i in np.unique(ds.sa.frame)]
        
        return Dataset(np.array(err).flatten())
    
    
"""
avg = TrialAverager(clf)
cv_storage = StoreResults()
cvte = CrossValidation(avg,
                       HalfPartitioner(),
                       errorfx=ErrorPerTrial(), 
                       #callback=cv_storage,
                       enable_ca=['stats', 'probabilities'])
                       
err = cvte(ds)
"""                       