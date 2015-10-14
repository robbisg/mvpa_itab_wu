import numpy as np
import scipy
from mvpa2.suite import Measure, TransferMeasure
from scipy.spatial.distance import mahalanobis, euclidean
from scipy.stats import pearsonr
import sklearn.covariance
import sklearn.metrics as sklm
import sklearn.svm as svr
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet, \
                        GraphLasso, ShrunkCovariance
from mvpa2.generators.partition import Partitioner
from mvpa2.datasets.base import Dataset
from timeit import itertools
                        


class SimilarityMeasure(Measure):
    
    def __init__(self):
        Measure.__init__(self)
        self.params = dict()
        
    def train(self, ds):
        self.params['mean'] = ds.samples.mean(0)
        self.is_trained = True
    
    def untrain(self):
        self.is_trained = False
        self.params = {}


class MahalanobisMeasure(SimilarityMeasure):
    
    def __init__(self, p=0.05, method=sklearn.covariance.EmpiricalCovariance):
        
        SimilarityMeasure.__init__(self)
        #self.space = 'targets'
        self.p = p
        
        
    def train(self, ds):
        
        super(MahalanobisMeasure, self).train(ds)
        self.params['icov'] = self.method().fit(ds.samples).precision_
        self.is_trained = True
        
    
    def _call(self, ds):
          
        distances = []
        
        mean_ = self.params['mean']
        icov_ = self.params['icov']
          
        for ex in ds:   
            dist_ = mahalanobis(mean_, ex, icov_)
            distances.append(dist_)
        
        chi_sq = scipy.stats.distributions.chi2(mean_.shape[0])
        m_value = chi_sq.isf(self.p)
        
        distances = np.array(distances)
        value = np.count_nonzero((distances ** 2) < m_value)
        
        #space = self.get_space()
        return Dataset(np.array([value]))
    


class CorrelationMeasure(Measure):

    def __init__(self, p=0.05):
        
        Measure.__init__(self)
        #self.space = 'targets'
        self.p = p

    def _call(self, ds):
          
        distances = []
        probabilities = []
        
        mean_ = self.params['mean']
        
        if ds.samples.shape[1] == 1:
            return Dataset(np.array([0.]))
        
        samples = ds.samples
        
        for ex in samples:   
            corr_, p_ = pearsonr(mean_.squeeze(), ex.squeeze())
            distances.append(1 - corr_)
            probabilities.append(p_)
        
        probabilities = np.array(probabilities)
        distances = np.array(distances)
        probabilities = probabilities[distances <= 1]
        value = np.count_nonzero(probabilities < self.p)
        
        #space = self.get_space()
        return Dataset(np.array([value]))

   
    
class TargetCombinationPartitioner(Partitioner):
    
    def __init__(self, attr_task='task', **kwargs):
        
        Partitioner.__init__(self, **kwargs)
        
        self.__attr_task = attr_task
        
    
    def get_partition_specs(self, ds):
        #uniqueattr = ds.sa[self.__attr].unique()
        uniquetask = ds.sa[self.__attr_task].unique
        
        listattr = []
        for task in uniquetask:
            targets = ds[ds.sa[self.__attr_task].value == task].uniquetargets
            listattr.append(targets)                  

        rule = self._get_partition_specs(listattr)
        
        return rule
        
    def _get_partition_specs(self, listattr):
        
        prod = itertools.product(*listattr)
        
        return [('None', [item[0]], [item[1]])  for item in prod]



class RegressionMeasure(Measure):

    def __init__(self, trainer=svr.LinearSVR(), error_fx=sklm.r2_score):
        
        #super(RegressionMeasure, self).__init__()
        Measure.__init__(self)
        self.trainer = trainer
        self.fx = error_fx
        self.mse = sklm.mean_squared_error
        
    def train(self, ds):
        Measure.train(self, ds)
        self.trainer.fit(ds.samples, ds.targets)

    def _call(self, ds):
          
        y_pred = self.trainer.predict(ds.samples)
        
        err_ = self.fx(ds.targets, y_pred)
        mse_ = self.mse(ds.targets, y_pred)
        
        #space = self.get_space()
        return Dataset(np.array([err_, mse_]))
    





