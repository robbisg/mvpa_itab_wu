import numpy as np
from nitime.timeseries import TimeSeries
from nitime.analysis import SeedCorrelationAnalyzer, BaseAnalyzer
from scipy.spatial.distance import euclidean
from nitime import descriptors as desc
from numpy.random.mtrand import permutation
import matplotlib.pyplot as pl

import sys

class SeedAnalyzer(BaseAnalyzer):
    
    def __init__(self, seed_time_series=None, 
                 target_time_series=None, 
                 measure=euclidean, 
                 **kwargs):
        
        """measure is a function which takes two arrays
        and gives a number as output"""
        
        
        self.seed = seed_time_series
        self.target = target_time_series
        self._measure = measure
        self.kwargs = kwargs
    
        
    @desc.setattr_on_read
    def measure(self):

        kwargs = self.kwargs
        # If there is more than one channel in the seed time-series:
        if len(self.seed.shape) > 1:

            # Preallocate results
            Cxy = np.empty((self.seed.data.shape[0],
                            self.target.data.shape[0]), dtype=np.float)

            for seed_idx, this_seed in enumerate(self.seed.data):
                res = []
                for single_ts in self.target.data:
                    
                    try:
                        measure_sim = self._measure(this_seed, single_ts, **kwargs)
                        res.append(measure_sim)
                    except TypeError, _:
                        raise TypeError('Class measure must take 2 arguments!')
                
                Cxy[seed_idx] = np.array(res)
        # In the case where there is only one channel in the seed time-series:
        else:
            #To correct!!!
            len_target = self.target.shape[0]
            rr = [self._measure(self.seed.data, self.target.data[i]) for i in range(len_target)]
            Cxy = np.array(rr)
            
            
        return Cxy.squeeze()


class SeedCorrelationAnalyzerWrapper(SeedCorrelationAnalyzer, SeedAnalyzer):
    
    def __init__(self, seed_time_series=None, target_time_series=None):

        SeedCorrelationAnalyzer.__init__(self, 
                                         seed_time_series,
                                         target_time_series)
    @desc.setattr_on_read
    def measure(self):
        
        return self.corrcoef


class SeedSimilarityAnalysis(object):
    
    def __init__(self, 
                 seed_ds=None, 
                 seed_analyzer=SeedCorrelationAnalyzerWrapper,
                 **kwargs):
        
        self.seed_ds = seed_ds
        self.seed_analyzer = seed_analyzer
        self.kwargs = kwargs
        return
    
    def run(self, target_ds):
        
        seed_ds = self.seed_ds
        
        ts_seed = TimeSeries(seed_ds, sampling_interval=1.)
        ts_target = TimeSeries(target_ds, sampling_interval=1.)
        
        kwargs = self.kwargs
        
        seed_analyzer = self.seed_analyzer(ts_seed, ts_target, **kwargs)
        
        #print ts_seed.shape, ts_target.shape
        
        self._measure = seed_analyzer.measure
        
        return self._measure
    
    def permutation_test(self, target_ds, n_permutation=100, axis=0):
        
        dimx = self.seed_ds.shape[0]
        dimy = target_ds.shape[0]
        
        self.n_permutation = n_permutation
        
        fp = np.memmap('/media/robbis/DATA/perm.dat',
                       dtype='float32', 
                       mode='w+',
                       shape=(self.n_permutation, 
                              dimx,
                              dimy)
                       )
        
        for i in range(self.n_permutation):
            
            progress(i, self.n_permutation)
            
            target_p = self._permute(target_ds, axis)
            fp[i,:] = self.run(target_p)
            
        progress(i, self.n_permutation)
        self.null_dist = fp
    
        return fp
            
    def _permute(self, ds, axis):
                
        ind = range(ds.shape[axis])
        
        ind = permutation(ind)
        
        if axis == 0:
            ds_ = ds[ind]
        elif axis == 1:
            ds_ = ds[:,ind]
        elif axis == 2:
            ds_ = ds[:,:,ind]
        
        return ds_
    
    def p_values(self, true_values, tails=0):
        """tails = [0, two-tailed; 1, upper; -1, lower]"""
        
        #check stuff
        if tails == 0:
            count_ = np.abs(self.null_dist) > np.abs(true_values)
        else:
            count_ = (tails * self.null_dist) > (tails * true_values)
            
        p_values = np.sum(count_, axis=0) / np.float(self.n_permutation)
        
        self._pvalues = p_values
        
        return p_values
    
    def save_results(self, file_path_pattern):
        
        """file_path_pattern could be a dir or a pattern
        if it is a dir then /path/to/dir/file1.txt is saved
        if it is a pattern then /path/to/pattern-file1.txt is saved"""
        
        fname = file_path_pattern+'p_values.txt'
        np.savetxt(fname, self._pvalues)
        
        fname = file_path_pattern+'p_values.png'
        pl.figure()
        
        pl.imshow(self._pvalues, interpolation='nearest')
        pl.colorbar()
        pl.savefig(fname)
        
        fname = file_path_pattern+'values.txt'
        np.savetxt(fname, self._measure)
        
        fname = file_path_pattern+'values.png'
        pl.figure()
        
        pl.imshow(self._measure, interpolation='nearest', vmax=1, vmin=-1)
        pl.colorbar()
        pl.savefig(fname)



class SimilarityAnalyzer(BaseAnalyzer):
    
    def __init__(self, 
                 time_serie=None, 
                 measure=euclidean, 
                 **kwargs):
        
        """measure is a function which takes two arrays
        and gives a number as output"""
        
        self._measure = measure
        self.time_serie = time_serie
        BaseAnalyzer.__init__(self, time_serie)
        
    @desc.setattr_on_read
    def measure(self):
        
        vars_ = self.time_serie.data.shape[0]
        result = np.zeros((vars_, vars_))
        
        for i in range(vars_):
            ts_seed = TimeSeries(self.time_serie.data[i], sampling_interval=1.)
            ts_target = TimeSeries(self.time_serie.data[i+1:], sampling_interval=1.)
            S = SeedAnalyzer(ts_seed, ts_target, self._measure)
            result[i,i+1:] = S.measure
            
        return result
        
        
