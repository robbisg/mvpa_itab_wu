import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.distance import squareform, pdist
import logging

logger = logging.getLogger(__name__)

def less_equal(x1, x2):
    return np.logical_and(x1 <= x2, x1 != 0)


def greater_equal(x1, x2):
    return np.logical_and(x1 >= x2, x1 != 0)
    



def peak_mapper(peak_type):
    
    mapper = {'min': less_equal,
              'max': greater_equal              
              }

    return mapper[peak_type]


def get_subsampler(subsampler_type):
    
    # TODO: use subsampler type to set parameters using string!
    
    mapper = {'variance': VarianceSubsampler(),
              'speed' : SpeedSubsampler(),
              'variance+mean':VarianceSubsampler(prefilter=MeanThresholder()),
              'speed+mean':SpeedSubsampler(prefilter=MeanThresholder())              
              }
    
    return mapper[subsampler_type]





class Subsampler(object):
    
    def __init__(self, peak, order):
        self.peak = peak_mapper(peak)
        self.order = order

    
    def fit(self, data):
        return NotImplemented()
    
    
    def subsample(self, data):
        
        #Check if it has been fitted
                
        arg = argrelextrema(np.array(self.measure), 
                            self.peak,
                            axis=1,
                            order=self.order)
        
        self.arg = arg
        
        X = data[arg]
        
        return X



class NoneThresholder(object):
    
    def fit(self, measure):
        return measure 



class SpeedSubsampler(Subsampler):
    
    def __init__(self, peak='min', order=5, distance='euclidean', prefilter=NoneThresholder()):
        self.peak = peak_mapper(peak)
        self.order = order
        self.distance = distance
        self.prefilter = prefilter
    
    
    def fit(self, data):
        """
        From the data it extract the points with low local velocity 
        and returns the arguments of these points and the 
        speed for each point.    
        """
    
        subj_speed = []
        for i in range(data.shape[0]):
            distance_ = squareform(pdist(data[i], self.distance))
            
            speed_ = [distance_[i, i+1] for i in range(distance_.shape[0]-1)]
            subj_speed.append(np.array(speed_))
        
        subj_speed = np.vstack(subj_speed)
        
        self.measure = self.prefilter.fit(subj_speed)
                
        return self
    
    

class VarianceSubsampler(Subsampler):
    
    def __init__(self, peak='max', order=5, prefilter=NoneThresholder()):
        self.peak = peak_mapper(peak)
        self.order = order
        self.prefilter = prefilter
        
    
    def fit(self, data):
        
        stdev_data = data.std(axis=2)
        self.measure = self.prefilter.fit(stdev_data)
        
        return self
    


class MeanThresholder(object):

    def fit(self, measure):
        mean_measure = np.sum(measure, axis=1)/np.sum(measure != 0, axis=1)
        threshold_measure = measure * (measure > mean_measure[...,None])                                        
        
        return threshold_measure
        
            
        