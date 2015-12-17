import numpy as np
import scipy as sp
import minepy

def mutual_information(signal1, signal2, bins=40):
    
    pdf_, _, _ = np.histogram2d(signal1, signal2, bins=bins, normed=True)
    
    pdf_ /= np.float(pdf_.sum())
    
    Hxy = joint_entropy(pdf_)
    Hx = entropy(pdf_.sum(axis=0))
    Hy = entropy(pdf_.sum(axis=1))
    
    return ( Hx + Hy - Hxy)
            
        
def entropy(pdf_):
        
    entropy_ = 0.
    for i in np.arange(pdf_.shape[0]):
        if pdf_[i] != 0:
            entropy_ += pdf_[i] * np.log2(pdf_[i])
        else:
            entropy_ += 0
 
    return -1 * entropy_


def joint_entropy(pdf_):
    
    entropy_ = 0
    
    for i in range(pdf_.shape[0]):        
        entropy_+= entropy(pdf_[i])
    
    return -1 * entropy_


def mic(signal1, signal2):
    
    mine = minepy.MINE()
    mine.compute_score(signal1, signal2)
    
    return mine.mic()
    
def correlation(y_true, y_pred):
    return sp.stats.stats.pearsonr(y_true, y_pred)[0]


def ranking_correlation(corr):
    """From a correlation array returns the index order of absolute correlation
    magnitude.
    """
    return  np.argsort(np.abs(corr))[::-1]