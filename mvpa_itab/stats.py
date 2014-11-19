from mvpa2.suite import CrossValidation
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.clfs.svm import LinearCSVMC
import numpy as np
from collections import Counter
from hgext.mq import fold

def permutations(ds, n_permutations=500):
    '''
    Procedure to do permutation tests
    
    
    TODO:
    
    - It is basically a shit procedure, 
    classifiers, cross-validation and some other things
    are fixed.
    
    '''
    
    
    clf = LinearCSVMC(C=1, probability=1,
                              enable_ca=['probabilities', 'estimates'])
    k=2
    ds_movie = ds[ds.targets != 'rest']
    ds_rest = ds[ds.targets == 'rest']
    shuffled_labels = np.random.permutation(ds_movie.targets)
    distributions = []
    
    for i in range(n_permutations):
        
        if i == 500:
            print '33%'
        elif i == 1000:
            print '66%'
        
        shuffled_labels = randomize_labels(ds_movie)
        
        index = 0
        while (shuffled_labels == ds_movie.targets).all():
            shuffled_labels = randomize_labels(ds_movie)
    
        if i == 0:
            shuffled_labels = ds_movie.targets
     
        acc = cross_validate(ds_movie, clf, NFoldPartitioner(cvtype=k), shuffled_labels)
        
        pred = clf.predict(ds_rest)
    
        perc = np.count_nonzero(np.array(pred) == 'movie')/np.float(len(pred))
        
        if np.mean(acc) > 0.5:
            index = 1
        
        distributions.append([acc, perc, index])
    
    
    return np.array(distributions)


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
        
