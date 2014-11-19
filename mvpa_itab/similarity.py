import numpy as np
import scipy
from mvpa2.suite import FeatureSelectionClassifier, Measure, TransferMeasure
from scipy.spatial.distance import mahalanobis, euclidean
from scipy.stats import pearsonr
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet, \
                        GraphLasso, ShrunkCovariance
                        
def similarity_measure(ds_tar, ds_src, results, p_value=0.05, method='mahalanobis'):
    
    if method == 'mahalanobis':
        return similarity_measure_mahalanobis (ds_tar, ds_src, results, p_value)
    elif method == 'euclidean':
        return similarity_measure_euclidean (ds_tar, ds_src, results, p_value)
    elif method == 'correlation':
        return similarity_measure_correlation (ds_tar, ds_src, results, p_value)


def similarity_measure_euclidean (ds_tar, ds_src, results, p_value):
    
    print 'Computing Euclidean similarity...'
    #classifier = results['classifier']
    
    #Get classifier from results
    classifier = results['fclf']

    #Make prediction on training set, to understand data distribution
    prediction_src = classifier.predict(ds_src)
    #prediction_src = results['predictions_ds']
    true_predictions = np.array(prediction_src) == ds_src.targets
    example_dist = dict()
    
    #Extract feature selected from each dataset
    if isinstance(classifier, FeatureSelectionClassifier):
        f_selection = results['fclf'].mapper
        ds_tar = f_selection(ds_tar)
        ds_src = f_selection(ds_src)
    
    #Get no. of features
    df = ds_src.samples.shape[1]
    print 'dof '+ str(df)

    #Set a t distribution
    t_stu = scipy.stats.t(df)
    
    #Set the p-value and the threshold value to validate predictions
    m_value = t_stu.isf(p_value * 0.5)
    
    
    '''
    Get class distribution information: mean and covariance
    '''
    
    for label in np.unique(ds_src.targets):
        
        #Get examples correctly classified
        mask = ds_src.targets == label
        example_dist[label] = dict()
        true_ex = ds_src.samples[mask * true_predictions]
        
        #Get Mean and Covariance to draw the distribution
        mean_ = np.mean(true_ex, axis=0)
        example_dist[label]['mean'] = mean_
        std_ = np.std(true_ex, axis=0)
        example_dist[label]['std'] = std_.mean()
        example_dist[label]['threshold'] = euclidean(mean_, 
                                                     mean_ + m_value * std_.mean())
        
    #Get target predictions (unlabelled)
    prediction_target = results['predictions']    
    
    #Test of data prediction
    euclidean_values = []
    normalized_values = []
    
    true_predictions = []
    
    for l, ex in zip(prediction_target, ds_tar.samples):
        
        #Keep mahalanobis distance between examples and class distribution
        dist = euclidean(example_dist[l]['mean'], ex)
        
        euclidean_values.append(dist)
        std_ = example_dist[l]['std']
        
        true_predictions.append(dist < example_dist[l]['threshold'])
        
        normalized_values.append(dist/np.sqrt(np.sum(np.power(std_,2))))
        
        
    distances = dict()
    


    '''
    threshold = euclidean(example_dist['trained']['mean'], 
                          example_dist['untrained']['mean'])
    '''
    for c in np.unique(prediction_target):
            
            mean_ = example_dist[c]['mean']
            std_ = example_dist[c]['std']
            
            distances[c] = []
            
            for ex in ds_tar.samples:
                distances[c].append(euclidean(mean_, ex))
            
            distances[c] = np.array(distances[c])
            threshold = euclidean(mean_, mean_ + m_value * std_.mean())
            
            values = np.array(distances[c])
            print values.shape
            
            
            #true_predictions += (values < threshold)
            
            
    '''
    Squared Mahalanobis distance is similar to a chi square distribution with 
    degrees of freedom equal to the number of features.
    '''
    
    #mahalanobis_values = np.array(mahalanobis_values) ** 2
    

    
    print 'threshold '+  str(m_value)
    #print p_value
    
    #Mask true predictions

    p_values = 1 - 2 * t_stu.cdf(np.array(normalized_values))
    #print np.count_nonzero(p_values)
    '''
    Get some data
    '''
    full_data = np.array(zip(ds_tar.targets, prediction_target, euclidean_values, p_values))
    #print np.count_nonzero(np.float_(full_data.T[3]) == p_values)
    
    #true_data = full_data[true_predictions]

    return full_data, np.bool_(true_predictions), threshold, p_values, distances    
    
    
def similarity_measure_mahalanobis (ds_tar, ds_src, results, p_value=0.95):
    
    print 'Computing Mahalanobis similarity...'
    #classifier = results['classifier']
    
    #Get classifier from results
    classifier = results['fclf']

    #Make prediction on training set, to understand data distribution
    prediction_src = classifier.predict(ds_src)
    #prediction_src = results['predictions_ds']
    true_predictions = np.array(prediction_src) == ds_src.targets
    example_dist = dict()
    
    #Extract feature selected from each dataset
    if isinstance(classifier, FeatureSelectionClassifier):
        f_selection = results['fclf'].mapper
        ds_tar = f_selection(ds_tar)
        ds_src = f_selection(ds_src)
    
    
    '''
    Get class distribution information: mean and covariance
    '''
    
    for label in np.unique(ds_src.targets):
        
        #Get examples correctly classified
        mask = ds_src.targets == label
        example_dist[label] = dict()
        true_ex = ds_src.samples[mask * true_predictions]
        
        #Get Mean and Covariance to draw the distribution
        mean_ = np.mean(true_ex, axis=0)
        example_dist[label]['mean'] = mean_
        '''
        cov_ = np.cov(true_ex.T)
        example_dist[label]['cov'] = cov_
        '''
        print 'Estimation of covariance matrix for '+label+' class...'
        print true_ex.shape
        try:
            #print 'Method is MinCovDet...'
            #print true_ex[:np.int(true_ex.shape[0]/3),:].shape
            #cov_ = MinCovDet().fit(true_ex)
            cov_ = LedoitWolf().fit(true_ex)
            #cov_ = EmpiricalCovariance().fit(true_ex)
            #cov_ = GraphLasso(alpha=0.5).fit(true_ex)
            #cov_ = OAS(alpha=0.1).fit(true_ex)
        except MemoryError, err:
            print 'Method is LedoitWolf'
            cov_ = LedoitWolf(block_size = 15000).fit(true_ex)
            
            
        #example_dist[label]['i_cov'] = scipy.linalg.inv(cov_)
        example_dist[label]['i_cov'] = cov_.precision_
        print 'Inverted covariance estimated...'
        
    #Get target predictions (unlabelled)
    prediction_target = results['predictions']
    
    
    #Test of data prediction
    mahalanobis_values = []
    for l, ex in zip(prediction_target, ds_tar.samples):
        
        #Keep mahalanobis distance between examples and class distribution
        #mahalanobis_values.append(mahalanobis(example_dist[l]['mean'], ex, example_dist[l]['i_cov']))
        mahalanobis_values.append(mahalanobis(example_dist[l]['mean'], ex, example_dist[l]['i_cov']))
    
    distances = dict()
    for c in np.unique(prediction_target):
            distances[c] = []
            for ex in ds_tar.samples:
                distances[c].append(mahalanobis(example_dist[c]['mean'], ex, example_dist[c]['i_cov']))
            
            distances[c] = np.array(distances[c]) ** 2
    '''
    Squared Mahalanobis distance is similar to a chi square distribution with 
    degrees of freedom equal to the number of features.
    '''
    
    mahalanobis_values = np.array(mahalanobis_values) ** 2
    
    #Get no. of features
    df = ds_src.samples.shape[1]
    
    print df

    #Set a chi squared distribution
    c_squared = scipy.stats.chi2(df)
    
    #Set the p-value and the threshold value to validate predictions
    m_value = c_squared.isf(p_value)
    threshold = m_value
    
    print m_value
    print p_value
    
    #Mask true predictions
    true_predictions = (mahalanobis_values < m_value)
    p_values = 1 - c_squared.cdf(mahalanobis_values)
    print np.count_nonzero(p_values)
    '''
    Get some data
    '''
    full_data = np.array(zip(ds_tar.targets, prediction_target, mahalanobis_values, p_values))
    #print np.count_nonzero(np.float_(full_data.T[3]) == p_values)
    
    #true_data = full_data[true_predictions]

    return full_data, true_predictions, threshold, p_values, distances
    
def similarity_measure_correlation (ds_tar, ds_src, results, p_value):


    print 'Computing Mahalanobis similarity...'
    #classifier = results['classifier']
    
    #Get classifier from results
    classifier = results['fclf']

    #Make prediction on training set, to understand data distribution
    prediction_src = classifier.predict(ds_src)
    #prediction_src = results['predictions_ds']
    true_predictions = np.array(prediction_src) == ds_src.targets
    example_dist = dict()
    
    #Extract feature selected from each dataset
    if isinstance(classifier, FeatureSelectionClassifier):
        f_selection = results['fclf'].mapper
        ds_tar = f_selection(ds_tar)
        ds_src = f_selection(ds_src)
    
    
    '''
    Get class distribution information: mean and covariance
    '''
    
    for label in np.unique(ds_src.targets):
        
        #Get examples correctly classified
        mask = ds_src.targets == label
        example_dist[label] = dict()
        true_ex = ds_src.samples[mask * true_predictions]
        
        #Get Mean and Covariance to draw the distribution
        mean_ = np.mean(true_ex, axis=0)
        example_dist[label]['mean'] = mean_
        '''
        cov_ = np.cov(true_ex.T)
        example_dist[label]['cov'] = cov_
        '''
        print 'Estimation of covariance matrix for '+label+' class...'
        print true_ex.shape
        try:
            print 'Method is Correlation...'
            #print true_ex[:np.int(true_ex.shape[0]/3),:].shape
            #cov_ = MinCovDet().fit(true_ex)
            #cov_ = LedoitWolf().fit(true_ex)
            #cov_ = EmpiricalCovariance().fit(true_ex)
            #cov_ = GraphLasso(alpha=0.5).fit(true_ex)
            #cov_ = OAS(alpha=0.1).fit(true_ex)
        except MemoryError, err:
            print 'Method is LedoitWolf'
            cov_ = LedoitWolf(block_size = 15000).fit(true_ex)
            
            
        #example_dist[label]['i_cov'] = scipy.linalg.inv(cov_)
        #example_dist[label]['i_cov'] = cov_.precision_
        print 'Inverted covariance estimated...'
        
    #Get target predictions (unlabelled)
    prediction_target = results['predictions']
    
    
    #Test of data prediction
    correlation_distances = []
    p_correlation = []
    for l, ex in zip(prediction_target, ds_tar.samples):
        
        #Keep mahalanobis distance between examples and class distribution
        dist, p = pearsonr(example_dist[l]['mean'], ex)
        correlation_distances.append(1-dist)
        p_correlation.append(p)
    
    distances = dict()
    for c in np.unique(prediction_target):
            distances[c] = []
            for ex in ds_tar.samples:
                distances[c].append(1-pearsonr(example_dist[l]['mean'], ex)[0])
            
            distances[c] = np.array(distances[c])
    '''
    Squared Mahalanobis distance is similar to a chi square distribution with 
    degrees of freedom equal to the number of features.
    '''    
    #Get no. of features
    df = ds_src.samples.shape[1]
    
    print df

    #Set a chi squared distribution
    c_squared = scipy.stats.chi2(df)
    
    #Set the p-value and the threshold value to validate predictions
    m_value = c_squared.isf(p_value)
    threshold = m_value
    
    print m_value
    print p_value
    
    p_correlation = np.array(p_correlation)
    correlation_distances = np.array(correlation_distances)
    #Mask true predictions
    true_predictions = (p_correlation < p_value)
    #p_values = 1 - c_squared.cdf(mahalanobis_values)
    #print np.count_nonzero(p_values)
    '''
    Get some data
    '''
    full_data = np.array(zip(ds_tar.targets, prediction_target, 
                             correlation_distances, p_correlation))
    #print np.count_nonzero(np.float_(full_data.T[3]) == p_values)
    
    #true_data = full_data[true_predictions]

    return full_data, true_predictions, threshold, p_correlation, distances       
##################################################################################
def similarity_confidence(ds_src, ds_tar, results):
    
    classifier = results['classifier']
    
    sensana = classifier.get_sensitivity_analyzer()
    weights = sensana(ds_src)
    
        
    prediction_src = classifier.predict(ds_src)
    true_predictions = prediction_src == ds_src.targets
    
    example_dist = dict()
    new_examples = dict()
    
    ##############################################################
    def calculate_examples(mean, sigma, weights, c = 2):
        from scipy.linalg import norm
        
        mean_p = mean + c * (weights/norm(weights)) * norm(sigma)
        mean_m = mean - c * (weights/norm(weights)) * norm(sigma)
        
        return np.array([mean_p, mean_m])
    ##############################################################
    
    il = 0
    values_est = dict()
    for label in np.unique(ds_src.targets):
        
        mask = ds_src.targets == label
        example_dist[label] = dict()
        new_examples[label] = []
        true_ex = ds_src.samples[mask * true_predictions]
        
        #Calculate examples average
        mean_ = np.mean(true_ex, axis=0)
        example_dist[label]['mean'] = mean_
        
        #Calculate examples standard deviation
        var_ = np.var(true_ex, axis=0)
        example_dist[label]['std'] = var_
        
        cov_ = np.cov(true_ex.T)
        example_dist[label]['cov'] = cov_
        
        mask_weights = np.array([s[0] == label or s[1] == label for s in weights.targets])    
        labels = np.array([s for s in weights.targets[mask_weights]])
        
        labels = labels[labels != label]
        weights_ = weights.samples[mask_weights]
        
        i = 0
        for l in labels:
            example_dist[label][l] = weights_[i]
            vec = calculate_examples(mean_, var_, weights_[i], 2)
            new_examples[label].append(vec)
            i = i + 1
            
        new_examples[label] = np.vstack(new_examples[label])
        
        predictions = classifier.predict(new_examples[label])  
        
        print predictions
        
        #predictions.reverse()
        
        values_est[label] = []
        for el in classifier.ca.estimates:
            k_list = []
            #p = predictions.pop()
            for k in el.keys():
                if k[0] == il:
                    if el[k] >= 1:
                        k_list.append(el[k])
                    else:
                        k_list.append(1)
                        
            values_est[label].append(k_list)
            
        il = il + 1
        
    
    predictions_tar = classifier.predict(ds_tar)

class SimilarityMeasure(Measure):
    
    def __init__(self):
        Measure.__init__(self)
        
    def _precall(self, ds):
        pass
        #common preprocess
        #Calculate mean?


class MahalanobisMeasure(Measure):
    
    #print 'Hello Mahalanobis'
    
    def __init__(self, p=0.05):
        
        Measure.__init__(self)
        self.p = p
        
    def train(self, ds):
        
        pass
        
    
    def _call(self, ds):
        
        #Partioner should be used instead
        ds_train = ds[ds.sa.task == 'task']
        ds_test = ds[ds.sa.task == 'rest']
        
        mean_ = ds_train.samples.mean(0)
        
        icov_ = EmpiricalCovariance().fit(ds_train.samples).precision_
        
        distances = []
        for ex in ds_test:
            
            dist_ = mahalanobis(mean_, ex, icov_)
            distances.append(dist_)
        
        chi_sq = scipy.stats.distributions.chi2(mean_.shape[0])
        m_value = chi_sq.isf(self.p)
        
        distances = np.array(distances)
        value = np.count_nonzero((distances ** 2) < m_value)
        
        return value
        
        
        