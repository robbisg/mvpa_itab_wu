from main_wu import *
from io import *
from mvpa2.clfs.transerror import ConfusionMatrix
import os
    
    
def test_spatiotemporal(path, subjects, conf_file, type, **kwargs):
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
        if arg == 'balance':
            balance = kwargs[arg]
    
    total_results = dict()
    data_path = conf['data_path']
    for subj in subjects:
        try:
            ds = load_dataset(data_path, subj, type, **conf)
        except Exception, err:
            print err
            continue
        ds = preprocess_dataset(ds, type, **conf)
        
        if 'balance' in locals() and balance == True:
            if conf['label_included'] == 'all' and \
                conf['label_dropped'] == 'none':
                ds = balance_dataset(ds, 'fixation')
        
        r = spatiotemporal(ds, **conf)
        
        total_results[subj] = r
    
    conf['analysis_type'] = 'spatiotemporal'
    conf['analysis_task'] = type
    conf['classes'] = np.unique(ds.targets)
    
    save_results(path, total_results, conf)
    
    return total_results

def test_spatial(path, subjects, conf_file, type, **kwargs):
    
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    total_results = dict()
    
    data_path = conf['data_path']
    
    for subj in subjects:
        print '------'
        try:
            ds = load_dataset(data_path, subj, type, **conf)
        except Exception, err:
            print err
            continue
        
        ds = preprocess_dataset(ds, type, **conf)
        
        r = spatial(ds, **conf)
        
        total_results[subj] = r
        
        
    
    conf['analysis_type'] = 'spatial'
    conf['analysis_task'] = type
    conf['classes'] = np.unique(ds.targets)  
    #save_results()
    save_results(path, total_results, conf)
    
    return total_results


def test_clustering(path, subjects, analysis, conf_file, source='task', **kwargs):    
    
    if source == 'task':
        target = 'rest'
    else:
        target = 'task'
     
    conf_src = read_configuration(path, conf_file, source)
    conf_tar = read_configuration(path, conf_file, target)
    
    ##############################################
    conf_src['label_included'] = 'all'
    conf_src['label_dropped'] = 'none'
    conf_src['mean_samples'] = 'True'
    ##############################################
    for arg in kwargs:
        conf_src[arg] = kwargs[arg]
        conf_tar[arg] = kwargs[arg]
        
    total_results = dict()
    
    data_path = conf_src['data_path']
    
    for subj in subjects:
        try:
            ds_src = load_dataset(data_path, subj, source, **conf_src)
            ds_tar = load_dataset(data_path, subj, target, **conf_tar)
        except Exception, err:
            print err
            continue
        
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 
        
        if conf_src['label_included'] == 'all' and \
                conf_src['label_dropped'] != 'fixation':
                ds_src = balance_dataset(ds_src, 'fixation')
        
        r = clustering_analysis(ds_src, ds_tar, analysis, **kwargs)
        
        total_results[subj] = r
        
    conf_src['analysis_type'] = 'clustering'
    conf_src['analysis_task'] = 'task'
    conf_src['analysis_func'] = analysis.func_name
    conf_src['classes'] = np.unique(ds_src.targets)
    
    save_results(path, total_results, conf_src)

    return total_results


def test_transfer_learning(path, subjects, analysis,  conf_file, source='task', **kwargs):
    
    if source == 'task':
        target = 'rest'
    else:
        if source == 'rest':
            target = 'task'
    
    
    if source == 'saccade':
        target = 'face'
    else:
        if source == 'face':
            target = 'saccade'
    
    
    conf_src = read_configuration(path, conf_file, source)
    conf_tar = read_configuration(path, conf_file, target)
    
    ##############################################    
    ##############################################
    ##    conf_src['label_included'] = 'all'    ##   
    ##    conf_src['label_dropped'] = 'none'    ##
    ##    conf_src['mean_samples'] = 'False'    ##
    ##############################################
    ##############################################
    
    for arg in kwargs:
        conf_src[arg] = kwargs[arg]
        conf_tar[arg] = kwargs[arg]
    
    for arg in conf_src:
        if arg == 'map_list':
            map_list = conf_src[arg].split(',')
    
    
    data_path = conf_src['data_path']
    total_results = dict()
    
    for subj in subjects:
        print '--------'
        try:
            ds_src = load_dataset(data_path, subj, source, **conf_src)
            ds_tar = load_dataset(data_path, subj, target, **conf_tar)
        except Exception, err:
            print err
            continue
        
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 

        if conf_src['label_included'] == 'all' and \
           conf_src['label_dropped'] != 'fixation':
                print 'Balancing dataset...'
                ds_src = balance_dataset(ds_src, 'fixation')        
        
        r = transfer_learning(ds_src, ds_tar, analysis, **conf_src)
        
        
        
        pred = np.array(r['classifier'].ca.predictions)
        targets = r['targets']
        
        c_m = ConfusionMatrix(predictions=pred, targets=targets)
        c_m.compute()
        r['confusion_target'] = c_m
        print c_m
        
        tr_pred = similarity_measure_mahalanobis(r['ds_tar'], r['ds_src'], r)
        r['mahalanobis_similarity'] = tr_pred
        
        #print tr_pred
        
        c_mat_mahala = ConfusionMatrix(predictions=tr_pred.T[1], targets=tr_pred.T[0])
        c_mat_mahala.compute()
        r['confusion_mahala'] = c_mat_mahala
        
        d_prime, beta, c, c_new = signal_detection_measures(pred, targets, map_list)
        r['d_prime'] = d_prime
        r['beta'] = beta
        r['c'] = c
        r['confusion_total'] = c_new
        
        '''
        d_prime_maha, c_new_maha = d_prime_statistics(tr_pred.T[1], tr_pred.T[0], map_list)
        r['d_prime_maha'] = d_prime_maha
        r['confusion_tot_maha'] = c_new_maha
        '''
        
        total_results[subj] = r
        
    conf_src['analysis_type'] = 'transfer_learning'
    conf_src['analysis_task'] = source
    conf_src['analysis_func'] = analysis.func_name
    conf_src['classes'] = np.unique(ds_src.targets)
    
    
    save_results(path, total_results, conf_src)
    
    return [total_results, r['ds_src'], r['ds_tar']]


def test_searchlight(path, subjects, conf_file, type, **kwargs):
    
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    total_results = dict()
    data_path = conf['data_path']
    for subj in subjects:
        
        ds = load_dataset(data_path, subj, type, **conf)
        ds = preprocess_dataset(ds, type, **conf)
        
        r = searchlight(ds, **kwargs)
        
        total_results[subj] = r
    
    conf['analysis_type'] = 'searchlight'
    conf['analysis_task'] = type
    conf['classes'] = np.unique(ds.targets)  
    #save_results()
    save_results(path, total_results, conf)
    
    return total_results


def test_group_mvpa(path, subjects, analysis,  conf_file, source='task', **kwargs):
    
    conf = read_configuration(path, conf_file, type)
    
    for arg in kwargs:
        conf[arg] = kwargs[arg]
    
    total_results = dict()
    data_path = conf['data_path']
    for subj in subjects:
        
        ds = load_dataset(data_path, subj, type, **conf)
        ds = preprocess_dataset(ds, type, **conf)
        
        
        r = searchlight(ds, **kwargs)
        
        total_results[subj] = r       
    


#####################################################################

def signal_detection_measures(predictions, targets, map_list):
    
    '''
    map_list =  two-element list, is what we expect the classifier did first element
                first element is the prediction
                second element is the label of second ds to be predicted
    '''
    
    mask_predictions = np.array(predictions) == map_list[0]
    mask_targets = np.array(targets) == map_list[1]
    
    new_predictions = np.array(predictions, dtype='|S20').copy()
    new_targets = np.array(targets, dtype='|S20').copy()
    
    new_label_1 = map_list[0]+'-'+map_list[1]
    new_label_2 = np.unique(new_predictions[~mask_predictions])[0]+'-'+ \
                    np.unique(new_targets[~mask_targets])[0]
    
    new_predictions[mask_predictions] = new_label_1
    new_targets[mask_targets] = new_label_1
    
    new_predictions[~mask_predictions] = new_label_2
    new_targets[~mask_targets] = new_label_2
    
    
    c_matrix = ConfusionMatrix(predictions=new_predictions, targets=new_targets)
    c_matrix.compute()
    
    from scipy.stats import norm
    
    hit_rate = c_matrix.stats['TP']/np.float_(c_matrix.stats['P'])
    false_rate = c_matrix.stats['FP']/np.float_(c_matrix.stats['N'])
    
    d_prime = norm.ppf(hit_rate) - norm.ppf(false_rate)
    
    beta = - d_prime[0] * 0.5 * (norm.ppf(hit_rate) + norm.ppf(false_rate)) 
    
    c = - 0.5 * (norm.ppf(hit_rate) + norm.ppf(false_rate))
    
       
    return d_prime[0], beta[0], c[0], c_matrix


def similarity_measure_mahalanobis (ds_tar, ds_src, results, p_value=0.01):
    
    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import LedoitWolf
    
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
        
        cov_ = LedoitWolf(block_size = 10000).fit(true_ex).covariance_
        example_dist[label]['i_cov'] = scipy.linalg.inv(cov_)
        #print cov_.shape
        #print true_ex.shape
        #print cov_
        #print example_dist[label]['i_cov']
        
    #Get target predictions (unlabelled)
    prediction_target = results['predictions']
    
    
    #Test of data prediction
    mahalanobis_values = []
    for l, ex in zip(prediction_target, ds_tar.samples):
        #Keep mahalanobis distance between examples and class distribution
        mahalanobis_values.append(mahalanobis(example_dist[l]['mean'], ex, example_dist[l]['i_cov']))
    
    '''
    Squared Mahalanobis distance is similar to a chi square distribution with 
    degrees of freedom equal to the number of features.
    '''
    
    
    mahalanobis_values = np.array(mahalanobis_values) ** 2
    #Get no. of features
    df = ds_tar.samples.shape[1]
    #print df
    #Set a chi squared distribution
    c_squared = scipy.stats.chi2(df)
    #Set the p-value and the threshold value to validate predictions
    m_value = c_squared.isf(p_value)
    #print m_value
    #Mask true predictions
    true_predictions = (mahalanobis_values < m_value)
    
    '''
    Get some data
    '''
    full_data = np.array(zip(ds_tar.targets, prediction_target, mahalanobis_values))
    
    true_data = full_data[true_predictions]
    
    #print full_data
    return true_data
    
    
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
    
    
def get_merged_ds(path, subjects, analysis,  conf_file, source='task', **kwargs):
    
    if source == 'task':
        target = 'rest'
    else:
        if source == 'rest':
            target = 'task'
    
    
    if source == 'saccade':
        target = 'face'
    else:
        if source == 'face':
            target = 'saccade'
    
    
    conf_src = read_configuration(path, conf_file, source)
    conf_tar = read_configuration(path, conf_file, target)
    
    ##############################################    
    ##############################################
    ##    conf_src['label_included'] = 'all'    ##   
    ##    conf_src['label_dropped'] = 'none'    ##
    ##    conf_src['mean_samples'] = 'False'    ##
    ##############################################
    ##############################################
    
    for arg in kwargs:
        conf_src[arg] = kwargs[arg]
        conf_tar[arg] = kwargs[arg]
    
    data_path = conf_src['data_path']
    
    for subj in subjects:
        print '--------'
        try:
            ds_src = load_dataset(data_path, subj, source, **conf_src)
            ds_tar = load_dataset(data_path, subj, target, **conf_tar)
        except Exception, err:
            print err
            continue
        
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar) 
        
        ds_merged = vstack((ds_src, ds_tar))
        
        '''
        methods = ['iso', 'pca', 'forest', 'embedding', 'mds']
        
        for m, i in zip(methods, range(len(methods))):
            plot_scatter_2d(ds_merged, method=m, fig_number=i+1)
        '''
        r = spatial(ds_merged, **conf_src)
    
    return r
           