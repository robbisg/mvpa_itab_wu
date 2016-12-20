#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################
# pylint: disable=maybe-no-member, method-hidden, no-member
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg.linalg import LinAlgError
from itertools import cycle
import logging

def plot_transfer_graph_prob_fitted(path, name, analysis_folder):
    
    from sklearn.linear_model import Ridge
    from scipy.interpolate import UnivariateSpline
    result_file = open(
                       os.path.join(path, 
                               '0_results', 
                               analysis_folder, 
                               name, 
                               name+'_classifier.pyobj')
                       , 'r')
    
    results = pickle.load(result_file)
    
    runs = 12
    
    probabilities = results.probabilities
    prob = np.array([p[1][p[0]] for p in probabilities])
    pred = np.array([p[0] for p in probabilities])
    lab = np.unique(results.predictions)
    run_length = len(prob)/runs
       
    ridge = Ridge()
    
    f = plt.figure(figsize=(11,8))
    f2 = plt.figure(figsize=(11,8))
    data_sm = dict()
    data_or = dict()
    for c in np.unique(lab):
        data_sm[c] = []
        data_or[c] = []
    for i in range(12):
        if i < 6:
            aggregate = 1
            l = '_pre'
        else:
            aggregate = 2
            l = '_post'
        avg = []
        for c in np.unique(pred):
            a = f.add_subplot(3,2,(c*2)+aggregate)
            a2 = f2.add_subplot(3,2,(c*2)+aggregate)
            a.set_title(lab[c]+l)
            #v = prob[i*run_length:(i+1)*run_length]
            v = prob[i*run_length:(i+1)*run_length] * (pred[i*run_length:(i+1)*run_length] == c)
            v[len(v)-1] = 0
            yy = v.copy()

            xx = np.linspace(0, len(v), len(v))
            s = UnivariateSpline(xx, yy, s=5)
            ys = s(xx)
            try:
                ridge.fit(np.vander(xx, 7), yy)
                y_fit = ridge.predict(np.vander(xx, 7))
            except LinAlgError,err:
                ridge.fit(np.vander(xx, 9), yy)
                y_fit = ridge.predict(np.vander(xx, 9))
            
            data_sm[lab[c]].append(ys)
            data_or[lab[c]].append(v)

            a.plot(y_fit)
            a2.plot(ys)

            a.set_ybound(upper=1.1, lower=-0.1)
            a2.set_ybound(upper=1.1, lower=-0.1)
            
    f.legend(a.get_lines()[:],range(runs), loc=0)
    fname = os.path.join(path,'0_results', 
                               analysis_folder, 
                               name, 
                               name+'_values_fitted_ov.png')
    f.savefig(fname)
    f2.savefig(fname[:-3]+'smooth.png')
    plt.close('all')
    return data_sm, data_or
    


def plot_transfer_graph_fitted(path, name, analysis_folder):
    
    from sklearn.linear_model import Ridge
    from scipy.interpolate import UnivariateSpline
    
    result_file = open(
                       os.path.join(path, 
                               '0_results', 
                               analysis_folder, 
                               name, 
                               name+'_classifier.pyobj')
                       , 'r')
    
    results = pickle.load(result_file)
    
    runs = 6
    
    values = results.estimates
    run_length = len(values)/runs
    
    ridge = Ridge()
    
    f = plt.figure()
    a = f.add_subplot(111)
    for i in range(runs):
        
        v = values[i*run_length:(i+1)*run_length]
        yy = v.copy()
        
        xx = np.linspace(0, len(v), len(v))
        
        try:
            ridge.fit(np.vander(xx, 12), yy)
            y_fit = ridge.predict(np.vander(xx, 12))
        except LinAlgError,err:
            ridge.fit(np.vander(xx, 9), yy)
            y_fit = ridge.predict(np.vander(xx, 9))
        
        a.plot(y_fit)
    
    a.legend(range(runs))
    fname = os.path.join(path,'0_results', 
                               analysis_folder, 
                               name, 
                               name+'_values_fitted.png')
    f.savefig(fname)
    plt.close('all')


def plot_transfer_graph(path, name, results):
    
    r_targets = np.array(results['targets'])
    r_prediction = np.array(results['classifier'].ca.predictions)
    if str(results['classifier'].find('SVM') != -1):
        r_probabilities = np.array(results['classifier'].ca.probabilities)
    r_values = np.array(results['classifier'].ca.estimates)      

    p_probabilities = np.array([p[1][p[0]] for p in r_probabilities])
      
    report = ''
        
    for target in np.unique(r_targets):
            
        mask_target = r_targets == target
        prediction = r_prediction[mask_target]
        probabilities = p_probabilities[mask_target]
        values = r_values[mask_target]
            
        f_pred = plt.figure()
        a_prob = f_pred.add_subplot(111)
        #a_pred = f_pred.add_subplot(212)
            
        report = report +'--------- '+target+' -------------\n'
            
        for label in np.unique(prediction):
            mask_label = prediction == label
            n_pred = np.count_nonzero(mask_label)
            n_vols = len(prediction)
                
            perc = float(n_pred)/float(n_vols)
            report = report+'percentage of volumes labelled as '+label+' : '+str(perc)+' \n'
            
            mean = np.mean(probabilities[mask_label])
            plt_mean = np.linspace(mean, mean, num=len(probabilities))
            
            report = report+'mean probability: '+str(mean)+'\n'
            
            a_prob.plot(probabilities * mask_label)
            color = a_prob.get_lines()[-1:][0].get_color()
            a_prob.plot(plt_mean, color+'--', linewidth=2)
            a_prob.set_ylim(bottom=0.55)
            #a_pred.plot(values * mask_label)
        a_prob.legend(np.unique(prediction))     
        #a_prob.legend(np.unique(prediction))
        #a_pred.legend(np.unique(prediction))

        for axes in f_pred.get_axes():
            max = np.max(axes.get_lines()[0].get_data()[1])
            for i in range(3):
                axes.fill_between(np.arange(2*i*125, (2*i*125)+125), 
                                  -max, max,facecolor='yellow', alpha=0.2)
        
        
        fname = os.path.join(path, name+'_'+target+'_probabilities_values.png')
        f_pred.savefig(fname)
    
    
    rep_txt = name+'_stats_probab.txt'   
    
    rep = open(os.path.join(path, rep_txt), 'w')
    rep.write(report)
    rep.close()  
    plt.close('all')


def plot_clusters_graph(path, name, results):
    
    
    color = dict({'trained':'r', 
              'fixation':'b', 
              'RestPre':'y', 
              'RestPost':'g',
              'untrained': 'k'})
    
    
    markers = dict({'trained':'p', 
              'fixation':'s', 
              'RestPre':'^', 
              'RestPost':'o',
              'untrained': '*'})
    
    colors = cycle('bgrcmykbgrmykbgrcmykbgrcmyk')
    
    clusters = results['clusters']
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    print targets
    report = ''
    
    for condition in clusters:
        print condition
        mask_target = targets == str(condition)
        m_predictions = predictions[mask_target]
        
        #####################################
        f_dist = plt.figure()
        a_dist = f_dist.add_subplot(111)
        a_dist.imshow(clusters[condition]['dist'])
        f_dist.colorbar(a_dist.get_images()[0])
        fname = os.path.join(path, name+'_'+condition+'_dist.png')
        f_dist.savefig(fname)
        #####################################
        
        cluster_labels = clusters[condition]['clusters']
        pos = clusters[condition]['pos']
        
        f_cluster = plt.figure()
        a_cluster = f_cluster.add_subplot(111)
        
        f_predict = plt.figure()
        a_predict = f_predict.add_subplot(111)
        
        report = report + '---------- '+condition+' -----------\n'
        
        for cluster in np.unique(cluster_labels):
            
            cl_mask = cluster_labels == cluster
            c_pos = pos[cl_mask]
            col = colors.next()
            a_cluster.scatter(c_pos.T[0], c_pos.T[1], color = col)
            
            report = report+'\nCluster n. '+str(cluster)+'\n'
            
            for label in np.unique(m_predictions):
                p_mask = m_predictions[cl_mask] == label
                labelled = np.count_nonzero(p_mask)
                
                a_predict.scatter(c_pos.T[0][p_mask], c_pos.T[1][p_mask], color = col, 
                         marker = markers[label])
                
                perc = np.float(labelled)/np.count_nonzero(cl_mask)
                report = report + label+' : '+str(perc*100)+'\n'
        
        
        a_cluster.legend(np.unique(cluster_labels))
        fname = os.path.join(path, name+'_'+condition+'_mds_clusters.png')
        f_cluster.savefig(fname)
        
        a_predict.legend(np.unique(predictions))
        fname = os.path.join(path, name+'_'+condition+'_mds_predictions.png')
        f_predict.savefig(fname)
    
    
    rep_txt = name+'_stats.txt'
    
    rep = open(os.path.join(path, rep_txt), 'w')
    rep.write(report)
    rep.close()
    plt.close('all')
    
def plot_cv_results(cv, err, title):
    # make new figure
    plt.figure()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # null distribution samples
    dist_samples = np.asarray(cv.null_dist.ca.dist_samples)
    for i in range(len(err)):
        c = colors.next()
        # histogram of all computed errors from permuted data per CV-fold
        plt.hist(np.ravel(dist_samples[i]), bins=20, color=c,
                label='CV-fold %i' %i, alpha=0.5,
                range=(dist_samples.min(), dist_samples.max()))
        # empirical error
        plt.axvline(np.asscalar(err[i]), color=c)

    # chance-level for a binary classification with balanced samples
    plt.axvline(0.5, color='black', ls='--')
    # scale x-axis to full range of possible error values
    plt.xlim(0,1)
    plt.xlabel(title)


def plot_scatter_2d(ds_merged, method='mds', fig_number = 1):
    
    from sklearn import decomposition, manifold, lda, ensemble
    """
    methods: 'mds', 'pca', 'iso', 'forest', 'embedding'
    """
    
    data = ds_merged.samples
    
    stringa=''
    if method == 'pca':
        clf = decomposition.RandomizedPCA(n_components=2)
        stringa = 'Principal Component Analysis'
    ########    
    elif method == 'iso':
        clf = manifold.Isomap(30, n_components=2)
        stringa = 'Iso surfaces '
    #########    
    elif method == 'forest':
        hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
        data = hasher.fit_transform(data)
        clf = decomposition.RandomizedPCA(n_components=2)
        stringa = 'Random Forests'
    ########
    elif method == 'embedding':
        clf = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
        stringa = 'Spectral Embedding'
    #########
    else:
        clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
        stringa = 'Multidimensional scaling'
        
        
    ###########################
    #dist_matrix = squareform(pdist(data, 'euclidean'))

    print stringa+' is performing...'

    pos = clf.fit_transform(data)

    colors = cycle('bgrymkybgrcmybgrcmybgrcmy')
    
    f = plt.figure()
    a = f.add_subplot(111)
    a.set_title(stringa)
    for label in np.unique(ds_merged.targets):
        m = ds_merged.targets == label
        data_m = pos[m]
        c = colors.next()
        a.scatter(data_m.T[0].mean(), data_m.T[1].mean(), label=label, color=c, s=120)
        a.scatter(data_m.T[0][::2], data_m.T[1][::2], color=c)
        '''
        cov_ = np.cov(data_m.T)
        v, w = np.linalg.eigh(cov_)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 0.5
        ell = mpl.patches.Ellipse(np.mean(data_m, axis=0), v[0], v[1],
                                  180 + angle, color=c)
        ell.set_clip_box(a.bbox)
        ell.set_alpha(0.2)
        a.add_artist(ell)
        '''
    a.legend()
    return
