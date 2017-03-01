import matplotlib.pyplot as pl
import numpy as np
import os
from mvpa_itab.conn.operations import copy_matrix, array_to_matrix
from mvpa_itab.conn.states.utils import get_centroids
from sklearn.manifold.mds import MDS

def plot_states_matrices(X, 
                         labels,
                         node_number=[6,5,8,10,4,5,7], 
                         node_networks = ['DAN','VAN','SMN','VIS','AUD','LAN','DMN'],
                         use_centroid=False,
                         save_fig=True,
                         save_path="/media/robbis/DATA/fmri/movie_viviana",
                         save_name_condition=None,
                         **kwargs
                         ):
    """
    Plots the centroids in square matrix form.
    It could be used with original data and labels but also 
    with the original centroids if you set use_centroids as True. 
    
    """

    position = [sum(node_number[:i+1]) for i in range(len(node_number))]
    
    if not use_centroid:
        centroids = get_centroids(X, labels)
        n_states = len(np.unique(labels))
    else:
        centroids = X.copy()
        n_states = X.shape[0]
    
    position_label = [-0.5+position[i]-node_number[i]/2. for i in range(len(node_number))]
    
    for i in np.arange(n_states):
        fig = pl.figure()
        
        matrix_ = copy_matrix(array_to_matrix(centroids[i]), diagonal_filler=0)
        n_states = matrix_.shape[0]
        pl.imshow(matrix_, interpolation='nearest', vmin=0)
        for _, n_nodes in zip(node_networks, position):
            pl.vlines(n_nodes-0.5, -0.5, n_states-0.5)
            pl.hlines(n_nodes-0.5, -0.5, n_states-0.5)
        
        pl.title('State '+str(i+1))
        pl.xticks(position_label, node_networks)
        pl.yticks(position_label, node_networks)
        
        pl.colorbar()
        
        if save_fig:
            fname = "%s_state_%s.png" % (str(save_name_condition), str(i+1))
            pl.savefig(os.path.join(save_path, fname))
        
        #return fig
        
   
    
def plot_metrics(metrics_, metric_names, k_step):
    """
    Plots the clustering metrics.
    """
    
    fig = pl.figure(figsize=(12,10))
    n_rows = np.ceil(len(metric_names)/2.)
    for i, m in enumerate(metrics_.T):
        ax = fig.add_subplot(int(n_rows), 2, i+1)
        ax.plot(k_step, m, '-o')
        ax.set_title(metric_names[i])
        
    #pl.show()
    return fig
    


def plot_dynamics(state_dynamics, condition, path, **kwargs):
    """
    Plot the dynamics of the states for each session.
    """
    
    fname_cfg = {'prefix':'',
                 'suffix':''}
    
    fname_cfg.update(kwargs)
    
    
    for i, ts in enumerate(state_dynamics):
        _ = pl.figure(figsize=(18,10))
        for j, sts in enumerate(ts):
            pl.plot(sts, label=str(j+1))
        pl.legend()
        
        pl.xlabel("Time")
        pl.ylabel("Dissimilarity")
        
        fname = "%scondition_%s_session_%02d_dynamics%s.png" % (fname_cfg['prefix'],
                                                               condition,
                                                               i+1,
                                                               fname_cfg['suffix'],                                                               
                                                               )
        fname = os.path.join(path, fname)
        pl.savefig(fname)
        
    pl.close('all')
    return 



def plot_frequencies(state_frequency, condition, path):
    """
    Plots the frequency of the state
    """
    
    for i, ts in enumerate(state_frequency):
        _ = pl.figure(figsize=(12,10))
        freq = ts[0]
        values = ts[1]
        for j, f in enumerate(values):
            pl.plot(freq[1:150], f[1:150], label=str(j+1))
        pl.legend()
        
        pl.xlabel("Frequency")
        pl.ylabel("Power")
        
        fname = os.path.join(path, "condition_%s_session_%02d_freq.png" % (condition, i+1))
        pl.savefig(fname)
        
    pl.close('all')
    return         
    

  
def plot_positions(dict_centroids, **kwargs):
    
    configuration = {
                     "conditions": ['movie', 'scramble', 'rest'],
                     "colors": ['red', 'blue', 'green'],
                     "save_fig":False,
                     "path":None,
                     }
    
    configuration.update(kwargs)
        
    X_c = [v for k, v in dict_centroids.iteritems()]
    
    X_c = np.vstack(X_c)
    
    pos = MDS(n_components=2).fit_transform(X_c)

    color = dict(zip(configuration['conditions'], configuration['colors']))
    
    fig = pl.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    for i, c in enumerate(configuration["conditions"]):
        pos1 = i*5
        pos2 = (i+1)*5
        ax.scatter(pos[pos1:pos2, 0], pos[pos1:pos2, 1], c=color[c], s=150)
        for j, (x, y) in enumerate(pos[pos1:pos2]):
            ax.annotate(str(j+1), (x,y), fontsize=15)
    
    if configuration["save_fig"]:
        fname = os.path.join(configuration['path'], "mds.png")
        pl.savefig(fname)
        
    return fig


