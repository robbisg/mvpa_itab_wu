import matplotlib.pyplot as pl
import numpy as np
import os
from mvpa_itab.conn.operations import copy_matrix, array_to_matrix
from mvpa_itab.conn.states.states import get_centroids
from sklearn.manifold.mds import MDS

def plot_states_matrices(X, 
                         labels,
                         save_path,
                         condition_label,
                         node_number=[6,5,8,10,4,5,7], 
                         node_networks = ['DAN','VAN','SMN','VIS','AUD','LAN','DMN'],
                         use_centroid=False,
                         ):


    position = [sum(node_number[:i+1]) for i in range(len(node_number))]
    
    if not use_centroid:
        centroids = get_centroids(X, labels)
        total_nodes = len(np.unique(labels))
    else:
        centroids = X.copy()
        total_nodes = X.shape[0]
    
    position_label = [-0.5+position[i]-node_number[i]/2. for i in range(len(node_number))]
    
    for i in np.arange(total_nodes):
        fig = pl.figure()
        matrix_ = copy_matrix(array_to_matrix(centroids[i]))
        total_nodes = matrix_.shape[0]
        pl.imshow(matrix_, interpolation='nearest', vmin=0, vmax=1)
        for name, n_nodes in zip(node_networks, position):
            pl.vlines(n_nodes-0.5, -0.5, total_nodes-0.5)
            pl.hlines(n_nodes-0.5, -0.5, total_nodes-0.5)
        
        pl.title('State '+str(i+1))
        pl.xticks(position_label, node_networks)
        pl.yticks(position_label, node_networks)
        
        pl.colorbar()
        
        fname = "%s_state_%s.png" % (condition_label, str(i+1))
        
        pl.savefig(os.path.join(save_path, fname))
    
   
    
def plot_metrics(metrics_, metric_names, k_step):
    
    fig = pl.figure(figsize=(12,10))
    n_rows = np.ceil(len(metric_names)/2.)
    for i, m in enumerate(metrics_.T):
        ax = fig.add_subplot(int(n_rows), 2, i+1)
        ax.plot(k_step, m, '-o')
        ax.set_title(metric_names[i])
        
    #pl.show()
    return fig
    


def plot_dynamics(state_dynamics, condition, path, **kwargs):
    
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
    

  
def get_positions(dict_centroids, path):
    
    for k, v in dict_centroids.iteritems():
        print k, v
    
    X_c = [v for k, v in dict_centroids.iteritems()]
    
    X_c = np.vstack(X_c)
    
    pos = MDS(n_components=2).fit_transform(X_c)

    color = {'movie': 'red',
             'rest':'green',
             'scramble':'blue'
             }
    
    
    fig, ax = pl.subplots()
    for i, c in enumerate(['movie', 'scramble', 'rest']):
        pos1 = i*5
        pos2 = (i+1)*5
        ax.scatter(pos[pos1:pos2, 0], pos[pos1:pos2, 1], c=color[c], s=60)
        for j, (x, y) in enumerate(pos[pos1:pos2]):
            ax.annotate(str(j+1), (x,y))

    fname = os.path.join(path, "mds.png")
    pl.savefig(fname)
        
    pl.close('all')
    return  