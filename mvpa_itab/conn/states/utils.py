import matplotlib.pyplot as pl
import numpy as np
from mvpa_itab.conn.utils import copy_matrix, array_to_matrix
from mvpa_itab.conn.states.states import get_centroids


def plot_states_matrices(X, 
                         labels, 
                         node_number=[6,5,8,10,4,5,7], 
                         node_networks = ['DAN','VAN','SMN','VIS','AUD','LAN','DMN']):


    position = [sum(node_number[:i+1]) for i in range(len(node_number))]
    
    centroids = get_centroids(X, labels)
    
    total_nodes = len(np.unique(labels))
    
    position_label = [-0.5+position[i]-node_number[i]/2. for i in range(len(node_number))]
    
    for i in np.unique(labels):
        pl.figure()
        matrix_ = copy_matrix(array_to_matrix(centroids[i]))
        total_nodes = matrix_.shape[0]
        pl.imshow(matrix_, interpolation='nearest')
        for name, n_nodes in zip(node_networks, position):
            pl.vlines(n_nodes-0.5, -0.5, total_nodes-0.5)
            pl.hlines(n_nodes-0.5, -0.5, total_nodes-0.5)
        
        pl.title('State '+str(i+1))
        pl.xticks(position_label, node_networks)
        pl.yticks(position_label, node_networks)
        
    #Save!
    
   
    
def plot_metrics(metrics_, metric_names, k_step):
    
    fig = pl.figure()
    n_rows = np.ceil(len(metric_names)/2.)
    for i, m in enumerate(metrics_.T):
        ax = fig.add_subplot(int(n_rows), 2, i+1)
        ax.plot(k_step, m, '-o')
        ax.set_title(metric_names[i])
        
    pl.show()
    
    

def get_extrema_histogram(arg_extrema, n_timepoints):
    
    hist_arg = np.zeros(n_timepoints)
    n_subjects = len(np.unique(arg_extrema[0]))
    
    for i in range(n_subjects):
        sub_max_arg = arg_extrema[1][arg_extrema[0] == i]
        hist_arg[sub_max_arg] += 1
        
    return hist_arg