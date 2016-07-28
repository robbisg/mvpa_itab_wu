import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as pl
from scipy.spatial.distance import pdist, squareform
from itertools import permutations, combinations

ds_ = np.loadtxt('/media/robbis/DATA/fmri/loris/loris_new.csv', delimiter=',', dtype=np.str_)

column_legend = {'Paziente':0,
                 'colonna_controlli':1,
                'patologia':2,
                'sede ant':3,
                'sede post':4,
                'numero voxel':5,
                'volume lesione':6,
                'distanza in sigma':7,
                'broca x':8,
                'broca y':9,
                'broca z':10,
                'beta':11,
                'broca_TALA':12,
                'distanza_radq_broca_TALA':13,
                'broca_contr':14,
                'RADQ_broca_contr':15,
                'tpj x':16,
                'tpj y':17,
                'tpj z':18,
                'beta':19,
                'tpj_tala':20,
                'radq_tpj_tala':21,
                'd_tpj_controlli':22,
                'radq_tpj_con':23,
                'sts x':24,
                'sts y':25,
                'sts z':26,
                'beta':27,
                'sts_tala':28,
                'radq_sts':29,
                'sts_controlli':30,
                'radq_sts_cont':31,
                'dimensioni':32,               
                 }

subj_excluded = np.logical_and(ds_[1:,0] != 'CONBIA', ds_[1:,0] != 'FABBET')
ds_ = ds_[subj_excluded]

index_list = ['broca x','broca y','broca z','tpj x','tpj y','tpj z','sts x','sts y','sts z']

#index_list = ['broca x','broca y','broca z']
index_list = ['tpj x','tpj y','tpj z']
#index_list = ['sts x','sts y','sts z']

x_indexes = [column_legend[i] for i in index_list]

areas = [['broca x','broca y','broca z'],['tpj x','tpj y','tpj z'],['sts x','sts y','sts z']]


for index_list in areas:
    #for g1, g2 in list(combinations(range(4), 2)):
        x_indexes = [column_legend[i] for i in index_list]
        print '····························'
        X = ds_[1:, x_indexes]
        y = ds_[1:, column_legend['patologia']]
        y_patient = ds_[1:, column_legend['colonna_controlli']]
        y_ant = ds_[1:, column_legend['sede ant']]
        y_ant[y_ant==''] = '2'
        y_pos = ds_[1:, column_legend['sede post']]
        y_pos[y_pos==''] = '2'
        
        y_dim = ds_[1:, column_legend['dimensioni']]
        y_dim[y_dim==''] = '5'
        #g = sns.PairGrid(df, vars=index_list)
    
        X = np.int_(X)
        y = np.int_(y)
        #############################################################################################
        # Filter data
        #############################################################################################
        
        # Select anterior (#0) and posterior (#1)
        mask_ant = np.logical_and(y_ant != '2', y_ant == y_pos)
        mask_anterior = y_pos == '0' # It keeps controls that are coded as (#0)
        mask_posterior = y_ant == '1'
        
        mask_pat = y == 0 # 1: Cavernomi, 0: Gliomi
        mask_group = y!= 3 # Use Cavernomi, gliomi and patients
        
        # analysis_mask = np.logical_and(mask_ant, mask_pat)
        # analysis_mask = np.logical_and(mask_anterior, mask_group) # Anteriori gl, cav and pat
        
        # Uncomment if you want pair analysis
        #analysis_mask = np.logical_or(y == g1, y == g2)
        analysis_mask = np.logical_or(y == 2, y == 1)
        
        # Uncomment if you want to exclude a group
        #analysis_mask = y == 2
        
        # Uncomment if analysis is all vs all
        #analysis_mask = np.ones_like(y, dtype=np.bool)
        
        # Uncomment if you want exclude posterior or anterior samples
        analysis_mask = np.logical_and(analysis_mask, mask_posterior)
        
        # Big vs small
        mask_big = y_dim == '1'
        mask_small = y_dim == '0'
        
        mask_size = np.logical_or(y == 2, mask_big)
        
        # Uncomment if you want to exclude big or small lesions
        #analysis_mask = np.logical_and(analysis_mask, mask_size)
        
        
        #######################################################################################
        ################# ··············    Analysis ···························· #############
        #######################################################################################
        X = X[analysis_mask,:]
        #y = np.int_(y_ant[analysis_mask])
        y = y[analysis_mask]
        #y = np.int_(y_patient[analysis_mask])
        #y = np.int_(y_ant)[analysis_mask]
        pair_plot(X, y, 'label', index_list)
        # Build distance matrix pairwise
        dist_ = squareform(pdist(X, 'euclidean'))
        #pl.imshow(dist_[np.argsort(y),:][:,np.argsort(y)], interpolation='nearest')
        
        # Within cluster distance
        
        
        dispersion_ = []
        
        for k in np.unique(y):
            disp_ = cluster_dispersion(dist_, y, k)
            dispersion_.append(disp_)
        
        
        total_dispersion = cluster_dispersion(dist_, np.zeros_like(y), 0)
        relative_dispersion = np.array(dispersion_)/total_dispersion
        
        from numpy.random.mtrand import permutation
        
        permutation_ = np.zeros((np.unique(y).shape[0], 2000))
        
        for i in range(2000):
            y_perm = permutation(y)
            
            dispersion_p = []
            
            for j, k in enumerate(np.unique(y_perm)):
                disp_p = cluster_dispersion(dist_, y_perm, k)
                permutation_[j, i] = disp_p
        
        print index_list[0][:-2]
        for i,k in enumerate(np.unique(y)):
            print str(k)+': dispersion = '+str(relative_dispersion[i])+ \
                                               ' p = '+str(np.count_nonzero(permutation_[i]<dispersion_[i])/2000.)+ \
                                               ' n: '+str(np.count_nonzero(y==k))



def cluster_dispersion(distance, labels, cluster_number):
    
    cluster_ = labels == cluster_number
    cluster_distance = distance[cluster_,:][:,cluster_]
    upper_index = np.triu_indices(cluster_distance.shape[0], k=1)
    cluster_distance = cluster_distance[upper_index]
    dispersion_ = cluster_distance.sum()/np.count_nonzero(cluster_)
    
    return dispersion_




def pair_plot(X, y, label, index_labels):
    data_frame = dict(zip(index_labels, np.int_(X.T)))
    data_frame['zpat'] = np.int_(y)   
    
    df = pd.DataFrame(data_frame)
    
    g = sns.PairGrid(df, vars=index_labels, hue='zpat')
    g = g.map_diag(sns.kdeplot, lw=3, legend=False)
    g = g.map_upper(pl.scatter)
    g = g.map_lower(pl.scatter)
    g = g.add_legend()


def tal2mni(coords):
    
    matrix_rot  = [[1,  0,          0,      0],
                   [0,  0.9988,     0.05,   0],
                   [0,  -0.05,      0.9988, 0],
                   [0,  0,          0,      1. ]]
    
    matrix_up   = [[0.99,   0,      0,      0],
                   [0,      0.97,   0,      0],
                   [0,      0,      0.92,   0],
                   [0,      0,      0,      1. ]]
    
    matrix_down = [[0.99,   0,      0,      0],
                   [0,      0.97,   0,      0],
                   [0,      0,      0.84,   0],
                   [0,      0,      0,      1.]]

    
    mask_z = coords[:,2] < 0 # Z points < 0
    
    mni_coords = np.zeros_like(coords)
    
    up_matrix = np.dot(matrix_rot, matrix_up)[:-1,:-1]
    down_matrix = np.dot(matrix_rot, matrix_down)[:-1,:-1]
    
    mni_coords[mask_z, :] = np.dot(coords[mask_z,:], down_matrix)
    mni_coords[np.logical_not(mask_z), :] = np.dot(coords[np.logical_not(mask_z),:], up_matrix)
    
    return mni_coords

areas = [['broca x','broca y','broca z'],['tpj x','tpj y','tpj z'],['sts x','sts y','sts z']]

for area in areas:
    x_indexes = [column_legend[i] for i in area]
    print '····························'
    X = np.int_(ds_[1:, x_indexes])


    for c in ['x','y','z']:
        fname = "%s_%s_coordinates_black.png" % (i[:i.find(' ')], c)
        plot_connectome(np.zeros((X.shape[0], X.shape[0])), 
                        tal2mni(X)*[-1,1,1], 
                        node_color=colors.tolist(),
                        output_file=os.path.join('/media/robbis/DATA/fmri/loris/',fname),
                        figure=pl.figure(figsize=(15,15), facecolor='k'),
                        node_size=200, 
                        node_kwargs={'alpha':0.9},
                        black_bg=True,
                        display_mode=c)