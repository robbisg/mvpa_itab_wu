from scipy.io import loadmat, savemat
from scipy.signal import argrelextrema
from mvpa_itab.conn.utils import copy_matrix, array_to_matrix
from scipy.spatial.distance import squareform, pdist, euclidean
from scipy.signal._peak_finding import argrelmin
from mvpa_itab.conn.states.states import get_min_speed_arguments,\
    calculate_metrics
from mvpa_itab.conn.states.utils import get_extema_histogram


distance_fname = os.path.join('/media/robbis/DATA/fmri/movie_viviana/',
                              'pairwise_distance_all_subj_all_pts_movie.npy')
clustering_ = pickle.load(file('/media/robbis/DATA/fmri/movie_viviana/clustering_labels_2to30k_movie_all_sub_all_pts.obj', 'r'))
distance_matrix = np.load(distance_fname, mmap_mode='r')



data = loadmat('/media/robbis/DATA/fmri/movie_viviana/mat_corr_sub_REST.mat')
data = np.array(data['data'], dtype=np.float16)
data_ = data[:,:,np.triu_indices(data.shape[-1], k=1)[0], \
                np.triu_indices(data.shape[-1], k=1)[1]]

## Variance ##

arg_maxima, stdev_data = get_max_variance_arguments(data_)
hist_arg = get_extrema_histogram(arg_maxima, data_.shape[1])

### Speed ###

subj_min_speed, subj_speed = get_min_speed_arguments(data_)
hist_arg = get_extrema_histogram(subj_min_speed, data_.shape[1])
    

#### No. of clusters #####
X = data_[arg_maxima]
X = data_[subj_min_speed]

clustering_ = []

k_steps = range(2,31)

for k in k_steps:
    print '----- '+str(k)+' -------'
    km = KMeans(n_clusters=k).fit(X)
    labels = km.labels_
    clustering_.append(labels)
    

metrics_, k_step, metrics.keys() = calculate_metrics(X, clustering_)


    
    

    

     



        
    
    
    
    
