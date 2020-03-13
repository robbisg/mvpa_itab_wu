import scipy
from scipy.spatial.distance import pdist, squareform, euclidean
from sklearn import metrics
from pyitab.analysis.states.metrics import *
from sklearn.datasets.samples_generator import make_blobs

default_metrics = {'Silhouette': metrics.silhouette_score,
                    'Krzanowski-Lai': kl_criterion,
                    'Global Explained Variance':global_explained_variance,
                    'Within Group Sum of Squares': wgss,
                    'Explained Variance':explained_variance,
                    'Index I': index_i,
                    "Cross-validation":cross_validation_index
                    }
from tqdm import tqdm
def test_metrics():

    #metrics_ = dict()
    hit = dict()
    for name, _ in default_metrics.items():
       # metrics_[name] = []
        hit[name] = 0


    for i in tqdm(range(500)):

        n_clusters = np.random.randint(3, 10)
        centers = np.random.randint(-20, 20, (n_clusters, 3))

        k_step = range(2, 10)
        
        X, labels_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.3,
                                    random_state=0)

        #print(n_clusters)
    
        clustering_ = []
        for k in range(2,10):
            
            km = KMeans(n_clusters=k).fit(X)
            labels = km.labels_
            clustering_.append(labels)
            
        
        metrics_ = dict()
        #hit = dict()
        for name, _ in default_metrics.items():
            metrics_[name] = []
            #hit[name] = 0


        #fig, ax = pl.subplots(3, 3)
        #ax[2][2].scatter(X[:,0], X[:,1],c=labels_true)
        for i, label in enumerate(clustering_):
            #ax[int(i/3)][int(i%3)].scatter(X[:,0], X[:,1],c=label)
            

            for name, metric in default_metrics.items():
                if name == 'Krzanowski-Lai':
                    if i == 0 or i == len(clustering_) - 1:
                        prev_labels = None
                        next_labels = None
                    else:
                        prev_labels = clustering_[i-1]
                        next_labels = clustering_[i+1]
                    
                    m = metric(X, 
                            label, 
                            previous_labels=prev_labels, 
                            next_labels=next_labels,
                            precomputed=False)
                else:
                    m = metric(X, label)
                
                metrics_[name].append(m)

        #fig, ax = pl.subplots(3, 3)

        
        
        for i, (name, values) in enumerate(metrics_.items()):
            if name in ['Silhouette', 'Krzanowski-Lai', 'Index I']:
                guessed_cluster = np.nonzero(np.max(values) == values)[0][0] + k_step[0]
                #ax[int(i/3)][int(i%3)].plot(k_step, values, 'o-')
            else:
                data = np.vstack((k_step, values)).T
                #data = MinMaxScaler().fit_transform(data)
                theta = np.arctan2(values[-1] - values[0],
                                   k_step[-1] - k_step[0])
                co = np.cos(theta)
                si = np.sin(theta)
                rotation_matrix = np.array(((co, -si), (si, co)))
                # rotate data vector
                data = data.dot(rotation_matrix)
                fx = np.max
                if name != 'Global Explained Variance':
                    fx = np.min
                guessed_cluster =  np.nonzero(data[:, 1] == fx(data[:, 1]))[0][0] +  k_step[0]
                #ax[int(i/3)][int(i%3)].plot(k_step, data[:, 1], 'o-')

            #print(name, guessed_cluster)

            hit[name] += int(guessed_cluster == n_clusters)

            
            #ax[int(i/3)][int(i%3)].set_title(name)
    #pl.close('all')



        

        
    