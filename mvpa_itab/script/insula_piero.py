import os
import numpy as np
from sklearn.decomposition.pca import PCA
from sklearn.metrics import r2_score
from numpy.random.mtrand import permutation
path = '/home/robbis/Share/dati_insulaDrive/'
gm_vbm = np.genfromtxt('/home/robbis/Share/dati_insulaDrive/GM_INSULA_vbm_Y_E_M_sbj.txt')
lista_file_conn = os.listdir('/home/robbis/Share/dati_insulaDrive/')
lista_file_conn = [f for f in lista_file_conn if f.find('Corr') != -1]
f_ = open('/home/robbis/Share/dati_insulaDrive/ROINAMES.txt', 'r')
line_ = f_.readline()
node_names = line_[:-1].split(',')
groups = gm_vbm[:,0]

conn_data = []
for f in lista_file_conn:
    data_ = np.genfromtxt(os.path.join(path, f))
    conn_data.append(data_)
    
conn_data = np.array(conn_data)

conn_data = conn_data[:,:13,:13]

mask = np.ones((13,13))
mask[np.tril_indices(13)] = 0
mask = np.bool_(mask)

conn_data = conn_data[:,mask]

group_mask = np.ones_like(groups, dtype=np.bool)
#group_mask = groups == 2

X = conn_data[group_mask]
y = gm_vbm[group_mask, 1]

pca = PCA(n_components=10)
#X_ = pca.fit_transform(X)
X_ = X
#skf = StratifiedShuffleSplit(groups[group_mask], n_iter=10, test_size=0.25)
skf = KFold(len(y), n_folds=len(y))
svr_rbf = SVR(kernel='rbf', C=1e3)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)




y_ = y
dist_r2 = []
for i in range(500):
    y_permuted = permutation(y)
    y_ = y_permuted
    
    y_pred = np.zeros_like(y)
    count_ = np.zeros_like(y)
    for train_index, test_index in skf:
        
        #pl.figure()
        X_train = X_[train_index]
        y_train = y_[train_index]
        
        X_test = X_[test_index]
        
        #y_reg = svr_rbf.fit(X_train, y_train).predict(X_test)
        #y_reg = svr_lin.fit(X_train, y_train).predict(X_test)
        y_reg = svr_poly.fit(X_train, y_train).predict(X_test)
        #pl.scatter(y[test_index], y_rbf)
    
        y_pred[test_index] += y_reg
        count_[test_index] += 1

    dist_r2.append(r2_score(y_, y_pred/count_))

pl.hist(np.array(dist_r2))

for i in range(10):
    pl.figure()
    pl.scatter(X_[:,i], y,c=groups, cmap=pl.cm.rainbow)