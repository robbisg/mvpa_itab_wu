import os
import numpy as np
from sklearn.decomposition.pca import PCA
from sklearn.metrics import r2_score
from numpy.random.mtrand import permutation
from sklearn.cross_validation import *
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics.metrics import mean_squared_error
from sklearn.linear_model.coordinate_descent import LassoCV, ElasticNetCV
from mvpa.clfs.enet import ENET
from mvpa_itab.stats import Correlation

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
group_mask = groups == 1

X = conn_data[group_mask]
y = gm_vbm[group_mask, 1]

pca = PCA(n_components=10)
#X_ = pca.fit_transform(X)
X_ = X
skf = StratifiedShuffleSplit(groups[group_mask], n_iter=35, test_size=0.25)
#skf = KFold(len(y), n_folds=10)
#svr_rbf = SVR(kernel='rbf', C=1e3)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_ = y
dist_r2 = []
dist_mse = []
for alpha in np.arange(0, 0.5, 0.01):
    #y_permuted = permutation(y)
    #y_ = y_permuted
    
    y_pred = np.zeros_like(y)
    count_ = np.zeros_like(y)
    lasso = ElasticNet(alpha=alpha, fit_intercept=True)
    
    for train_index, test_index in skf:
             
        #pl.figure()
        X_train = X_[train_index]
        y_train = y_[train_index]
        
        X_test = X_[test_index]
        
        #y_reg = svr_rbf.fit(X_train, y_train).predict(X_test)
        #y_reg = svr_lin.fit(X_train, y_train).predict(X_test)
        #y_reg = svr_poly.fit(X_train, y_train).predict(X_test)
        y_reg = lasso.fit(X_train, y_train).predict(X_test)
        
        #pl.scatter(y[test_index], y_rbf)
        #print np.count_nonzero(lasso.coef_)
        
        y_pred[test_index] += y_reg
        count_[test_index] += 1
    
    dist_mse.append(mean_squared_error(y_, y_pred/count_))
    dist_r2.append(r2_score(y_, y_pred/count_))

pl.plot(np.arange(0, 0.5, 0.01), np.array(dist_mse))

for i in range(10):
    pl.figure()
    pl.scatter(X_[:,i], y,c=groups, cmap=pl.cm.rainbow)

################################################
n_rows = 3   
indexes = np.array(zip(*np.triu_indices(13, 1)))
color = 'bgr'
labels = ['elderly', 'mci', 'young']
j = 0
for _, x in enumerate(X.T):
    if (j%n_rows) == 0:
        f = pl.figure()
    
    for i in range(n_rows):
        a = f.add_subplot(n_rows, n_rows,(n_rows)*(j%n_rows)+(i+1))
        title = node_names[indexes[j][0]]+' -- '+node_names[indexes[j][1]]
        pl.scatter(x[groups==i], y[groups==i], c=color[i], s=40, label=labels[i])
        a.set_title(title)
        pl.legend()
    
    j+=1
######################################################
enetcv = ElasticNetCV(alphas=np.linspace(1, 0.05, 50), 
                          cv=ShuffleSplit(len(y), n_iter=50, test_size=0.25))

lassocv = LassoCV(alphas=np.linspace(1, 0.05, 50), 
                          cv=ShuffleSplit(len(y), n_iter=50, test_size=0.25))
for i in range(n_rows):
    
    X_ = conn_data[groups==i,:]
    y_ = y[groups==i]
    
    enetcv = ElasticNetCV(alphas=np.linspace(1, 0.05, 50), 
                          cv=ShuffleSplit(len(y_), n_iter=50, test_size=0.25))

    lassocv = LassoCV(alphas=np.linspace(1, 0.05, 50), 
                          cv=ShuffleSplit(len(y_), n_iter=50, test_size=0.25))
    
    
    lassocv.fit(X_, y_)
    enetcv.fit(X_, y_)
    f = pl.figure()
    a = f.add_subplot(211)
    pl.plot(lassocv.coef_, c=color[i], label=labels[i])
    a = f.add_subplot(212)
    pl.plot(enetcv.coef_, c=color[i], label=labels[i])
    
##################################################

permut_ = []
for i in np.arange(1000):
    
    y_permuted = permutation(y)
    cv=ShuffleSplit(len(y), n_iter=50, test_size=0.25)
    
    mse_ = []
    
    svr_rbf = SVR(kernel='rbf', C=1)
    svr_lin = SVR(kernel='linear', C=1)
    svr_poly = SVR(kernel='poly', C=1, degree=2)
    
    for train_index, test_index in cv:
                 
            #pl.figure()
            X_train = X[train_index]
            y_train = y_permuted[train_index]
            
            X_test = X[test_index]
            
            y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
            y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
            y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
            
            mse_rbf = mean_squared_error(y_permuted[test_index], y_rbf)
            mse_lin = mean_squared_error(y_permuted[test_index], y_lin)
            mse_poly = mean_squared_error(y_permuted[test_index], y_poly)
            
            mse_.append([mse_rbf, mse_lin, mse_poly])
            
    permut_.append(mse_)
     
permut_ = np.array(permut_)

c = Correlation(X)
for i in range(3):
    corr = c.run(X[groups==i],y[groups==i])[0]
    pl.plot(corr, marker='o', c=color[i])

pl.xticks(np.arange(78), label_list, rotation=45)





def nested_cross_validation(X, y, test_size=0.15):
    """
        Basically it is used to evaluate testing error for small datasets
        it computes also the model on training dataset!
        
    """
    
    # First cross-validation to separate tes    
    random_cv = ShuffleSplit(len(y), n_iter=100, test_size=test_size)
    
    mse_ = []
    best_alpha = []
    validation_error = []
    features_selected = []
    
    for train, test in random_cv:
        
        X_tr = X[train]
        y_tr = y[train]
        
        lassocv = LassoCV(alphas=np.linspace(0.01, 0.001, 50), 
                          cv=ShuffleSplit(len(y_tr), n_iter=50, test_size=test_size))
        
        lassocv.fit(X_tr, y_tr)
        
        y_pred = lassocv.predict(X[test])
        err = mean_squared_error(y[test], y_pred)
        
        best_alpha.append(lassocv.alpha_)
        mse_.append(err)
        validation_error.append(lassocv.mse_path_)
        features_selected.append(lassocv.coef_)
        
    
    features_selected = np.array(features_selected)
    validation_error = np.array(validation_error)
    mse_ = np.array(mse_)
    best_alpha = np.array(best_alpha)
    
    # Plot results (to be coded in another function)
    
    # Feature selected plot
    
    f = pl.figure()
    a = f.add_subplot(211)
    a.imshow(features_selected, aspect='auto')
    # colorbar
    
    a2 = f.add_subplot(212)
    sum_ = np.array(features_selected != 0, dtype=np.int).sum(axis=0)
    a2.bar(np.arange(sum_.shape[0]), sum_)
    
    
        
        

        