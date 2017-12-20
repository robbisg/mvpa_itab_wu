import os
import random
import numpy as np
import matplotlib.pylab as pl
from sklearn.decomposition.pca import PCA
from sklearn.metrics import r2_score
from numpy.random.mtrand import permutation
from itertools import combinations
from sklearn.cross_validation import *
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics.metrics import mean_squared_error
from sklearn.linear_model.coordinate_descent import LassoCV, ElasticNetCV
from sklearn.svm import SVR
from mvpa_itab.stats import Correlation, CrossValidation

path = '/home/robbis/Share/dati_insulaDrive/'
lista_file_conn = os.listdir('/home/robbis/Share/dati_insulaDrive/')
lista_file_conn = [f for f in lista_file_conn if f.find('Corr') != -1]
lista_file_conn.sort()
conn_data = []
for f in lista_file_conn:
    data_ = np.genfromtxt(os.path.join(path, f))
    conn_data.append(data_)
conn_data = np.array(conn_data)
   
f_ = open('/home/robbis/Share/dati_insulaDrive/ROINAMES.txt', 'r')
line_ = f_.readline()
node_names = line_[:-1].split(',')

gm_vbm = np.genfromtxt('/home/robbis/Share/dati_insulaDrive/GM_INSULA_vbm_Y_E_M_sbj.txt')
groups = gm_vbm[:,0]


conn_data = conn_data[:,:13,:13]

mask = np.ones((13,13))
mask[np.tril_indices(13)] = 0
mask = np.bool_(mask)

conn_data = conn_data[:,mask]

group_mask = np.ones_like(groups, dtype=np.bool)
#group_mask = groups == 1

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
labels_group = ['elderly', 'mci', 'young']
j = 0
for _, x in enumerate(X.T):
    if (j%n_rows) == 0:
        f = pl.figure()
    
    for i in range(n_rows):
        a = f.add_subplot(n_rows, n_rows,(n_rows)*(j%n_rows)+(i+1))
        title = node_names[indexes[j][0]]+' -- '+node_names[indexes[j][1]]
        pl.scatter(x[groups==i], y[groups==i], c=color[i], s=40, label=labels_group[i])
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
    pl.plot(lassocv.coef_, c=color[i], label=labels_group[i])
    a = f.add_subplot(212)
    pl.plot(enetcv.coef_, c=color[i], label=labels_group[i])
    
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



################################################
label_list = []
indexes = np.array(zip(*np.triu_indices(13, 1)))
for i in range(X.shape[1]):
    conn_ = node_names[indexes[i][0]]+' -- '+node_names[indexes[i][1]]
    label_list.append(conn_)


#################################################

c = Correlation(X)

for i in [1]:
    corr = c.transform(X[groups==i], y[groups==i])[0]
    pl.plot(corr, marker='o', c=color[i], label=labels_group[i])

pl.xticks(np.arange(78), label_list, rotation=90)

###########################################

X1 = X[groups==1]
y1 = y[groups==1]


alpha = 1.
algorithms_ = [SVR(kernel='rbf', C=1), 
               #SVR(kernel='linear', C=1),
               #SVR(kernel='poly', C=1, degree=3),
               #SVR(kernel='rbf', C=10),
               #SVR(kernel='rbf', C=0.5),
               #ElasticNet(alpha=alpha, fit_intercept=True),
               #Lasso(alpha=alpha, fit_intercept=True)
               ]

labels_alg = ['SVR(kernel=\'rbf\', C=1)', 
               #'SVR(kernel=\'linear\', C=1)',
               #'SVR(kernel=\'poly\', C=1, degree=3)',
               #'SVR(kernel=\'rbf\', C=10, degree=2)',
               #'SVR(kernel=\'rbf\', C=0.5)',
               #'ElasticNet',
               #'Lasso',
               ]


repetitions = 500
#n_permutation = 1000
n_trial = 1
arg_ = np.argsort(np.abs(corr))[::-1]
mse_t = np.zeros((n_trial, arg_.shape[0], len(algorithms_), repetitions))

index_ = np.arange(len(y1))
for n in range(n_trial):
    
    for i in range(arg_.shape[0]):
        
        X_fit = X1[:,arg_[:i+1]] # Sequential slicing
        
        for j, alg_ in enumerate(algorithms_):
            cv = ShuffleSplit(len(y1), n_iter=repetitions, test_size=0.25)
    
            k = 0
            for train_index, test_index in cv:
                
                train_index_perm = train_index
                
                X_train = X_fit[train_index]
                y_train = y1[train_index_perm]
                
                
                X_test = X_fit[test_index]
                y_predict = alg_.fit(X_train, y_train).predict(X_test)
                
                mse = mean_squared_error(y1[test_index], y_predict)
                
                mse_t[n,i,j,k] = mse
            
                k+=1
                
    p1_ = []
    mse_avg = mse_.mean(1)
    
    mse_t1_avg = mse_t[n].squeeze().mean(1)
    for v, dist in zip(mse_t1_avg, mse_avg):
        m_ = np.count_nonzero(v >= dist)
        p1_.append(m_)
    
    p1_ = np.array(p1_)/2000.
    pl.scatter(np.arange(1,79), mse_t1_avg, c='b')
    pl.scatter(np.arange(1,79)[p1_<=0.05], mse_t1_avg[p1_<=0.05], c='b', s=45)

############## Plots #############################       
color = 'bgrykcm'    
for i in range(len(labels_alg)-2):
    pl.plot(np.arange(78), 
               mse_[:,i].mean(1), 
               #yerr=mse_[:,i].std(1)*0.5, 
               label=labels_alg[i], 
               marker='o',
               color=color[i])
    '''
    pl.hlines(mse_[:,i].mean(), 
              0, 
              78, 
              color=color[i], 
              linestyle='--', 
              linewidth=2.5)
    '''

for i in range(len(labels_alg)-2):
    pl.figure()
    pl.plot(np.arange(78), 
               mse_mean[:,i], 
               #yerr=mse_[:,i].std(1)*0.5, 
               label=labels_alg[i], 
               marker='o',
               color=color[i])
    
    v_ = [np.count_nonzero(mse_perm_cv_mean[s,i] > mse_mean[s,i]) for s in range(78)]
    p_ = np.array(v_)/1000.
    
    sign_ = p_ <= 0.01
    x_ = np.nonzero(sign_)[0]
    y_ = mse_mean[np.nonzero(sign_)[0], i]
    
    for j in range(len(x_)):
        pl.text(x_[j], y_[j], '*', fontsize=20)
    
    pl.legend()

############# Cross-Validation repetitions effect #####################

# Looking for changes of mse with respect to number of cv
repetitions = np.arange(10, 201, 10) # array([ 10,  20,  ..., 190, 200])

for i in range(arg_.shape[0]):
    
    X_fit = X1[:,arg_[:i+1]] # Sequential slicing
    
    for j, alg_ in enumerate(algorithms_):
        
        for r in repetitions:
            
            method = ShuffleSplit(len(y1), n_iter=r, test_size=0.25)
            
            cv = CrossValidation(method, alg_)
            
            mse_ = cv.transform(X_fit, y1)
        
########### Permute feature set #########################
     
repetitions = 200
n_combination = 2000
n_permutation = 1
arg_ = np.argsort(np.abs(corr))[::-1]
mse_ = np.zeros((n_combination, len(algorithms_), repetitions, n_permutation))
n_features = 6

index_ = np.arange(len(y1))
comb = -1
features = []
while (comb<(n_combination-1)):
    
    # Randomly choosing 6 features
    feature_set = permutation(np.arange(arg_.shape[0]))
    feature_set = feature_set[:n_features]
    if np.all(np.sort(feature_set) == np.sort(arg_[:n_features])):
        continue
    else:
        X_fit = X1[:,feature_set]
        comb+=1
        features.append(feature_set)
    
    for j, alg_ in enumerate(algorithms_):
        cv=ShuffleSplit(len(y1), n_iter=repetitions, test_size=0.25)
        #k = 0
        #for train_index, test_index in cv:
        for p in range(n_permutation):
            
            #index_perm = permutation(index_)
            #y_perm = y1[index_perm]
            
            #for p in range(n_permutation):
            k = 0
            for train_index, test_index in cv:
                
                X_train = X_fit[train_index]
                y_train = y1[train_index]
                
                
                X_test = X_fit[test_index]
                y_predict = alg_.fit(X_train, y_train).predict(X_test)
                
                mse = mean_squared_error(y1[test_index], y_predict)
                
                mse_[comb,j,k,p] = mse
            
                k+=1
            #k+=1
            
###################### Permutation of test set #########################

X1 = X[groups==1]
y1 = y[groups==1]

X1 = X_
y1 = y_

alpha = 1.
algorithms_ = [SVR(kernel='rbf', C=1), 
               SVR(kernel='linear', C=1),
               SVR(kernel='poly', C=1, degree=3),
               SVR(kernel='rbf', C=10),
               SVR(kernel='rbf', C=0.5),
               ElasticNet(alpha=alpha, fit_intercept=True),
               Lasso(alpha=alpha, fit_intercept=True)
               ]

labels_alg = ['SVR(kernel=\'rbf\', C=1)', 
               'SVR(kernel=\'linear\', C=1)',
               'SVR(kernel=\'poly\', C=1, degree=3)',
               'SVR(kernel=\'rbf\', C=10, degree=2)',
               'SVR(kernel=\'rbf\', C=0.5)',
               'ElasticNet',
               'Lasso',
               ]

repetitions = 200
n_permutation = 1
arg_ = np.argsort(np.abs(corr))[::-1]

mse_ = np.zeros((arg_.shape[0], len(algorithms_), repetitions, n_permutation))
r2_ = np.zeros((arg_.shape[0], len(algorithms_), repetitions, n_permutation))

index_ = np.arange(len(y1))

for i in range(arg_.shape[0]):
    
    X_fit = X1[:,arg_[:i+1]] # Sequential slicing
    
    for j, alg_ in enumerate(algorithms_):
        cv=ShuffleSplit(len(y1), n_iter=repetitions, test_size=0.25)
        k = 0
        for train_index, test_index in cv:
            
            for p in range(n_permutation):
                
                if n_permutation == 1:
                    train_index_perm = train_index
                else:
                    train_index_perm = permutation(train_index)
                
                X_train = X_fit[train_index]
                
                #y_train = y_perm[train_index_perm]
                y_train = y1[train_index_perm]
                
                
                X_test = X_fit[test_index]
                y_predict = alg_.fit(X_train, y_train).predict(X_test)
                
                mse = mean_squared_error(y1[test_index], y_predict)
                r2 = r2_score(y1[test_index], y_predict)
                
                mse_[i,j,k,p] = mse
                r2_[i,j,k,p] = r2
                #k+=1
            k+=1
            
            
########## Permutation of data either train and test ##################
X1 = X[groups==1]
y1 = y[groups==1]


alpha = 1.
algorithms_ = [SVR(kernel='rbf', C=1), 
               #SVR(kernel='linear', C=1),
               #SVR(kernel='poly', C=1, degree=3),
               #SVR(kernel='rbf', C=10),
               #SVR(kernel='rbf', C=0.5),
               #ElasticNet(alpha=alpha, fit_intercept=True),
               #Lasso(alpha=alpha, fit_intercept=True)
               ]

labels_alg = ['SVR(kernel=\'rbf\', C=1)', 
               #'SVR(kernel=\'linear\', C=1)',
               #'SVR(kernel=\'poly\', C=1, degree=3)',
               #'SVR(kernel=\'rbf\', C=10, degree=2)',
               #'SVR(kernel=\'rbf\', C=0.5)',
               #'ElasticNet',
               #'Lasso',
               ]


repetitions = 200
n_permutation = 5000
arg_ = np.argsort(np.abs(corr))[::-1]
mse_ = np.zeros((arg_.shape[0], len(algorithms_), repetitions, n_permutation))
corr_ = np.zeros((arg_.shape[0], len(algorithms_), repetitions, n_permutation))
index_ = np.arange(len(y1))
total_ = mse_.size
count_ = 0
for i in range(arg_.shape[0]):
    
    X_fit = X1[:,arg_[:i+1]] # Sequential slicing
    
    for j, alg_ in enumerate(algorithms_):
        cv=ShuffleSplit(len(y1), n_iter=repetitions, test_size=0.25)

        for p in range(n_permutation):
            
            index_perm = permutation(index_)
            y_perm = y1[index_perm]

            k = 0
            for train_index, test_index in cv:
                
                train_index_perm = train_index
                
                X_train = X_fit[train_index]
                
                y_train = y_perm[train_index_perm]

                X_test = X_fit[test_index]
                y_predict = alg_.fit(X_train, y_train).predict(X_test)
                
                mse = mean_squared_error(y1[test_index], y_predict)
                cor = np.corrcoef(y1[test_index], y_predict)
                
                mse_[i,j,k,p] = mse
                corr_[i,j,k,p] = cor[0,1]
                count_ += 1
                k+=1
                
                progress(count_, total_)


p1_ = []
mse_avg = mse_.mean(1)
mse_t1_avg = mse_t1.mean(1)
for v, dist in zip(mse_t1_avg, mse_avg):
    m_ = np.count_nonzero(v >= dist)
    p1_.append(m_)
    
p1_ = np.array(p1_)/2000.

p2_ = []
mse_avg = mse_.mean(1)
mse_t2_avg = mse_t2.mean(1)
for v, dist in zip(mse_t2_avg, mse_avg):
    m_ = np.count_nonzero(v >= dist)
    p2_.append(m_)
    
p2_ = np.array(p2_)/2000.

p3_ = []
mse_avg = mse_.mean(1)
mse_t3_avg = mse_t3.mean(1)
for v, dist in zip(mse_t3_avg, mse_avg):
    m_ = np.count_nonzero(v >= dist)
    p3_.append(m_)
    
p3_ = np.array(p3_)/2000.

pl.boxplot(mse_avg.T, showmeans=True, showfliers=False)
pl.scatter(np.arange(1,79), mse_t1_avg, c='b')
pl.scatter(np.arange(1,79)[p1_<=0.05], mse_t1_avg[p1_<=0.05], c='b', s=45)
pl.scatter(np.arange(1,79), mse_t2_avg, c='g')
pl.scatter(np.arange(1,79)[p2_<=0.05], mse_t2_avg[p2_<=0.05], c='g', s=45)
pl.scatter(np.arange(1,79), mse_t3_avg, c='r')
pl.scatter(np.arange(1,79)[p3_<=0.05], mse_t3_avg[p3_<=0.05], c='r', s=45)


############### Controls ###################
gm_can = np.genfromtxt('/home/robbis/Share/CAN_NET_GMperc.csv', skip_header=1, delimiter=',')
can_labels = ['MCC','R_aINS','L_pINS','L_AMY']
repetitions = 200
n_permutation = 2000
arg_ = np.argsort(np.abs(corr))[::-1]
mse_can = np.zeros((arg_.shape[0], len(algorithms_), repetitions, gm_can.shape[1]))

index_ = np.arange(len(y1))
for ii, y1 in enumerate(gm_can.T):
    
    for i in range(arg_.shape[0]):
    
        X_fit = X1[:,arg_[:i+1]] # Sequential slicing
        
        for j, alg_ in enumerate(algorithms_):
            
            cv=ShuffleSplit(len(y1), n_iter=repetitions, test_size=0.25)
    
            k = 0
            for train_index, test_index in cv:
                
                train_index_perm = train_index
                
                X_train = X_fit[train_index]
                
                y_train = y1[train_index_perm]

                X_test = X_fit[test_index]
                y_predict = alg_.fit(X_train, y_train).predict(X_test)
                
                mse = mean_squared_error(y1[test_index], y_predict)
                
                mse_can[i,j, k, ii] = mse
            
                k+=1

t_control = np.zeros((78, 4))
p_control = np.zeros((78, 4))
for j, x in enumerate(mse_control):
    for i in range(4):
        t_c, p_c = ttest_ind(x[:,i], x[:,4])
        t_control[j,i] = t_c
        p_control[j,i] = p_c

colors_ = 'bgrc'
x_ = np.arange(78)
for i in range(4):
    y_ = mse_control.mean(1)[:,i]
    j_ = p_control[:,i] <= 0.01
    for xx, yy in zip(x_[j_], y_[j_]):
        pl.text(xx, yy, '*', fontsize=20)