import h5py
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp
fname = "C:/Users/AlessioB/Desktop/REFTEP ANN/sub-1_band-mu_iplv.mat"
mat1 = h5py.File(fname)
fname = "C:/Users/AlessioB/Desktop/REFTEP ANN/sub-1_band-betalow_iplv.mat"
mat2 = h5py.File(fname)
fname = "C:/Users/AlessioB/Desktop/REFTEP ANN/sub-1_band-betahigh_iplv.mat"
mat3 = h5py.File(fname)


X = np.hstack((mat1['iPLV'].value[:,::20],
               mat2['iPLV'].value[:,::20],
               mat3['iPLV'].value[:,::20]))

Y = mat1['AmpsMclean'].value

Y=np.log(Y.T)
#Y=sp.stats.zscore(Y)
#plt.hist(Y)

Y=Y[:,0]
threshold=np.median(Y)
Y[Y<threshold]=0
Y[Y>=threshold]=1

X=X[:,np.std(X,0)>0]
X=np.log(np.abs(X)/(1-np.abs(X)))
#X=sp.stats.zscore(X)

#pca = PCA(n_components=2)
#pca.fit(X.T)
#Xred=pca.components_.T
#Xred=sp.stats.zscore(Xred)


#vectvox=np.random.randint(0,X.shape[1],100)
#vectvox=np.random.permutation(100)
#Xred=X[:,vectvox_app_fewer[vectvox[1:50]]]

Xred=X    
NVox=Xred.shape[1]
SizeLayer=int(NVox/10)
res=np.zeros(100)
for iiter in range(100):
    X_train, X_test, y_train, y_test = train_test_split(Xred, Y, train_size=0.75)
    scaler = StandardScaler()  # doctest: +SKIP
    scaler.fit(X_train)  # doctest: +SKIP
    X_train = scaler.transform(X_train)  # doctest: +SKIP
    X_test = scaler.transform(X_test)  # doctest: +SKIP
    clf = MLPClassifier(hidden_layer_sizes=(SizeLayer), activation='relu', max_iter=500).fit(X_train, y_train)
    res[iiter]=clf.score(X_test, y_test)


plt.hist(res)
plt.show()