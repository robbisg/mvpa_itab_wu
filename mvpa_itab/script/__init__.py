import os
os.environ['SKLEARN_SITE_JOBLIB'] = "1"
from dask.distributed import Client
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
import joblib


client = Client(processes=False)
joblib.parallel_backend('dask')

diabetes = datasets.load_diabetes()
X = diabetes.data[:50]
y = diabetes.target[:50]


model = linear_model.LinearRegression()

cv_results = cross_validate(model, X, y, cv=10, return_train_score=False,
                            verbose=100)

##################################################################
from dask_kubernetes import KubeCluster                             
from dask.distributed import Client
import os
os.environ['SKLEARN_SITE_JOBLIB'] = "1"
from dask.distributed import Client
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from pyitab.ext.sklearn._validation import cross_validate
import joblib


cluster = KubeCluster.from_yaml('pods.yml')
                                                                                                                          
pods = cluster.scale(6)
client = Client(cluster.scheduler_address)

diabetes = datasets.load_diabetes()
from dask.array import from_array
X = from_array(diabetes.data, chunks='auto')
y = from_array(diabetes.target, chunks='auto')


model = linear_model.LinearRegression()

with joblib.parallel_backend('dask', scatter=[model, X, y]):
    cv_results = cross_validate(model, X, y, cv=20, return_train_score=False,
                            verbose=1)

###################################
from pyitab.analysis.searchlight import SearchLight
from sklearn.model_selection import *
from pyitab.utils import load_test_dataset
import joblib
from dask.distributed import Client
from dask_kubernetes import KubeCluster

cluster = KubeCluster.from_yaml('pods.yml')
                                                                                                                          
pods = cluster.scale(6)
client = Client(cluster.scheduler_address)

ds = load_test_dataset()

cv = KFold()
with joblib.parallel_backend('dask'): 
    scores = SearchLight(cv=cv).fit(ds) 