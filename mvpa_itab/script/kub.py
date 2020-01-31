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
with joblib.parallel_backend('dask', scatter=[ds]): 
    scores = SearchLight(cv=cv).fit(ds) 