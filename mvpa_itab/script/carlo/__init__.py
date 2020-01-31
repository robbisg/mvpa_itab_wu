############## Convert ######################
from mvpa_itab.io.base import load_dataset, read_configuration
from mvpa_itab.main_wu import detrend_dataset
from mvpa_itab.timewise import AverageLinearCSVM, ErrorPerTrial, StoreResults
from mvpa2.measures.base import CrossValidation, Dataset
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import mean_group_sample
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error
import nibabel as ni
import numpy as np
from mvpa2.clfs.base import Classifier
from mvpa2.generators.resampling import Balancer

from sklearn.cross_validation import StratifiedKFold

from sklearn.svm import SVC
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from mvpa2.suite import debug, sphere_searchlight
from mvpa2.suite import *

from numpy.random.mtrand import permutation
from mvpa_itab.test_wu import load_subjectwise_ds
from mvpa_itab.pipeline.deprecated.partitioner import MemoryGroupSubjectPartitioner




subjects =  [
             '110929angque',
             '110929anngio',
             '111004edolat',
             '111006giaman',
             '111006rossep',
             '111011marlam',
             '111011corbev',
             '111013fabgue',
             '111018montor',
             '111020adefer',
             '111020mardep',
             '111027ilepac',
             '111123marcai',
             '111123roblan',
             '111129mirgra',
             '111202cincal',
             '111206marant',
             '111214angfav',
             '111214cardin',
             '111220fraarg',
             '111220martes',
             '120119maulig',
             '120126andspo',
             '120112jaclau'
             ]       

group1 =    [
             '110929angque',
             '110929anngio',
             '111004edolat',
             '111006giaman',
             '111006rossep',
             '111011marlam',
             '111123marcai',
             '111123roblan',
             '111129mirgra',
             '111202cincal',
             '111206marant',
             '120126andspo',
             ]

tasks = ['memory', 
         'decision']

evidences = [1, 3, 5]



