############## Convert ######################
from mvpa_itab.io.base import load_dataset, read_configuration
from mvpa_itab.main_wu import preprocess_dataset
from mvpa2.measures.base import CrossValidation, Dataset
from mvpa2.generators.partition import NFoldPartitioner, HalfPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.suite import mean_group_sample
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.misc.errorfx import mean_mismatch_error

import numpy as np
from mvpa2.clfs.base import Classifier
from mvpa2.generators.resampling import Balancer
path = '/media/robbis/DATA/fmri/memory/'#110929anngio/'
"""
fold = ['RESIDUALS_MVPA', 'RESIDUALS_MVPA/SINGLE_TRIALS']

ff = open('convert.sh', 'w')

for d in fold:
    flist = os.listdir(os.path.join(path,d))
    flist = [f for f in flist if f.find('.ifh') != -1]
    for f in flist:
        fin = os.path.join(path,d,f.split('.4dfp.ifh')[0])
        
        ff.write('nifti_4dfp -n '+fin+' '+fin+'\n')
        
        fout = os.path.join(path,d,f.split('.ifh')[0])
        ff.write('rm '+fout+'.img'+'\n')
        ff.write('rm '+fout+'.ifh'+'\n')
        
ff.close()



################ Rename + Merge Betas ######################
import csv
path = '/media/robbis/DATA/fmri/memory/111006giaman/'

#csvfile = open(os.path.join(path, 's05_giaman_4RES-CLA.csv'), 'r')
csvfile = open(os.path.join(path, 's02_anngio_4RES-CLA.csv'), 'r')
csvrd = csv.reader(csvfile, delimiter=',', quotechar='|')
line0 = csvrd.next()
line0 = line0[1:]
line1 = csvrd.next()
line1 = line1[:360-len(line0)]
events = line0 + line1


#fold = ['BETA_GOonly', 'BETA_IMA_GO']
fold = ['RESIDUALS_MVPA/SINGLE_TRIALS']
ff = open('move_fl.sh', 'w')

for d in fold:
    flist = os.listdir(os.path.join(path,d))
    ordd = []
    for i, item in enumerate(events):
        n = [j for j, it in enumerate(flist) if it.find(item+'_') != -1]
        ordd.append(n[0])

    ordflist = np.array(flist)[ordd]

    for i, fl in enumerate(ordflist):
        fin = os.path.join(path,d,fl)
        fout = os.path.join(path,d,'%03d_'%(i+1)+fl)
        ff.write('\nmv '+fin+' '+fout)
      
ff.close()

subjs = os.listdir('/media/robbis/DATA/fmri/memory/')
subjs = [s for s in subjs if s.find('.') == -1 and s.find('_') == -1]

ff = open('fslmerge__.sh', 'w')
for s in subjs:
    path = os.path.join('/media/robbis/DATA/fmri/memory/',s)
    for d in fold:
        flist = os.listdir(os.path.join(path,d))
        flist.sort()
        flist[0] = os.path.join(path,d,flist[0])
        stringa = ' '+os.path.join(path,d,'')
        strlist = stringa.join(flist)
        output = os.path.join(path,d,'beta.nii.gz')
        ff.write('\nfslmerge -t '+output+' '+strlist)
ff.close()

################################################
import glob
for s in subjects:
    path = os.path.join('/media/robbis/DATA/fmri/memory/',s)
    csvlist = glob.glob(path+'/*.csv')

    csvfile = open(csvlist[0], 'r')
    csvrd = csv.reader(csvfile, delimiter=',', quotechar='|')
    line0 = csvrd.next()
    line0 = line0[1:]
    line1 = csvrd.next()
    line1 = line1[:360-len(line0)]
    events = line0 + line1
    events = np.array(events, dtype=np.dtype('|S5'))
    chunks = np.int_(np.linspace(1,12.99,len(events)))
    events = np.vstack((events, chunks))
    np.savetxt(os.path.join(path,'beta_attributes.txt'), events.T, fmt='%s')


"""
class TrialAverager(Classifier):
    
    # Wrapper class, obviously better!
    
    def __init__(self, clf):
        self._clf = clf
        
    def _train(self, ds):
        avg_mapper = mean_group_sample(['trial']) 
        ds = ds.get_mapped(avg_mapper)
        return self._clf._train(ds)
    
    def _call(self, ds):
        # Function overrided to let the results have
        # some dataset attributes
        
        res = self._clf._call(ds)

        if isinstance(res, Dataset):
            for k in ds.sa.keys():
                res.sa[k] = ds.sa[k]
            return res
        else:
            return Dataset(res, sa=ds.sa)


class AverageLinearCSVM(LinearCSVMC):
    """Extension of the classical linear SVM

    Classes inherited from this class gain ability to access
    collections and their items as simple attributes. Access to
    collection items "internals" is done via <collection_name> attribute
    and interface of a corresponding `Collection`.
    """
    
    def __init__(self, C=1):
        LinearCSVMC.__init__(self, C=1)
        
    def _train(self, ds):
        avg_mapper = mean_group_sample(['trial']) 
        ds = ds.get_mapped(avg_mapper)
        return LinearCSVMC._train(self, ds)
    
    
    def _call(self, ds):
        # Function overrided to let the results have
        # some dataset attributes
        
        res = LinearCSVMC._call(self, ds)

        if isinstance(res, Dataset):
            for k in ds.sa.keys():
                res.sa[k] = ds.sa[k]
            return res
        else:
            return Dataset(res, sa=ds.sa)

class StoreResults(object):
    
    def __init__(self):
        self.storage = []
            
    def __call__(self, data, node, result):
        self.storage.append(node.measure.ca.predictions),
        
    def _post_process(self, ds):
        predictions = np.array(self.storage)

class ErrorPerTrial(BinaryFxNode):
    
    def __init__(self, **kwargs):

        BinaryFxNode.__init__(self, fx=mean_mismatch_error, space='targets', **kwargs)


    def _call(self, ds):
        # extract samples and targets and pass them to the errorfx
        targets = ds.sa[self.get_space()].value
        # squeeze to remove bogus dimensions and prevent problems during
        # comparision later on
        values = np.atleast_1d(ds.samples.squeeze())
        if not values.shape == targets.shape:
            # if they have different shape numpy's broadcasting might introduce
            # pointless stuff (compare individual features or yield a single
            # boolean
            raise ValueError("Trying to compute an error between data of "
                             "different shape (%s vs. %s)."
                             % (values.shape, targets.shape))
        err = [self.fx(values[ds.sa.frame == i], targets[ds.sa.frame == i]) 
               for i in np.unique(ds.sa.frame)]
        
        return Dataset(np.array(err).flatten())
    
subjects = ['110929angque', '110929anngio']
tasks = ['RESIDUALS_MVPA', 'BETA_MVPA']
res = []
for subj in subjects:
    for task in tasks:
        conf = read_configuration(path, 'remote_memory.conf', task)
        ds = load_dataset(path, subj, task, **conf)

        ds = preprocess_dataset(ds, task, **conf)
        
        ds.targets = ds.sa.decision
        
        balanc = Balancer(count=3, apply_selection=True)
        gen = balanc.generate(ds)
        #clf = AverageLinearCSVM(C=1)
        
        clf = LinearCSVMC(C=1)
        #avg = TrialAverager(clf)
        #cv_storage = StoreResults()

        cvte = CrossValidation(clf,
                               HalfPartitioner(),
                               #errorfx=ErrorPerTrial(), 
                               #callback=cv_storage,
                               enable_ca=['stats', 'probabilities'])

        sl = sphere_searchlight(cvte, radius=3, space = 'voxel_indices')
        maps = []
        for ds_ in gen:
            print 'Here we are!'
            sl_map = sl(ds_)
            sl_map.samples *= -1
            sl_map.samples +=  1
            sl_map.samples = sl_map.mean(axis=len(sl_map.shape)-1)
            map = map2nifti(sl_map, imghdr=ds.a.imghdr)
            maps.append(map)
        
        res.append(maps)
            #err = cvte(ds_)
