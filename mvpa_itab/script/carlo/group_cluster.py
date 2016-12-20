import numpy as np
import os
import nibabel as ni
from mvpa2.suite import *
import mvpa2.algorithms.group_clusterthr as gct
from itertools import product

if __debug__:
    debug.active += ["GCTHR"]
subjects = ['110929angque',
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
#              '111214cardin',
#              '111220fraarg',
#              '111220martes',
#              '120119maulig',
#              '120126andspo',
#              '120112jaclau'
             ]

mask_pattern = '%s_mask.nii.gz'
masks_ = [ni.load(os.path.join('/media/robbis/DATA/fmri/memory/', mask_pattern % (s))).get_data()
          for s in subjects]
img = ni.load(os.path.join('/media/robbis/DATA/fmri/memory/', mask_pattern % (s)))

masks_ = np.array(masks_).mean(axis=0)
mask = ni.Nifti1Image(np.int_(masks_ > 0.5)[...,:np.newaxis], img.get_affine())

# Load data
path = '/media/robbis/DATA/fmri/memory/0_results/permutation/'
pattern = "%s_permutation_3_memory_3_float32.nii.gz"

img_list = [ni.load(os.path.join(path, pattern %(s))).get_data() for s in subjects]

ds = fmri_dataset(img_list, chunks=np.repeat(range(len(subjects)), 100), mask=mask)

# Group clustering
feature_thresh_prob = 0.005
clthr = gct.GroupClusterThreshold(n_bootstrap=1000,
                                  feature_thresh_prob=feature_thresh_prob,
                                  fwe_rate=0.01, n_blocks=3)
clthr.train(ds)
thr = clthr._thrmap
 
# Verificare come fare i blob
# I blob è una mappa media di ogni soggetto
task_list = ['memory', 'decision']
evidence_list = ['1', '3', '5']
path_blob = '/media/robbis/DATA/fmri/memory/0_results/balanced_analysis/local'
file_blob = "total_%s_evidence_%s_total.nii.gz"

for task, evidence in product(task_list, evidence_list):
    filename = os.path.join(path_blob, file_blob % (task, evidence))                     
    blob = fmri_dataset(ni.load(filename), mask=mask)
    blob = blob[:len(subjects)]
    res = clthr(blob)
    results = np.zeros_like(res.samples.squeeze())
    cluster = res.fa.clusters_featurewise_thresh
    for cluster_label in np.unique(cluster)[1:]:
        results[cluster == cluster_label] = res.a.clusterstats[cluster_label-1]['prob_raw']
    
    results_map = ds.mapper.reverse1(results)
    output_name = "total_%s_evidence_%s_total_permutation_cluster.nii.gz" % (task, evidence)
    ni.save(ni.Nifti1Image(results_map, img.get_affine()), os.path.join(path_blob, output_name))

for i in ['1','3','5']:
    img_ = ni.load(os.path.join(path_blob, "total_memory_evidence_%s_total.nii.gz" % i))
    data = img_.get_data()
    data = data[data!=0]
    freq, bins = np.histogram(data, bins=np.linspace(0.4, 0.7, 50))
    pdf = gaussian_kde(data)
    pl.plot(pdf(bins), label=i)

pattern = "%s_permutation_3_memory_3_float32.nii.gz"
permutation_fname = "%s_permutation_3_memory_3_float32_demeaned.nii.gz"
mask_pattern = '%s_mask.nii.gz'
path = '/media/robbis/DATA/fmri/memory/0_results/permutation/'

for i, s in enumerate(subjects):
    
    mask_img = ni.load(os.path.join('/media/robbis/DATA/fmri/memory/', mask_pattern % (s)))
    
    img = ni.load(os.path.join(path, pattern %(s)))
        
    masked_data = apply_mask(img, mask_img)
        
    mean_ = masked_data.mean()
    demeaned_data = masked_data - mean_
        
    demeaned_img = unmask(demeaned_data, mask_img)
        
    save_pattern_ = permutation_fname % (s)
    
    ni.save(demeaned_img, path+save_pattern_)
    
    
    
    
    
    
    
    
    