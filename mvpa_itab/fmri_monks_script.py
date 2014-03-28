#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

from mvpa2.suite import *
import nibabel as ni
import datetime as ora

#path = '/home/robbis/fmri_datasets/monks/'
#attrib = 'monks_attrib_pymvpa.txt'


def analyzeMonk(name, fmri4d, path, attrib, mask):
    
    
    '''
    Time information
    '''
    timenow = ora.datetime.now()
    timeString = timenow.strftime('%y%m%d')

    experimentInfo = [timeString]
    
    '''
    Loading attributes
    '''                          
    attr = SampleAttributes(path+attrib)  
    
    '''
    Dataset loading:
    takes information from attributes to assign labels and chunks
    '''
    mydataset = fmri_dataset(fmri4d, targets = attr.targets, chunks = attr.chunks, mask = mask) 

           
    print 'fMRI dataset loaded...'
    
    '''
    Detrending
    '''
    poly_detrend(mydataset, polyord = 1, chunks_attr = 'chunks');
    
    detrended_ds = mydataset;
    
    nodeD = detrended_ds.a.mapper.nodes[len(detrended_ds.a.mapper.nodes)-1]
    experimentInfo.append(str(nodeD))
    
    
    '''
    Zscoring operation (targets indicates to which mean and variance it normalizes data)
    '''
    #zscore(detrended_ds, param_est =('targets',['Rest']))
    zscore(detrended_ds)
    fds = detrended_ds
    
    nodeZ = detrended_ds.a.mapper.nodes[len(detrended_ds.a.mapper.nodes)-1]
    experimentInfo.append(str(nodeZ) + str(nodeZ.param_est))
    
    '''
    Deleting volumes unuseful for classification, belonging to label that we don't want to include
    '''
    fds = fds[fds.sa.targets != 'Rest']
    fds = fds[fds.sa.targets != 'RestNo']
    #fds = fds[fds.sa.targets != 'NoVolume']
    #fds = fds[fds.sa.targets != 'NoControl']
    
    spcl = get_samples_per_chunk_target(fds)
    experimentInfo.append(zip(fds.sa['targets'].unique, spcl[0]))
    
    '''
    Setting up classifier algorithm
    '''
    clf = LinearCSVMC()
        
    '''
    Extract information from classifier class
    '''
    summary = clf.summary()
    clfName = summary[summary.find('<')+1:summary.find('>')]
    experimentInfo.append(clfName)

    '''
    Feature selection method: 
    it uses 
        - a measure function, to rank features
        - a selector, to have a criterion to select features from the rank
    '''
    #fsel = SensitivityBasedFeatureSelection(OneWayAnova(), FractionTailSelector(0.1, mode = 'select', tail = 'upper'))
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  custom_tail_selector)
    

    nodeFSL = fsel._SensitivityBasedFeatureSelection__feature_selector
    nodeSENSA = fsel._SensitivityBasedFeatureSelection__sensitivity_analyzer
    experimentInfo.append(str(nodeFSL)+str(nodeSENSA))
    '''
    Unify Feature Selection and Classifier
    '''
    fclf = FeatureSelectionClassifier(clf, fsel)
    
    '''
    Setting up cross-validation
        - it needs a partitioner to split data
    '''
    cvte = CrossValidation(fclf, NFoldPartitioner(cvtype = 1), enable_ca=['stats', 'repetition_results'])
    experimentInfo.append(str(cvte.generator))
    
    print 'Classifying with '+clfName+' ...'
    
    '''
    Running cross-validation with selected classifier on dataset
    '''
    cv_results = cvte (fds)
        
    '''
    Get object to extract sensitivity, with respect to classifier
    '''
    cv_sensana = fclf.get_sensitivity_analyzer()
    
    '''
    Running sensitivity analysis on dataset
    '''
    sens = cv_sensana(fds)


    '''
    Reverse classifier weights to dataset space (3D)
    '''
    sens_comb_two = mydataset.a.mapper.reverse(sens.samples) 
    
    #fileResult = open(pathFile+'/20010407_xxxxxx.txt','w')
    
    #plot_args = { 
                # 'background' : pathFile+'MPRAGE/'+name+'_mprage_TAL.hdr',
                #  'background_mask' : pathFile+'MPRAGE/MNI_brain_template_TAL.nii.gz',
    #            'do_stretch_colors' : False, 
    #            'cmap_bg' : 'gray', 
    #            'cmap_overlay' : 'autumn', # YlOrRd_r # pl.cm.autumn
    #            'interactive' : cfg.getboolean('examples', 'interactive', True), 
    #            }


    #niftiresults = map2nifti(sens, imghdr=mydataset.a.imghdr)


    #fig = pl.figure(figsize=(12, 4), facecolor='white')
    #subfig = plot_lightbox(overlay=sens_comb_two[0], vlim=(0.5, None), slices=range(23,25), fig=fig, **plot_args)

    '''
    Creates a dataset with classifier weights
    '''
    sens_ds = dataset_wizard(sens_comb_two)
    
    print 'Writing Nifti sensitivity map...'
    
    '''
    Save maps to disk
    '''
    niftiresults = map2nifti(sens_ds, imghdr=mydataset.a.imghdr) 

    
    print 'Writing matrix results...'
 


    print 'Mean error on cv folds/n'
    print np.mean(cv_results)


    print 'TOTAL CONFUSION MATRIX\n'
    print cvte.ca.stats

    
    resultsData = [cvte.ca.stats.stats['ACC'], cvte.ca.stats, cvte.ca.stats.matrices]
        
    return {'subj':name, 'pipeline':experimentInfo, 'results': resultsData, 'map':niftiresults}
    
    
def custom_tail_selector(seq):
    seq1 = FractionTailSelector(0.0, mode='discard', tail='upper')(seq)
    seq2 = FractionTailSelector(0.1, mode='select', tail='upper')(seq)
    return list(set(seq1).intersection(seq2))

#fileResult.close()
#Start from python console execfile('/home/robbis/development/eclipse/PyMVPA/src/script.py')
