from mvpa2.suite import *

def multimodal_mvpa(ds_eeg, ds_fmri, clf):
    
    print '*********** Multimodal MVPA ************'
    
    print 'Detrending datasets...'
    poly_detrend(ds_fmri, polyord = 1, chunks_attr = 'chunks');
    
    poly_detrend(ds_eeg, polyord = 1, chunks_attr = 'chunks');
    
    print 'Zscoring datasets...'
    zscore(ds_fmri)

    zscore(ds_eeg)
    
    featureDiff = ds_eeg.samples.shape[1] - ds_fmri.samples.shape[1]
     
    print 'Feature difference is :' + str(featureDiff) + '[positive value = EEG feature are more than fMRI]'
    
    data_fmri = ds_fmri.samples
    data_eeg = ds_eeg.samples
    
    data_fmrieeg = np.hstack([data_fmri, data_eeg])
    
    #if ds_fmri.sa.chunks == ds_eeg.sa.chunks:
    ds_fmrieeg = Dataset.from_wizard(data_fmrieeg, targets = ds_fmri.sa.targets, chunks = ds_fmri.sa.chunks)
        
    del data_fmri, data_eeg, data_fmrieeg
    
    ds_fmrieeg = ds_fmrieeg[ds_fmrieeg.sa.targets != 'Rest']
    
    clf = LinearCSVMC()
    
    #fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  FractionTailSelector(0.4, mode = 'select', tail = 'upper'))
    
    #fclf = FeatureSelectionClassifier(clf, fsel)
    
    cvte = CrossValidation(clf, NFoldPartitioner(cvtype = 1), enable_ca=['stats', 'repetition_results'])
    
    res = cvte(ds_fmrieeg)
    
    print cvte.ca.stats
    
    sensana = clf.get_sensitivity_analyzer()
    
    sens_fmrieeg = sensana(ds_fmrieeg)
    
    sens_tot = np.hsplit(sens_fmrieeg, np.array([ds_fmri.samples.shape[1]]))
    
    sens_fmri = sens_tot[0]
    sens_eeg = sens_tot[1]
    
    rev_fmri = ds_fmri.a.mapper.reverse(sens_fmri)
    rev_eeg = ds_eeg.a.mapper.reverse(sens_eeg)
    
    #plot rev_fmri, rev_eeg
    
    #np.savetxt('/media/DATA/fmri/monks/results/'+name)
    
    return dict({'fmri': rev_fmri, 'eeg': rev_eeg})
    
def analyze(ds):
    
    print '*********** MVPA ************'
    
    print 'Detrending datasets...'
    poly_detrend(ds, polyord = 1, chunks_attr = 'chunks');
    
    print 'Zscoring datasets...'
    zscore(ds)
        
    ds = ds[ds.sa.targets != 'Rest']
    
    clf = LinearCSVMC()
    
    #fsel = SensitivityBasedFeatureSelection(OneWayAnova(),  FractionTailSelector(0.4, mode = 'select', tail = 'upper'))
    
    #fclf = FeatureSelectionClassifier(clf, fsel)
    
    cvte = CrossValidation(clf, NFoldPartitioner(cvtype = 1), enable_ca=['stats', 'repetition_results'])
    
    res = cvte(ds)
    
    print cvte.ca.stats   
    
    