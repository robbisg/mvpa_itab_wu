from mvpa.suite import *
import nibabel as ni
import datetime as ora
from papyon.util.iso8601.iso8601 import frac


def analyzeMonkTime(fmri4d, path, attrib, mask):

#From the directory of the dataset, we extract
    #name = 'chrwoo'

                               
    attr = SampleAttributes(path+attrib)
    
    
    mydataset = fmri_dataset(fmri4d, targets = attr.targets, chunks = attr.chunks, mask = mask) 
    
    #mydataset = mydataset[mydataset.sa.targets != 'RestNo']
         
    print 'fMRI dataset loaded...'
    #Detrending ---- is it necessary??
    detrender = PolyDetrendMapper(polyord = 1, chunks_attr = 'chunks');
    detrended_ds = mydataset.get_mapped(detrender);
    
    orig_ds = mydataset.copy()
    
    
    #Detrending operation with baseline the resting state
    zscore(detrended_ds, chunks_attr = 'chunks', param_est =('targets',['Rest']))
    #zscore(detrended_ds)
    
    
    events = find_events(targets = detrended_ds.sa.targets, chunks = detrended_ds.sa.chunks)
    
    
    #Changing event duration to watch time profiles before and after stimulus
    
    detrended_ds = detrended_ds[detrended_ds.sa.targets != 'NoRest']
    
    task_events = [e for e in events if e['targets'] in ['Vipassana','Samatha']]

    
    
    for e in task_events:
        e['onset'] += 2
        e['duration'] = 20
        print e
    
    evds = eventrelated_dataset(detrended_ds, events = task_events)
    
    clf = LinearCSVMC()
    
    fsel = FeatureSelection(OneWayAnova(), FractionTailSelector(0.1, mode = 'select', tail = 'upper'))
    
    fclf = FeatureSelectionClassifier(clf, fsel)
    
    sclf = SplitClassifier(fclf, NGroupPartitioner(3), enable_ca=['stats','repetition_results'])
    
    
    
    #cv = CrossValidation(SVM(), NFoldPartitioner(cvtype=2), enable_ca=['stats', 'repetition_results'])
    
    sensana = sclf.get_sensitivity_analyzer()
    
    sens = sensana(evds)
    
    sens_comb_two = evds.a.mapper.reverse(sens.samples) 
    
    '''
    Creates a dataset with classifier weights
    '''
    sens_ds = dataset_wizard(sens_comb_two)
    
    print 'Writing Nifti sensitivity map...'
    
    '''
    Save maps to disk
    '''
    #niftiresults = map2nifti(sens_ds, imghdr=mydataset.a.imghdr) 
    
    #ni.save(niftiresults,'/media/DATA/fmri/spatio_temporal.nii.gz')
    
    
    example_voxels = [(52,23,33), (49,21,33)]

    """

    First we plot the orginal signal after initial detrending. To do this, we
    apply the timeseries segmentation to the original detrended dataset and
    plot to mean signal for all face and house events for both of our example
    voxels.

    """
    vx_lty = ['-', '--']
    t_col = ['b', 'r']

    pl.subplot(311)
    for i, v in enumerate(example_voxels):
        slicer = np.array([tuple(idx) == v for idx in mydataset.fa.voxel_indices])
        evds_detrend = eventrelated_dataset(detrended_ds[:, slicer], events=events)
        for j, t in enumerate(evds.uniquetargets):
            pl.plot(np.mean(evds_detrend[evds_detrend.sa.targets == t], axis=0),
                    t_col[j] + vx_lty[i],
                    label='Voxel %i: %s' % (i, t))
            pl.ylabel('Detrended signal')
            pl.axhline(linestyle='--', color='0.6')
            pl.legend()

"""

In the next step we do exactly the same again, but this time for the
normalized data.

"""

    pl.subplot(312)
    for i, v in enumerate(example_voxels):
        slicer = np.array([tuple(idx) == v for idx in mydataset.fa.voxel_indices])
        evds_norm = eventrelated_dataset(detrended_ds[:, slicer], events=events)
        for j, t in enumerate(evds.uniquetargets):
            pl.plot(np.mean(evds_norm[evds_norm.sa.targets == t], axis=0),
                    t_col[j] + vx_lty[i])
            pl.ylabel('Normalized signal')
            pl.axhline(linestyle='--', color='0.6')

"""

Finally, we plot the associated SVM weight profile for each peristimulus
timepoint of both voxels. For easier selection we do a little trick and
reverse-map the sensitivity profile through the last mapper in the
dataset's chain mapper (look at ``evds.a.mapper`` for the whole chain).
This will reshape the sensitivities into ``cross-validation fold x volume x
voxel features``.

"""

    pl.subplot(313)
    smaps = evds.a.mapper[-1].reverse(sensitivities)

    for i, v in enumerate(example_voxels):
        slicer = np.array([tuple(idx) == v for idx in mydataset.fa.voxel_indices])
        smap = smaps.samples[:,:,slicer].squeeze()
        plot_err_line(smap, fmt='ko', linestyle=vx_lty[i])
        pl.xlim((0,12))
        pl.ylabel('Sensitivity')
        pl.axhline(linestyle='--', color='0.6')
        pl.xlabel('Peristimulus volumes')

    if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
        pl.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    