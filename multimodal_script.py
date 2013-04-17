from mvpa2.suite import *

import meditation.eeg_load as eeg
import meditation.fmri_load as fmri
from meditation.multimodal import multimodal_mvpa

fmriList = ['chrwoo', 
#            'kuatsa', 
            'lucrus', 
            'supmee', 
            'sawsam', 
            'phimoo', 
            'lucpri', 
            'jutpre', 
#            'phrjan'
            ]

eegList = ['monaco10002_Noch_1-45', 
#           'monaco40002_Noch_1-45', 
            'monaco30002_Noch_1-45', 
            'monaco80002_Noch_1-45', 
            'monaco70002_Noch_1-45', 
            'monaco50002_Noch_1-45',
            'monaco60002_Noch_1-45',
           'monaco90002_Noch_1-45',
 #          'monaco100002_Noch_1-45'
           ]


eegPath = '/media/DATA/fmri/monks/eeg/'

fmriPath = '/media/DATA/fmri/monks/fmri/'

attr_p = '/home/robbis/fmri_datasets/monks/monks_attrib_pymvpa.txt'

######################

file = open('/media/DATA/fmri/monks/multimodal_res_psd_nofsel.txt', 'w')
results = dict()
for eegName, fmriName in zip(eegList, fmriList):
    
    print '--------Analyizing '+fmriName+'-------------'
    
    ds_eeg  =   eeg.load_eeg_dataset(eegPath, eegName, attr_p, 4.086, eliminated_vols = 5, type='time')
    ds_fmri =   fmri.load_fmri_dataset_3d(fmriPath, fmriName, attr_p)
    
    res = multimodal_mvpa(ds_eeg, ds_fmri, LinearCSVMC(C=1))
    '''
    file.write(fmriName)
    file.write(str(res))
    '''
    results[fmriName] = res
    

#file.close()
if eeg = 'True':
    f = plt.figure()
    numSamples, numRows = 2043,30
    #eegfile = cbook.get_sample_data('eeg.dat', asfileobj=False)
    #print ('loading eeg %s' % eegfile)
    #data = np.fromstring(open(eegfile, 'rb').read(), float)
    data = eeg
    data.shape = numSamples, numRows
    t = 4080.0 * np.arange(numSamples, dtype=float)/numSamples
    ticklocs = []
    ax = f.add_subplot(111)
    ax.set_xlim(0,4080)
    ax.set_xticks(np.arange(0,4086, 500))
    dmin = data.min()
    dmax = data.max()
    dr = (dmax - dmin)*0.7 # Crowd them a bit.
    y0 = dmin
    y1 = (numRows-1) * dr + dmax
    ax.set_ylim(y0, y1)

    segs = []
    for i in range(numRows):
        segs.append(np.hstack((t[:,np.newaxis], data[:,i,np.newaxis])))
        ticklocs.append(i*dr)

    offsets = np.zeros((numRows,2), dtype=float)
    offsets[:,1] = ticklocs

    lines = LineCollection(segs, offsets=offsets,
                           transOffset=None,
                           )

    ax.add_collection(lines)

    # set the yticks to use axes coords on the y axis
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(ch_info)

    ax.set_xlabel('time (s)')
