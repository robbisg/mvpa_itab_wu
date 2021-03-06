import numpy as np
from mvpa2.misc.io.base import SampleAttributes
from mvpa2.datasets.base import Dataset
from mvpa_itab.eeg_load import spectrum_eeg

def load_eeg_data(path, filename, TR, eliminatedVols = None):
    """
    @return: Dataset from mvpa2.suite
    
    @param path: Where the file is located
    @param filename: The name of the file to be read (N.: filename shouldn't have extensions)
    @param attrib: The list of labels linked with the dataset.
    @param TR: The lenght of data to be included in one example. 
    """
    hdr = open(path+filename+'.vhdr', 'r')

    i = 0
    line = 'on'
    print ' ************** EEG Loading *******************'
    print 'Loading '+filename
    while len(line)!= 0 :
        line = hdr.readline()
        i+=1
        if (line.count('Coordinates') == 1):
            c = i
        if (line.count('SamplingInterval=') == 1):
            dt = float(line.split('=')[1]) / 1000000.
        if (line.count('Channel Infos') == 1):
            info = i

    #print i

    c += 1; 
    info += 1

    print path+filename+'.vhdr'+' reading....'
    loc = np.genfromtxt(path+filename+'.vhdr',
                    dtype={'names': ('radius', 'theta', 'psi'),
                            'formats': ('f4', 'f4', 'f4')},
                    delimiter = ',',
                    comments=';',
                    converters = {0: lambda s: float(s[s.find('=')+1:])},
                    skip_header = c)

    print info

    ch_info = np.genfromtxt(path+filename+'.vhdr',
                                dtype={'names': ('label', 'reference', 'resolution', 'unit'),
                                       'formats': ('S10', 'S1','S1','S3')},
                               delimiter = ',',
                               comments=';',
                               usecols = (0),
                               skip_footer = i - c,
                               converters = {0: lambda s: s[s.find('=')+1:]},
                               skip_header = info)


    mrk = open(path+filename+'.vmrk', 'r')

    i=0
    line = 'on'
    while len(line)!= 0 :
        line = mrk.readline()
        i += 1
        if (line.count('Marker Infos') == 1):
            c = i
    
    print 'Markers reading....'
    markers = np.genfromtxt(path+filename+'.vmrk',
                            dtype={'names': ('volume', 'descr', 't_position', 'size','chan-no'),
                               'formats': ('i4', 'S12', 'i4', 'S1', 'i4')},
                            delimiter = ',',
                            comments=';',
                            converters = {0: lambda s: int(s[2:s.find('=')])-1},
                            usecols = (0,2),
                            skip_header = c+1)

    #Canceling first element of markers!
    markers = markers[1:]
    
    print 'Loading data....'
    #EEG data loadin
    data = np.loadtxt(path+filename+'.dat', skiprows = 1, dtype='float32')
    
    print 'EEG channels referencing....'
    #Re-reference signals
    data = np.array([(x - np.average(x)) for x in data])

    #Calculate signal variance 
    stat = np.array([np.var(x) for x in data.T])
    
    #Index of channels with high variance
    ordChannels = np.argsort(stat.T)
    
    #Selects first n. channels with high variance
    channelSelected = np.sort(ordChannels[:])

    #offset in collected data
    offset = markers[0][1]
    
    #n. of samples to include (in sec.)
    nSamples = TR #sec.
    onSet = -0. #sec.
    
    #total no. of channels
    nChannels = ordChannels.shape[0]
    
    #no. of trials
    nTrials = markers.shape[0]

    #Constructing array to store samples
    dataRes = np.empty([nChannels, nTrials, nSamples/dt], dtype='float32')

    for i in range(nTrials):
        for j in channelSelected:
            start = markers[i][1] + (onSet/dt)
            stop  = start + (nSamples/dt)
            nVec = data.T[j][start:stop]
            dataRes [j][i] = nVec

    #Select data based on channels Selected
    dataRes = np.take(dataRes, channelSelected, axis=0)
    ch_infoSel = np.take(ch_info, channelSelected)

    #reshape data to feed Dataset
    dataRes = np.reshape(dataRes, (nTrials, -1, nSamples/dt))
    del data
    #Eliminate first n. runs
    
    if eliminatedVols == None:
        volToEliminate = 0
    else:
        volToEliminate = eliminatedVols
    dataRes = dataRes[:nTrials-volToEliminate]

    eeg_keys = ['channel_ids', 'dt', 'locations']
    eeg_value = [ch_infoSel, dt, loc]
    
    
    eeg_info = dict(zip(eeg_keys, eeg_value))
   
    return dataRes, eeg_info
  
  
def load_eeg_dataset(path, filename, attrib, TR, eliminated_vols = None, **kwargs):
    """
    
    **kwargs:
    
       - type: 
              * 'psd': Power Spectrum Density using matplotlib.specgram
                       additional parameters to be included NFFT and noverlap
              * 'fft': Power Spectrum and Phase using scipy.fft
              * 'time': EEG timecourse in every TR. 
    """
    
    type = 'time'
    
    for arg in kwargs:
        if (arg == 'type'):
            type = kwargs[arg]

    
    print 'type = ' + type
    #load eeg data
    [data, eeg_info] = load_eeg_data(path, filename, TR, eliminatedVols = eliminated_vols)
    
    channel_ids = eeg_info['channel_ids']
    dt = eeg_info['dt']
    
    kwargs['dt'] = dt
    
    if (type == 'psd') or (type == 'fft'):
        [samples, freq] = spectrum_eeg(data, **kwargs)

        data = samples.reshape(samples.shape[0], samples.shape[1], -1)
    
    #mvpa analysis: attributes and dataset
    attr = SampleAttributes(attrib)
    print 'Building dataset...'
    ds = Dataset.from_channeltimeseries(data, 
                                        channelids = channel_ids,
                                        targets = attr.targets,
                                        chunks = attr.chunks)
    
    if (type == 'psd') or (type == 'fft'):
        ds.a['frequiencies'] = freq
        ds.a['shape'] = samples.shape
    
    ds.a['timepoints'] = np.arange(0, TR, dt)
        
    del data
    
    if 'samples' in locals():
        del samples
    
    return ds


def sphere2cartesianLocation (location):
    import math
    
    newLocs = np.empty((location.shape[0], 3))
    
    f = np.pi/180
    
    
    for i in range(location.shape[0]):
        newLocs[i] = [location[i]['radius'] * math.cos(location[i]['psi'] * f) * math.sin(location[i]['theta'] * f),
                      location[i]['radius'] * math.sin(location[i]['psi'] * f) * math.sin(location[i]['theta'] * f),
                      location[i]['radius'] * math.cos(location[i]['theta'] * f)]
        
    return np.array(newLocs)