#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

import numpy as np
from eeg_load import calSignalFeatures
from mvpa2.suite import Dataset, SampleAttributes
path = '/home/robbis/Share/'
filename = 'monaco50002_Noch_1-45'
hdr = open(path+filename+'.vhdr', 'r')

i = 0
line = 'on'
while len(line)!= 0 :
    line = hdr.readline()
    i+=1
    if (line.count('Coordinates') == 1):
        c = i
    if (line.count('SamplingInterval=') == 1):
        dt = float(line.split('=')[1]) / 1000000.
    if (line.count('Channel Infos') == 1):
        info = i

print i

c += 1; 
info += 1


loc = np.genfromtxt(path+filename+'.vhdr',
                    dtype={'names': ('radius', 'theta', 'psi'),
                            'formats': ('f4', 'f4', 'f4')},
                    delimiter = ',',
                    comments=';',
                    converters = {0: lambda s: float(s[s.find('=')+1:])},
                    skiprows = c)

print info

ch_info = np.genfromtxt(path+filename+'.vhdr',
                        dtype={'names': ('label', 'reference', 'resolution', 'unit'),
                               'formats': ('S10', 'S1','S1','S3')},
                        delimiter = ',',
                        comments=';',
                        usecols = (0),
                        skip_footer = i - c,
                        converters = {0: lambda s: s[s.find('=')+1:]},
                        skiprows = info)


mrk = open(path+filename+'.vmrk', 'r')

i=0
line = 'on'
while len(line)!= 0 :
    line = mrk.readline()
    i += 1
    if (line.count('Marker Infos') == 1):
        c = i

markers = np.genfromtxt(path+filename+'.vmrk',
                        dtype={'names': ('volume', 'descr', 't_position', 'size','chan-no'),
                               'formats': ('i4', 'S12', 'i4', 'S1', 'i4')},
                       delimiter = ',',
                       comments=';',
                       converters = {0: lambda s: int(s[2:s.find('=')])-1},
                       usecols = (0,2),
                       skiprows = c+1)

#Canceling first element of markers!
markers = markers[1:]

#EEG data loadin
data = np.loadtxt(path+filename+'.dat', skiprows = 1, dtype = 'float32')

#Re-reference signals
data = np.array([(x - np.average(x)) for x in data])

#Calculate signal variance
stat = calSignalFeatures(data.T)
#Index of channels with high variance
ordChannels = np.argsort(stat.T)
#Selects first n. channels with high variance
channelSelected = np.sort(ordChannels[:])


#offset in collected data
offset = markers[0][1]
#n. of samples to include (in sec.)
nSamples = 4.08 #sec.
onSet = -0. #sec.
#total no. of channels
nChannels = ordChannels.shape[0]
#no. of trials
nTrials = markers.shape[0]

#Constructing array to store samples
dataRes = np.empty([nChannels, nTrials, nSamples/dt], dtype = 'float32')

for i in range(nTrials):
    for j in channelSelected:
        start = markers[i][1] + (onSet/dt)
        stop  = start + (nSamples/dt)
        nVec = np.array(data.T[j][start:stop])
        if nVec.shape[0] != (nSamples/dt):
            nVec.resize(nSamples/dt)
            print 'yes'
        dataRes [j][i] = nVec

#Select data based on channels Selected
dataResSel = np.take(dataRes, channelSelected, axis=0)
ch_infoSel = np.take(ch_info, channelSelected)

#reshape data to feed Dataset
dataResSel = np.reshape(dataResSel, (nTrials, -1, nSamples/dt))

#Eliminate first n. runs
volToEliminate = 5
dataResSel = dataResSel[:nTrials-volToEliminate]


#mvpa analysis: attributes and dataset
attr = SampleAttributes('/home/robbis/fmri_datasets/monks/monks_attrib_pymvpa.txt')

ds = Dataset.from_channeltimeseries(dataResSel, 
                                    t0 = 0, 
                                    dt = dt, 
                                    channelids = ch_infoSel,
                                    targets = attr.targets,
                                    chunks = attr.chunks)


del dataRes, dataResSel
