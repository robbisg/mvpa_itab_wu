#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

import numpy as np
import scipy.signal as ssig
import scipy.stats as sstat
from matplotlib.mlab import specgram, psd
from spectrum import *
from numpy.fft import *
        
        
          
def spectrum_eeg(data, **kwargs):
    
    
    
    if kwargs['type'] == 'psd':
        [p, f] = psd_eeg(data, **kwargs)
    else:
        [p, f] = fft_eeg(data, **kwargs)  
    
      
              
    return [p, f]

def psd_eeg(data, **kwargs):
    
    for arg in kwargs:
        if (arg == 'NFFT'):
            NFFT = int(kwargs[arg])
        if (arg == 'dt'):
            Fs = 1./float(kwargs[arg])
        if (arg == 'noverlap'):
            noverlap = int(kwargs[arg])
    
    
    px_list = []
    
    print 'PSD Computing...'      
    for sample in range(data.shape[0]):
        for ch in range(data.shape[1]):
            eeg = data[sample, ch, :]
            [Pxx, freq, t] = specgram(eeg, NFFT=NFFT, noverlap=noverlap, Fs=Fs)
            px_list.append(Pxx)
    
    
    shape = px_list[0].shape
    pot = np.array(px_list)
    
    pot = pot.reshape(data.shape[0], data.shape[1], shape[0], -1)
    
            
    del px_list
    
    return [pot, freq]
           



def fft_eeg(data, **kwargs):
    
    for arg in kwargs:
        if (arg == 'dt'):
            Fs = 1/float(kwargs[arg]) 
                
    
    px_list = []
    ph_list = []
    
    print 'FFT Computing...'   
    for sample in range(data.shape[0]):
        for ch in range(data.shape[1]):
            eeg = data[sample, ch, :]
            four = fft(eeg)
            pxx = np.power(np.abs(four), 2)
            pxx = pxx / np.max(pxx)
            phase = np.angle(four)/np.pi
            freq = fftfreq(eeg.size, d=1./Fs)
            
            pxx = pxx[:pxx.size/2]
            phase = phase[:phase.size/2]
            freq = freq[:freq.size/2]
            
            px_list.append(pxx)
            ph_list.append(phase)
  
  
    phi = np.array(ph_list)
    pot = np.array(px_list)
    
    phi = phi.reshape(data.shape[0], data.shape[1], -1)
    pot = pot.reshape(data.shape[0], data.shape[1], -1)
     
    samples = np.dstack((pot, phi))
                        
    del phi, pot, px_list, ph_list
            
    return [samples, freq]
        
   
