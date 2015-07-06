from ..main_wu import *
from mvpa_itab.lib_io import read_configuration

path = '/media/robbis/DATA/fmri/memory/'
subjects = ['110929angque', '110929anngio', '111006giaman']

conf = read_configuration(path, 'memory.conf', 'BETA_GOonly')
tasks = conf['types'].split(',')

for ss in subjects:
    
    continue

    