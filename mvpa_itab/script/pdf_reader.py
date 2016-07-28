import os
import numpy as np
path_ = '/home/robbis/rossep/'
dirs_ = os.listdir(path_)

dirs_.sort()
dirs_ = dirs_[:-1]

selected = []
total_dirs = len(dirs_)
for i,d in enumerate(dirs_):
    path_file = os.path.join(path_, d)
    
    output = os.popen('ls '+path_file+'/*.txt').read()
    outlist = output.split('\n')
    outlist.sort()
    
    total_files = len(outlist[1:])
    for j,f in enumerate(outlist[1:]):
        
        cat_ = os.popen('cat '+f).read()
        
        conditions = np.array([cat_.find('figure(') != -1,
                      cat_.find('<?xml') == -1,
                      cat_.find('import') != -1,
                      cat_.find('Package:') == -1
                      ], dtype=np.bool)
        
        if np.all(conditions):
            selected.append(f)
        
        progress = 100. * ((i*total_files)+(j+1))/(total_dirs * total_files)
        update_progress(progress)


def update_progress(progress):
    sys.stdout.write( '\r[{0}] {1}%'.format('#'*np.int(progress/10), progress))
    sys.stdout.flush()  