#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################
# pylint: disable=maybe-no-member, method-hidden, no-member

import os
import sys
import datetime as ora
import nibabel as ni
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import itertools

#from mvpa2.suite import h5load
      
def enable_logging():
    import logging
    root = logging.getLogger()
    form = logging.Formatter('%(name)s - %(levelname)s: %(lineno)d \t %(filename)s \t%(funcName)s \t --  %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(form)
    root.addHandler(ch)
    root.setLevel(logging.INFO)
    
    return root



def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    
    
def writeResults(experiment, list):
    
    '''
    Extract general information about the experiment
    '''
    listElem = list[0]
    
    settings = listElem['pipeline']
    
    time = settings[0]
    clf = settings[4]
    experiment = settings[len(settings)-1]
    
    path = '/media/DATA/fmri/'+experiment+'/results/'
    newDir = time+'_'+experiment+'_'+clf
    
    timenow = ora.datetime.now()
    timeString = timenow.strftime('%H%M')
    
    if os.listdir(path).count(newDir) == 0:
        print('Making new directory...')
        os.mkdir(path+time+'_'+experiment+'_'+clf)
    else:
        print('Directory found...')
    os.mkdir(path+time+'_'+experiment+'_'+clf+'/'+timeString)
    '''Write experiment settings'''
    pathToNewDir = path+time+'_'+experiment+'_'+clf+'/'+timeString+'/'    
    experimentFile = open(pathToNewDir+'0__analysis_step.txt', 'w')
    
    for elem in settings:
        experimentFile.write(str(elem))
        experimentFile.write('\n')
            
    experimentFile.close()
    
    '''
    Writing results: maps and data
    '''
    
    i = 0
    
    resultSummaryFile = open(pathToNewDir+'0__classifier_results.txt','w')
       
       
    for elem in list:
        
        name = elem['subj']
        results = elem['results']
        
        if results[0] > 0.65:
            res = 'OK'
            i+=1
        else:
            res = 'NO'
        
        '''Storing File with total results for each subject'''
        resultSummaryFile.write(name+'\n')
        resultSummaryFile.write(str(results[0])+'\n')
        resultSummaryFile.write(str(results[1]))
        
        
        '''Storing confusion fold matrix for each subject'''
        subjResultsFile = open(pathToNewDir+name+'__'+res+'_results.txt','w') 
        
        for x in results[len(results)-1]:
            subjResultsFile.write(str(x))
        
        subjResultsFile.close()
        
        '''Saving Pattern Map to Disk'''
        ni.save(elem['map'], pathToNewDir+name+'_'+clf+'__'+res+'__.nii.gz');
    
    
    return i

def loadMask(index):
    
    '''   Load brain mask    '''
    maskList = os.listdir('/media/DATA/fmri/structures')
    
    if (index) <= len(maskList):    
        return maskList[index-1]
    else:
        return 


def fidl_convert(fidlPath, outPath, type):
    
    if type == 1:
        fidl2txt(fidlPath, outPath)
    else:
        fidl2txt_2(fidlPath, outPath)



def fidl2txt (fidlPath, outPath):
    '''
    args:
        fidlPath: complete path to fidl file
        outPath: complete path plus filename of the output
    '''
    
    
    print('Converting fidl file '+fidlPath+' in '+outPath)
    
    fidlFile = open(fidlPath)
    
    firstRow = fidlFile.readline().split()
    tokens = firstRow
    tokens.reverse()
    TR = float(tokens.pop())
    
    lastElem = tokens[0]
    
    runs = int(lastElem[-1])
    #runs = 12
    noEvents = len(tokens)/int(runs)
    
    eventLabels = []
    
    tokens.reverse()
    for i in range(noEvents):
        eventLabels.append(tokens[i][:-1])#eventLabels.append(tokens[i][:-1])
        
    
    fidlFile.close()
    data = np.loadtxt(fidlPath, skiprows = 1)
    
    onset = data.T[0]
    duration = data.T[2]
    events = data.T[1]
    
    runsList = np.floor(events/int(noEvents))
    
    targets= []
    chunks = []
    
    outFile = open(outPath, 'w')
    
    for i in range(len(runsList)):
        volumes = np.rint(duration[i]/TR)
        
        for j in range(int(volumes)):
            index = events[i] - runsList[i]*noEvents
            
            outFile.write(str(eventLabels[int(index)])+' '+str(int(runsList[i]))+'\n')
            targets.append(eventLabels[int(index)])
            chunks.append(int(runsList[i]))
            
    
    outFile.close()
            
            
            
def fidl2txt_2(fidlPath, outPath, runs=12., vol_run=248, stim_tr=4, offset_tr=0):
    '''
    Fidl file event to text utility.
    
    Parameters:
    ----------------------------------
    fidlPath: path to fidl file.
    outPath: output pathname
    runs: number of runs 
    vol_run: number of volumes per run
    stim_tr: number of volumes to consider a stimulus in the trial
    offset_tr: number of volumes to eliminate at event/stimulus beginning
    '''
    print('Converting fidl file '+fidlPath+' in '+outPath)
    
    fname, ext = fidlPath.split('.')
    
    if ext == 'csv':
        delimiter = ','
    else:
        delimiter = '\t'
    
    exp_end = vol_run * runs
    
    fidlFile = open(fidlPath, 'r')
        
    line_ = fidlFile.readline().split(delimiter)
    line_ = [l.split(' ') for l in line_ if l!='' and l!='\r\n' and l!='\n']
    line_ = list(itertools.chain(*line_))
    line_ = [l.split('\r')[0] for l in line_]
    line_ = [l.split('\n')[0] for l in line_]

    tokens = line_
    #tokens.reverse()
    TR = float(tokens[0])
    tokens = tokens[1:]
       
    fidlFile.close()
    data = np.loadtxt(fidlPath, 
                      skiprows = 1, 
                      delimiter=delimiter, 
                      dtype=np.str_)
    
    try:
        _ = np.float_(data[0])
    except ValueError as _:
        data = data[1:]
    
    onset = np.float_(data.T[0])
    #duration = data.T[2]
    #
    events = np.int_(data.T[1])
    
    outFile = open(outPath, 'w')
    
    eventLabels = []
    
    noEvents = len(tokens)
    
    #tokens.reverse()
    for i in range(noEvents):
        eventLabels.append(tokens[i][:])
    
    onset = np.append(onset, exp_end * TR)
    duration = [onset[i+1] - onset[i] for i in range(len(onset)-1)]
    duration = np.array(duration)
    
    if onset[0] != 0:
        f = 0
        while f < np.rint(onset[0]/TR):
            outFile.write(u'FIX 0 0 0 0\n')
            f = f + 1
    
    chunk_size = int(len(line_)/runs)
    r = 1
    sum_ = 0
    c_ = 0
    for i in range(len(onset)-1):

        #j = 0
        
        duration_volumes = np.rint(onset[i+1]/TR) - np.rint(onset[i]/TR)
        sum_ += duration_volumes   

        for j in range(int(duration_volumes)):
            
            #if (j < np.rint(duration[i]/TR)):#-1
            if (offset_tr < (j+1) <= offset_tr + stim_tr):#-1
                #outFile.write(eventLabels[int(events[i])]+' '+str(runArr[int(events[i])])+'\n')
                outFile.write(eventLabels[int(events[i])]+' '
                                            +str(int(i/chunk_size))+' '  #Chunk
                                            +str(int(r/vol_run))+' '     #Run
                                            +str(i)+' '                  #Event
                                            +str(j+1)+'\n')              #Frame
            else:
                #outFile.write(u'fixation '+str(runArr[int(events[i])])+'\n')
                outFile.write(u'FIX '
                              +str(int(i/chunk_size))+' '
                              +str(int(r/vol_run))+' '     #Run
                              +str(i)+' '
                              +str(j+1)+'\n')
            #j = j + 1
            r = r + 1

            c_ += 1
        print(i)
        print()
        print(c_ - sum_)
        
        if r == vol_run:
            r = 0

    print(sum_)
    print(c_)
    outFile.close()   



def extract_events_xls(filename, sheet_name):
    import xlrd
    
    book = xlrd.open_workbook(filename)
    try:
        sh = book.sheet_by_name(sheet_name)
    except XLRDError as e:
        raise XLRDError(e)
    
    onset = sh.col(0)
    onset = np.array([s.value for s in onset[1:]])
    
    events_num = sh.col_values(1)
    events_num = np.array(events_num[1:])
    
    logging.debug(events_num)
    
    duration = sh.col_values(2)
    duration = np.array(duration[1:])
    
    logging.debug(duration)
    
    duration = [onset[i+1] - onset[i] for i in range(len(onset)-1)]
    
    event_labels = sh.col_values(5)
    event_labels = np.array([e for e in event_labels if e != ''])
    
    logging.debug(event_labels)
    
    return onset, duration, event_labels, events_num


def build_attributes(out_path,
                     onset, 
                     duration, 
                     TR, 
                     events,
                     event_labels,
                     runs=12, vol_run=250, stim_vol=4, offset_tr=2):
    
    f = 0
    outFile = open(out_path, 'w')
    if onset[0] != 0:
        
        while f < np.rint(onset[0]/TR):
            outFile.write(u'FIX 0 1 0 0\n')
            f = f + 1
     
    onset = np.append(onset, vol_run * runs * TR)
    n_events_run = np.int(len(duration)/runs)
    
    r = 1
    
    for i in range(len(onset)-1):
        '''
        if i <= 1:
            runArr = np.array(np.ceil(np.bincount(np.int_(events[:2]))/4.) - 1, dtype=np.int)
        else:
            runArr = np.array(np.ceil(np.bincount(np.int_(events[:i+1]), 
                                              minlength=noEvents)/4.) - 1, 
                          dtype=np.int)
        '''
        j = 0
        
        if r == vol_run:
            r = 0
        else:
            r+=1
        
        while j < np.rint(onset[i+1]/TR) - np.rint(onset[i]/TR):
            
            #if (j < np.rint(duration[i]/TR)):#-1
            if (offset_tr <= j < offset_tr + stim_vol):#-1
                #outFile.write(eventLabels[int(events[i])]+' '+str(runArr[int(events[i])])+'\n')
                index = int(events[i])
                
                # Condition, Chunk, File, Event, Frame
                line = "%s %s %s %s %s\n" % (event_labels[index], str(i / n_events_run), str(r), str(i), str(j+1))
                #line = event_labels[index]+' '+str(i / n_events_run)+' '+str(i)+' '+str(j+1)+'\n'
            else:
                #outFile.write(u'fixation '+str(runArr[int(events[i])])+'\n')
                line = "%s %s %s %s %s\n" % ("FIX", str(i/n_events_run), str(r), str(i), str(j+1))
                #line = u'FIX '+str(i/n_events_run)+' '+str(i)+' '+str(j+1)+'\n'
            outFile.write(line)
            j = j + 1
            
    outFile.close()
    
    return


def beta_attributes(fidl_file, output_filename, nrun=12.):
    """
    Build attributes for betas from fidl event
    """
    
    file_ = open(fidl_file, 'r')
    line_ = file_.readline()
    line_ = line_.split('\t')[1:]
    line_ = [l.split(' ') for l in line_ if l != '' and l != '\r\n']
    line_ = list(itertools.chain(*line_))
    line_ = [l.split('\r')[0] for l in line_]
    line_ = ['_'.join(l.split('_')[:-1]) for l in line_]
    
    chunk_size = len(line_)/nrun
    chunks = [np.int(i/chunk_size) for i in range(len(line_))]
    
    attr_ = np.vstack((line_, chunks)).T
    np.savetxt(output_filename, attr_, fmt='%s')
    file_.close()
    

def enhance_attributes_ofp(filename, current_header=['targets', 'run', 'chunks', 'trial', 'frame']):
    
    """Only for Carlo's ofp experiment"""
    import re
    attr = np.loadtxt(filename, dtype=np.str_)
    
    assert attr.shape[1] == len(current_header)
    
    # decision
    pattern_face  = ['F.C', 'L.I']
    pattern_place = ['F.I', 'L.C']
    
    def is_in(pattern_list, string_):
        return np.array([re.search(pt, string_) != None for pt in pattern_list]).any()
    
    mask_face  = np.array([is_in(pattern_face, a) for a in attr.T[0]])
    mask_place = np.array([is_in(pattern_place, a) for a in attr.T[0]])
    
    
    decision = attr.T[0].copy()
    decision[mask_face] = 'F'
    decision[mask_place] = 'L'
    decision[np.logical_not(np.logical_or(mask_face, mask_place))] = 'X'
    
    
    # memory_status
    
    mask_face  = np.array([a[0] == 'F' and a[1] != 'I' for a in attr.T[0]])
    mask_place = np.array([a[0] == 'L' for a in attr.T[0]])
    
    memory_status = attr.T[0].copy()
    memory_status[mask_face] = 'F'
    memory_status[mask_place] = 'L'    
    memory_status[np.logical_not(np.logical_or(mask_face, mask_place))] = 'X'
    
    stim = []
    
    for a_ in attr.T[0]:
        
        if a_.find('R') != -1:
            stim.append(['N', '0', '0'])
        elif a_.find('X') != -1:
            stim.append(['X', '0', '0'])
        else:
            stim.append(list(a_)[:3]) # Used if attributes are more than 3

    
    stim = np.array(stim)
    
    attr_ = np.hstack((attr, stim, memory_status[:,np.newaxis], decision[:,np.newaxis]))
    current_header = current_header+['stim', 'evidence', 'accuracy', 'memory_status', 'decision']
    
    assert len(current_header) == attr_.shape[1]
    
    attr_ = np.vstack((current_header, attr_))
    
    filename_pts = filename.split('/')
    filename_ = filename_pts.pop()
    filename_new = filename_.split('.')[0]+'_plus.txt'
    filename_pts.append(filename_new)
    
    fname = '/'.join(filename_pts)

    np.savetxt(fname, attr_, fmt='%s')


def enhance_attributes_memory(filename, current_header=['targets', 'chunks', 'run', 'trial', 'frame']):
    
    """Only for Carlo's memory experiment"""
    import re
    attr = np.loadtxt(filename, dtype=np.str_)
    
    assert attr.shape[1] == len(current_header)
    
    pattern_old = ['O_._1', 'N_._0']
    pattern_new = ['O_._0', 'N_._1']
    
    def is_in(pattern_list, string_):
        return np.array([re.search(pt, string_)!=None for pt in pattern_list]).any()
    
    mask_old = np.array([is_in(pattern_old, a) for a in attr.T[0]])
    mask_new = np.array([is_in(pattern_new, a) for a in attr.T[0]])
    
    decision = attr.T[0].copy()
    decision[mask_new] = 'NEW'
    decision[mask_old] = 'OLD'
    
    
    pattern_old = ['O_._1', 'N_._0']
    pattern_new = ['O_._0', 'N_._1']
    
    
    mask_old = np.array([a[0] == 'O' for a in attr.T[0]])
    mask_new = np.array([a[0] == 'N' for a in attr.T[0]])
    
    memory_status = attr.T[0].copy()
    memory_status[mask_new] = 'NEW'
    memory_status[mask_old] = 'OLD'    
    
    stim = []
    
    for a_ in attr.T[0]:
        if a_ != 'FIX':
            stim.append(a_.split('_')[:3]) # Used if attributes are more than 3
        else:
            stim.append(['F', '0', '0'])
    
    stim = np.array(stim)
    
    attr_ = np.hstack((attr, stim, memory_status[:,np.newaxis], decision[:,np.newaxis]))
    current_header = current_header+['stim', 'evidence', 'accuracy', 'memory_status', 'decision']
    
    assert len(current_header) == attr_.shape[1]
    
    attr_ = np.vstack((current_header, attr_))
    
    filename_pts = filename.split('/')
    filename_ = filename_pts.pop()
    filename_new = filename_.split('.')[0]+'_plus.txt'
    filename_pts.append(filename_new)
    
    fname = '/'.join(filename_pts)

    np.savetxt(fname, attr_, fmt='%s')
    
    

def load_results(path, name, task):

    folder = '0_results'
    
    print('Opening ' + os.path.join(path, folder, name+'_'+task+'_120618_map.hdf5'))
    map = h5load(os.path.join(path, folder, name+'_'+task+'_120618_map.hdf5'))
    
    mapper = pickle.load(open(os.path.join(path, folder, name+'_'+task+'_120618_mapper.pyobj'), 'r'))
    
    rev_map = mapper.reverse(map.samples)
    
    
    
    fileName = [elem for elem in os.listdir(os.path.join(path, name,'rest')) if elem.find('.nii.gz') != -1][0]
    
    niftiimg = ni.load(os.path.join(path, name,'rest',fileName))
    
    
    ni.save(ni.Nifti1Image(rev_map.squeeze(), niftiimg.get_affine()), os.path.join(path, name,name+'_120618_nifti_map.nii.gz'))
      
    imgIn = os.path.join(path, name,name+'_120618_nifti_map.nii.gz')
    refIn = '/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm_brain.nii.gz'
    
    
    mat = [elem for elem in os.listdir(os.path.join(path, name, 'rest')) if elem.find('.mat') != -1 and elem.find('mni') != -1][0]
    
    command = 'flirt '+ \
              ' -in '+imgIn+ \
              ' -ref '+refIn+ \
              ' -init '+ os.path.join(path, name, 'rest',mat) +\
              ' -applyxfm' + \
              ' -out '+ os.path.join(path, name,name+'_nifti_map.nii.gz')[:-7] +'_120618_mni.nii.gz' 
                  
    print(command)
    os.system(command)
    
    results = pickle.load(open(os.path.join(path, folder, name+'_'+task+'_120618_res.pyobj'), 'r'))

    print('**************** '+name+' **********************')
    print(results.stats)
    
    
    
    mni_img = ni.load(os.path.join(path, name,name+'_nifti_map.nii.gz')[:-7] +'_120618_mni.nii.gz')
    mni_mask =  ni.load('/usr/share/data/fsl-mni152-templates/MNI152lin_T1_2mm_brain_mask.nii.gz')
    brain = ni.load('/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm.nii.gz')
                         
    res_masked = mni_img.get_data() * mni_mask.get_data()
    
    res_masked = (res_masked - np.mean(res_masked))/np.std(res_masked)
    
    ni.save(ni.Nifti1Image(res_masked, mni_mask.get_affine(), header = mni_mask.get_header()), 
            os.path.join(path, name,name+'_nifti_map.nii.gz')[:-7] +'_120618_res_masked_norm_mni.nii.gz')
    
    ni.save(ni.Nifti1Image(brain.get_data(), mni_mask.get_affine(), header = mni_mask.get_header()), 
            os.path.join(path,'_MNI152_T1_2mm.nii.gz'))
    
    return [results, res_masked]


def load_results_stats (path, namelist, datetime, task, mask = 'none'):
    """
        @param datetime: Indicates the moment of the analysis in a string formatted as AAAAMMGG_HHMM,
        could be used only a portion of that
        @param mask: Optional param, is only setted when we want to store the result of a particular brain mask used during analysis
        
        @return: A list of dictionaries with the info about the type of analysis and its results.
                To access the object type list[n_record]['info'] to get the info and list[n_record]['stats'] to get the results:
                where n_record is the integer index of a particular record.
        
    
    """
    resFolder = '0_results'
    
    dict_keys = ['day', 'hour', 'name', 'analysis' 'task', 'mask']
    res_keys = ['info', 'stats']
    
    
    if (mask == 'none'):
        mask = ''    
    
    fileList = os.listdir(os.path.join(path, resFolder))
    fileList = [f for f in fileList if f.find(task) != -1 and f.find(datetime) != -1 and f.find('res') != -1and f.find(mask)!= -1 ]                     

    res = []
    
    for e in fileList:
        d = dict(zip(dict_keys, e.split('_')))
        m = pickle.load(open(os.path.join(path, resFolder, e), 'r'))

        res.append(dict(zip(res_keys, [d, m])))

    return res
        
def read_stats(formatted_results):
    
    for el in formatted_results:
        for info in el['info']:
            print(info)
    
    
    

        
def load_results_map (path, namelist, datetime, task, mask='none'):
    """
        @param datetime: Indicates the moment of the analysis in a string formatted as AAAAMMGG_HHMM, could be used only a portion of that
        @param mask: Optional param, is only setted when we want to store the result of a particular brain mask used during analysis
    """    
    
    resFolder = '0_results'
    
    fileList = os.listdir(os.path.join(path, resFolder))
    
    if (mask == 'none'):
        mask = ''
    
    fileList = [f for f in fileList if f.find(task)     != -1 
                                   and f.find(datetime) != -1 
                                   and f.find('map')    != -1 
                                   and f.find(mask)     != -1
                                   ]
    fileList.sort()
    
    if (len(fileList) == 0):
        print('Results not found!')
        return
    
    fileName = [elem for elem in os.listdir(os.path.join(path, 'andant','rest')) if elem.find('.nii.gz') != -1][0]
    niftiimg = ni.load(os.path.join(path, 'andant', 'rest', fileName))
    affine_tr = niftiimg.get_affine()
    
    for i in range(0, len(fileList), 2):
        
        mapFile = fileList[i]
        mapperFile = fileList[i+1]
        
        parts = mapFile.split('_')
        name = parts[2]     
        
        print('Opening: ' + os.path.join(path, resFolder, mapFile))
        map = h5load(os.path.join(path, resFolder, mapFile))
        
        print('Opening: ' + os.path.join(path, resFolder, mapperFile))
        mapper = pickle.load(open(os.path.join(path, resFolder, mapperFile), 'r'))
    
        rev_map = mapper.reverse(map.samples)
        
        fileName = [elem for elem in os.listdir(os.path.join(path, name,'rest')) if elem.find('.nii.gz') != -1][0]
    
        niftiimg = ni.load(os.path.join(path, name,'rest',fileName))
    
        print('Saving results at: ' + os.path.join(path, name, mapFile.split('.')[0] + '.nii.gz'))
        ni.save(ni.Nifti1Image(rev_map.squeeze(), affine_tr), os.path.join(path, name, mapFile.split('.')[0] + '.nii.gz'))
        

def remove_flirt_files (path, type='both'):
    
    
    folders = os.listdir(path)
    folders = [elem for elem in folders if (elem.find('.') == -1) and (elem.find('_') == -1)]
    
    
    for folder in folders:
        
        fileListR = os.listdir(os.path.join(path,folder,'rest'))
        fileListT = os.listdir(os.path.join(path,folder,'task'))
        
        
        if type == 'all':
            fileListR = [elem for elem in fileListR if (elem.find('.nii.gz') != -1) or (elem.find('.mat') != -1)]
            fileListT = [elem for elem in fileListT if (elem.find('.nii.gz') != -1) or (elem.find('.mat') != -1)]
            
        if type == 'both':
            
            fileListR = [elem for elem in fileListR if (elem.find('flirt') != -1)]
            fileListT = [elem for elem in fileListT if (elem.find('flirt') != -1)]
     
        if type == 'mni':
                
                fileListR = [elem for elem in fileListR if (elem.find('mni') != -1)]
                fileListT = [elem for elem in fileListT if (elem.find('mni') != -1)]
                
        for fileR in fileListR:
            print('Deleting '+os.path.join(path,folder,'rest',fileR))
            os.remove(os.path.join(path,folder,'rest',fileR))
            
        for fileT in fileListT:
            print('Deleting '+os.path.join(path,folder,'task',fileT))
            os.remove(os.path.join(path,folder,'task',fileT))


def searchlight_to_mni(path, subj):
      
    searchFileList = os.listdir(os.path.join(path, subj))
    searchFileList = [elem for elem in searchFileList if elem.find('searchlight') != -1]
    
    brainFileList = os.listdir(os.path.join(path, subj, 'rest'))
    brainFileList = [elem for elem in brainFileList if elem.find('_crop_brain_flirt.nii.gz') != -1]

    imgIn = os.path.join(path, subj, 'rest',brainFileList[0])
    refIn = '/usr/share/fsl/4.1/data/standard/MNI152_T1_2mm_brain.nii.gz'
    
    command =   'flirt '+ \
                 ' -in ' + imgIn + \
                 ' -ref '+refIn+ \
                 ' -omat '+ os.path.join(path, subj, subj+'_matrix_to_mni.mat') + \
                 ' -dof 12' + \
                 ' -searchcost normmi' + \
                 ' -searchrx -180 180' + \
                 ' -searchry -180 180' + \
                 ' -searchrz -180 180'
        
    print(command)
    os.system(command)
    
    matrix = os.path.join(path, subj, subj+'_matrix_to_mni.mat')
    
    for file_ in searchFileList:
        
        imgIn = os.path.join(path, subj, file_)
      
        command = 'flirt '+ \
                  ' -in '+ imgIn + \
                  ' -ref '+ refIn + \
                  ' -init '+ matrix + \
                  ' -applyxfm' + \
                  ' -out '+ imgIn[:-7] +'_mni.nii.gz' 
                  
        print(command)
        os.system(command)
