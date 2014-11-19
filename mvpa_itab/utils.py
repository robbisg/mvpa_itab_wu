#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################


import os
import datetime as ora
import nibabel as ni
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm



from mvpa2.suite import h5load

'''
def writeConfiguration(name, experiment, zscoring = None, 
                                            detrending = None,
                                            classifier = None, 
                                            featsel = None, 
                                            modelSel = None):
    fileRes = open 
    
    
    nodeD = detrendeding.a.mapper.nodes[len(detrended_ds.a.mapper.nodes)-1]
    
    nodeZ = detrended_ds.a.mapper.nodes[len(detrended_ds.a.mapper.nodes)-1]
    experimentInfo.append(str(nodeZ) + str(nodeZ.param_est))
    
    spcl = get_samples_per_chunk_target(fds)
    experimentInfo.append(zip(fds.sa['targets'].unique, spcl[0]))
    
    summary = clf.summary()
    clfName = summary[summary.find('<')+1:summary.find('>')]
    experimentInfo.append(clfName)
    
    nodeFSL = fsel._SensitivityBasedFeatureSelection__feature_selector
    nodeSENSA = fsel._SensitivityBasedFeatureSelection__sensitivity_analyzer
    experimentInfo.append(str(nodeFSL)+str(nodeSENSA))
    
    experimentInfo.append(str(cvte.generator))
    
'''

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
        print 'Making new directory...'
        os.mkdir(path+time+'_'+experiment+'_'+clf)
    else:
        print 'Directory found...'
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
    
    resultSummaryFile = open (pathToNewDir+'0__classifier_results.txt','w')
       
       
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
    
    
    print 'Converting fidl file '+fidlPath+' in '+outPath
    
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
            

def fidl2txt_2(fidlPath, outPath, runs=12, vol_run=248, stim_tr=4, offset_tr=2):
    '''
    exp_end = ???
    '''
    print 'Converting fidl file '+fidlPath+' in '+outPath
    
    exp_end = vol_run * runs
    
    fidlFile = open(fidlPath)
    
    firstRow = fidlFile.readline().split(',')
    tokens = firstRow
    tokens.reverse()
    TR = float(tokens.pop())
       
    fidlFile.close()
    data = np.loadtxt(fidlPath, skiprows = 1, delimiter=',')
    
    onset = data.T[0]
    #duration = data.T[2]
    #
    events = data.T[1]
    
    outFile = open(outPath, 'w')
    
    eventLabels = []
    
    noEvents = len(tokens)
    
    tokens.reverse()
    for i in range(noEvents):
        eventLabels.append(tokens[i][:])
    
    onset = np.append(onset, exp_end * TR)
    duration = [onset[i+1] - onset[i] for i in range(len(onset)-1)]
    duration = np.array(duration)
    
    if onset[0] != 0:
        f = 0
        while f < np.rint(onset[0]/TR):
            outFile.write(u'FIX 0 0 0\n')
            f = f + 1
     
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

        while j < np.rint(onset[i+1]/TR) - np.rint(onset[i]/TR):
            
            #if (j < np.rint(duration[i]/TR)):#-1
            if (offset_tr < j < offset_tr + stim_tr):#-1
                #outFile.write(eventLabels[int(events[i])]+' '+str(runArr[int(events[i])])+'\n')
                outFile.write(eventLabels[int(events[i])]+' '
                                            +str(i/30)+' '  #Chunk
                                            +str(i)+' '     #Event
                                            +str(j+1)+'\n') #Frame
            else:
                #outFile.write(u'fixation '+str(runArr[int(events[i])])+'\n')
                outFile.write(u'FIX '+str(i/30)+' '+str(i)+' '+str(j+1)+'\n')
            j = j + 1
            
    outFile.close()   


def roi_wu_data(path, name, task, init_vol=0):
    
    imgFiles = os.listdir(path+'/'+name+'/'+task) 
    imgFiles = [elem for elem in imgFiles if (elem.find('mni') == -1) & (elem.find('.img') != -1) & (elem.find('.rec') == -1)]
    
    for img in imgFiles:
        
        n_vols = ni.analyze.load(os.path.join(path, name, task, img)).get_shape()[3]
        
        roi = 'fslroi '+ \
            os.path.join(path, name, task, img) + ' ' + \
            os.path.join(path, name, task, img[:-4] + '_crop') + ' ' +\
            str(init_vol) + ' ' + str(n_vols - init_vol)
        print roi
        os.system(roi)
    
    
def bet_wu_data_(path, name, task):
    
    task = ''
    print '--------- BRAIN EXTRACTION ----------'
    
    
    imgFiles = os.listdir(path+'/'+name+'/'+task) 
    #imgFiles = [elem for elem in imgFiles if (elem.find('mni') == -1) & (elem.find('crop.') != -1) & (elem.find('.rec') == -1)]
    imgFiles = [elem for elem in imgFiles if (elem.find('hdr') != -1)]
   
    for img in imgFiles:
        
        bet = 'bet '+ \
            os.path.join(path, name, task, img) + ' ' + \
            os.path.join(path, name, task, img[:-4] + '_brain') + ' ' +\
            '-F -f 0.60 -g 0'
        print bet
        os.system(bet)


def mc_wu_data(path, name, task):
     
    # nameL = os.listdir('/media/DATA/fmri/learning/')
    # nameL = [elem for elem in nameL if (elem.find('_') == -1) & (elem.find('.') == -1)]
    print '      ---- > Motion Correction <-----   '
    
    
    
    imgFiles = os.listdir(path+'/'+name+'/'+task) 
    imgFiles = [elem for elem in imgFiles if (elem.find('mni') == -1) & (elem.find('brain.') != -1) & (elem.find('.rec') == -1)]
    
    imgFiles.sort()
    
    restList = os.listdir(path+'/'+name+'/rest')
    restList = [elem for elem in restList if (elem.find('.img_') == -1) & (elem.find('rest1') != -1) & (elem.find('brain.') != -1) \
                                                                    &   (elem.find('.rec') == -1)]
    ref = restList.pop()
    
    #ref = imgFiles[len(imgFiles)-1    
    
    i = 0
    for img in imgFiles:
        i = i + 1
        imgIn = path+'/'+name+'/'+task+'/'+img
        refIn = path+'/'+name+'/rest/'+ref
        '''
        flirtMC = pe.Node(
                            interface = fsl.FLIRT(
                                                 in_file = imgIn,
                                                 reference = refIn, 
                                                 dof = 6
                                                 ),
                            name = 'flirtMC_'+str(i)
                            )
        flirtMC.inputs.out_matrix_file = path+'/'+name+'/'+task+'/'+img+'.mat'
        
        #flirtMC.base_dir = (path+'/'+name+'/'+task)
        flirtMC.run()  
        '''
        command =   'flirt '+ \
                    ' -in ' + imgIn + \
                    ' -ref '+refIn+ \
                    ' -searchcost normmi' +\
                    ' -omat '+path+'/'+name+'/'+task+'/'+img[:-7] +'_flirt.mat'+ \
                    ' -dof 12'
                  
                  
        
        print command
        os.system(command)
       
    i = 0
    for img in imgFiles:  
           
        i = i + 1
        imgIn = path+'/'+name+'/'+task+'/'+img
        refIn = path+'/'+name+'/rest/'+ref
        ''' 
        flirt8 =  pe.Node(
                            interface = fsl.ApplyXfm(
                                                 in_file = imgIn,
                                                 reference = refIn, 
                                                 out_file = path+'/'+name+'/'+task+'/'+'flirt_'+img+'.nii.gz',
                                                # out_matrix_file = path+'/'+name+'/'+task+'/'+img+'_flirt2.mat',
                                                 apply_xfm = True,
                                                 in_matrix_file = path+'/'+name+'/'+task+'/'+img +'.mat'
                                                 ),
                            name = 'flirt4D_'+str(i)
                            )
        
        #flirt8.base_dir = (path+'/'+name+'/'+task)
        flirt8.run()

        '''
        command = 'flirt '+ \
                  ' -in '+imgIn+ \
                  ' -ref '+refIn+ \
                  ' -init '+path+'/'+name+'/'+task+'/'+img[:-7] +'_flirt.mat'+ \
                  ' -applyxfm -interp nearestneighbour' + \
                  ' -out '+ path+'/'+name+'/'+task+'/'+img[:-7] +'_flirt.nii.gz' 
                  
        print command         
        os.system(command)



def wu_to_mni (path, name, task): 
    print '      ---- > MNI Coregistration <-----   '
    
    
    imgFiles = os.listdir( os.path.join(path,name,task) )
    imgFiles = [elem for elem in imgFiles if (elem.find('flirt.nii.gz') != -1) and (elem.find('mni') == -1)]
    
    imgFiles.sort()
    
    
   
    #ref = imgFiles[len(imgFiles)-1]
    
    i=0
    
    for img in imgFiles:
        i = i + 1
        imgIn = os.path.join(path,name,task,img)
        refIn = '/media/DATA/fmri/MNI152_T1_3mm_brain.nii.gz'
        '''
        flirtMC = pe.Node(
                            interface = fsl.FLIRT(
                                                 in_file = imgIn,
                                                 reference = refIn, 
                                                 dof = 6
                                                 ),
                            name = 'flirtMC_'+str(i)
                            )
        flirtMC.inputs.out_matrix_file = path+'/'+name+'/'+task+'/'+img+'.mat'
        
        #flirtMC.base_dir = (path+'/'+name+'/'+task)
        flirtMC.run()  
        '''
        command =   'flirt '+ \
                    ' -in ' + imgIn + \
                    ' -ref '+refIn+ \
                    ' -omat '+  os.path.join(path,name,task,img[:-7]) +'_mni.mat'+ \
                    ' -dof 12' + \
                    ' -searchcost normmi' + \
                    ' -searchrx -90 90' + \
                    ' -searchry -90 90' + \
                    ' -searchrz -90 90'
        
        print command
        os.system(command)
       
    i = 0
    for img in imgFiles:  
           
        i = i + 1
        imgIn = os.path.join(path,name,task,img)

        #refIn = path+'/'+name+'/rest/'+ref
        ''' 
        flirt8 =  pe.Node(
                            interface = fsl.ApplyXfm(
                                                 in_file = imgIn,
                                                 reference = refIn, 
                                                 out_file = path+'/'+name+'/'+task+'/'+'flirt_'+img+'.nii.gz',
                                                # out_matrix_file = path+'/'+name+'/'+task+'/'+img+'_flirt2.mat',
                                                 apply_xfm = True,
                                                 in_matrix_file = path+'/'+name+'/'+task+'/'+img +'.mat'
                                                 ),
                            name = 'flirt4D_'+str(i)
                            )
        
        #flirt8.base_dir = (path+'/'+name+'/'+task)
        flirt8.run()

        '''
        command = 'flirt '+ \
                  ' -in '+imgIn+ \
                  ' -ref '+refIn+ \
                  ' -init '+os.path.join(path,name,task,img[:-7]) +'_mni.mat'+ \
                  ' -applyxfm' + \
                  ' -out '+ os.path.join(path,name,task,img[:-7]) +'_mni.nii.gz' 
                  
        print command         
        os.system(command)

def watch_results (path, task):
    resFolder = '0_results'
    
    fileList = os.listdir(os.path.join(path, resFolder))
    fileList = [f for f in fileList if f.find(task) != -1]
    
    return fileList

def load_results(path, name, task):

    folder = '0_results'
    
    print 'Opening ' + os.path.join(path, folder, name+'_'+task+'_120618_map.hdf5')
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
                  
    print command         
    os.system(command)
    
    results = pickle.load(open(os.path.join(path, folder, name+'_'+task+'_120618_res.pyobj'), 'r'))

    print '**************** '+name+' **********************'
    print results.stats
    
    
    
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
            print info
    
    
    

        
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
        print 'Results not found!'
        return;
    
    fileName = [elem for elem in os.listdir(os.path.join(path, 'andant','rest')) if elem.find('.nii.gz') != -1][0]
    niftiimg = ni.load(os.path.join(path, 'andant', 'rest', fileName))
    affine_tr = niftiimg.get_affine()
    
    for i in range(0, len(fileList), 2):
        
        mapFile = fileList[i]
        mapperFile = fileList[i+1]
        
        parts = mapFile.split('_')
        name = parts[2]     
        
        print 'Opening: ' + os.path.join(path, resFolder, mapFile)
        map = h5load(os.path.join(path, resFolder, mapFile))
        
        print 'Opening: ' + os.path.join(path, resFolder, mapperFile)
        mapper = pickle.load(open(os.path.join(path, resFolder, mapperFile), 'r'))
    
        rev_map = mapper.reverse(map.samples)
        
        fileName = [elem for elem in os.listdir(os.path.join(path, name,'rest')) if elem.find('.nii.gz') != -1][0]
    
        niftiimg = ni.load(os.path.join(path, name,'rest',fileName))
    
        print 'Saving results at: ' + os.path.join(path, name, mapFile.split('.')[0] + '.nii.gz')
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
            print 'Deleting '+os.path.join(path,folder,'rest',fileR)
            os.remove(os.path.join(path,folder,'rest',fileR))
            
        for fileT in fileListT:
            print 'Deleting '+os.path.join(path,folder,'task',fileT)
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
        
    print command
    os.system(command)
    
    matrix = os.path.join(path, subj, subj+'_matrix_to_mni.mat')
    
    for file in searchFileList:
        
        imgIn = os.path.join(path, subj, file)
      
        command = 'flirt '+ \
                  ' -in '+ imgIn + \
                  ' -ref '+ refIn + \
                  ' -init '+ matrix + \
                  ' -applyxfm' + \
                  ' -out '+ imgIn[:-7] +'_mni.nii.gz' 
                  
        print command         
        os.system(command)

def mask_searchlight(path, input, min):
    
    sl_map = ni.load(os.path.join(path, input))
    
    
def map_to_mni(path, matrix):
    
    
    return
 
'''
def plot_matrix(matrix, labels, name):
    
    i = 0
    for elem in resList:
        fig = plt.figure(1)
        ax = fig.add_subplot(3,4,i)
        factor = 1
        
        if cmp(list[i], 'anndel') == 0:
            factor = 1500./1000.
        if cmp(list[i], 'augpel') == 0:
            factor = 1500./1250.
        
        cax = ax.imshow(factor * elem.stats.matrix, interpolation='nearest', vmin = min, vmax = max)
        ax.set_title(list[i])
        
        #axu = ax.twiny()
        ax.set_xticklabels(['','RestPost', '','RestPre'])
        ax.set_xlabel('Targets')       
        
        ax.set_yticklabels(['','RestPost', '','RestPre'])
        ax.set_ylabel('Predictions')

        ax.text(0,0, str(factor * elem.stats.matrix[0][0]), fontsize = 18, horizontalalignment='center')#, fontweight='bold')
        ax.text(1,0, str(factor * elem.stats.matrix[1][0]), fontsize = 18, horizontalalignment='center')#, fontweight='bold')
        ax.text(0,1, str(factor * elem.stats.matrix[0][1]), fontsize = 18, horizontalalignment='center')#, fontweight='bold')
        ax.text(1,1, str(factor * elem.stats.matrix[1][1]), fontsize = 18, horizontalalignment='center')#, fontweight='bold')
        
        #cbar = fig.colorbar(cax)
        #fig.savefig(os.path.join(path, '0_res_pictures',list[i]+'_conf_mat.png'))
        i = i + 1
    
    
    
    i = 0
    mat = np.zeros(resList[0].stats.matrix.shape)   
    
    for elem in resList:
        factor = 1
        if cmp(list[i], 'anndel') == 0:
            factor = 1500./1000.
        if cmp(list[i], 'augpel') == 0:
            factor = 1500./1250.
        mat = mat + factor * elem.stats.matrix
        i = i + 1
        
        
    X = matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(X, cmap=cm.jet, interpolation='nearest')

    numrows, numcols = X.shape

    #fig.add_axes(['Label1', 'Label2'])
    
    ax.set_title('Confusion Matrix for '+name)
    
    #plt.xlabel('Predictions')
    #plt.ylabel('Targets')
    #plt.title('Confusion Matrix for '+name)
    #plt.axis(['Label1', 'Label2','Label1', 'Label2'])
    plt.show()
    ----------------------------------------------------------------------
    
    for subj in nameL:
        searchFileList = os.listdir(os.path.join(path, subj))
        searchFileList = [elem for elem in searchFileList if elem.find('searchlight') != -1 and elem.find('mni')==-1]
        for img in searchFileList:
            ni_img = ni.load(os.path.join(path, subj, img))
            mask = ni_img.get_data() != 0
            mean = np.mean(ni_img.get_data()[mask])
            std = np.std(ni_img.get_data()[mask])
            numVx = np.sum(mask)
            plt.hist()
            print img[:-7]+' m: '+str(mean)+' s:'+str(std)
            
    ---------------------------------------------------------------------------
    
    for img in imgFiles:  
           
        i = i + 1
        imgIn = os.path.join(path,name, 'task,img)

        refIn = path+'/'+name+'/rest/'+ref
       
        command = 'flirt '+ \
                  ' -in '+imgIn+ \
                  ' -ref '+refIn+ \
                  ' -init '+os.path.join(path,'andant','task',ref) + \
                  ' -applyxfm' + \
                  ' -out '+ os.path.join(path,'1_ROI_VC','MNI',img[:-7]) +'_mni.nii.gz' 
                  
        print command         
        os.system(command)
        ---------------------------------------------------
        
        for roi in roilist:
            command = 'flirt ' + \
              ' -in /media/DATA/fmri/ROI_MNI/' + roi + \
              ' -applyxfm -init /media/DATA/fmri/mni_to_wu.mat ' +\
              ' -out /media/DATA/fmri/' + roi[:-7] +'wu.nii.gz'+\
              ' -paddingsize 0.0 -interp nearestneighbour' +\
              ' -ref /media/DATA/fmri/MNI_to_func_3mm.nii.gz'
            print command
            os.system(command)

        
'''    