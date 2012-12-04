import nibabel as ni
import os
from main_wu import *
from utils import *
import matplotlib.pyplot as plt
from mvpa2.suite import find_events, fmri_dataset, SampleAttributes
from itertools import cycle

def get_time():
        #Time acquisition for file name!
    tempo = time.localtime()
    
    datetime = ''
    i = 0
    for elem in tempo[:-3]:
        i = i + 1
        if len(str(elem)) < 2:
            elem = '0'+str(elem)
        if i == 4:
            datetime += '_'
        datetime += str(elem)
        
    return datetime    

def plot_transfer_graph(path, name, results):
    
    r_targets = np.array(results['targets'])
    r_prediction = np.array(results['classifier'].ca.predictions)
    r_probabilities = np.array(results['classifier'].ca.probabilities)
    r_values = np.array(results['classifier'].ca.estimates)      

    p_probabilities = np.array([p[1][p[0]] for p in r_probabilities])
      
    report = ''
        
    for target in np.unique(r_targets):
            
        mask_target = r_targets == target
        prediction = r_prediction[mask_target]
        probabilities = p_probabilities[mask_target]
        values = r_values[mask_target]
            
        f_pred = plt.figure()
        a_prob = f_pred.add_subplot(111)
        #a_pred = f_pred.add_subplot(212)
            
        report = report +'--------- '+target+' -------------\n'
            
        for label in np.unique(prediction):
            mask_label = prediction == label
            n_pred = np.count_nonzero(mask_label)
            n_vols = len(prediction)
                
            perc = float(n_pred)/float(n_vols)
            report = report+'percentage of volumes labelled as '+label+' : '+str(perc)+' \n'
            
            mean = np.mean(probabilities[mask_label])
            plt_mean = np.linspace(mean, mean, num=len(probabilities))
            
            report = report+'mean probability: '+str(mean)+'\n'
            
            a_prob.plot(probabilities * mask_label)
            color = a_prob.get_lines()[-1:][0].get_color()
            a_prob.plot(plt_mean, color+'--', linewidth=2)
            a_prob.set_ylim(bottom=0.55)
            #a_pred.plot(values * mask_label)
        a_prob.legend(np.unique(prediction))     
        a_prob.legend(np.unique(prediction))
        #a_pred.legend(np.unique(prediction))

        for axes in f_pred.get_axes():
            max = np.max(axes.get_lines()[0].get_data()[1])
            for i in range(3):
                axes.fill_between(np.arange(2*i*125, (2*i*125)+125), 
                                  -max, max,facecolor='yellow', alpha=0.2)
        
        
        fname = os.path.join(path, name+'_'+target+'_probabilities_values.png')
        f_pred.savefig(fname)
            
    rep_txt = name+'_stats.txt'   
    
    rep = open(os.path.join(path, rep_txt), 'w')
    rep.write(report)
    rep.close()  
    

def plot_clusters_graph(path, name, results):
    
    
    color = dict({'trained':'r', 
              'fixation':'b', 
              'RestPre':'y', 
              'RestPost':'g',
              'untrained': 'k'})
    
    
    markers = dict({'trained':'p', 
              'fixation':'s', 
              'RestPre':'^', 
              'RestPost':'o',
              'untrained': '*'})
    
    colors = cycle('bgrcmykbgrmykbgrcmykbgrcmyk')
    
    clusters = results['clusters']
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    print targets
    report = ''
    
    for condition in clusters:
        print condition
        mask_target = targets == str(condition)
        m_predictions = predictions[mask_target]
        
        #####################################
        f_dist = plt.figure()
        a_dist = f_dist.add_subplot(111)
        a_dist.imshow(clusters[condition]['dist'])
        f_dist.colorbar(a_dist.get_images()[0])
        fname = os.path.join(path, name+'_'+condition+'_dist.png')
        f_dist.savefig(fname)
        #####################################
        
        cluster_labels = clusters[condition]['clusters']
        pos = clusters[condition]['pos']
        
        f_cluster = plt.figure()
        a_cluster = f_cluster.add_subplot(111)
        
        f_predict = plt.figure()
        a_predict = f_predict.add_subplot(111)
        
        report = report + '---------- '+condition+' -----------\n'
        
        for cluster in np.unique(cluster_labels):
            
            cl_mask = cluster_labels == cluster
            c_pos = pos[cl_mask]
            col = colors.next()
            a_cluster.scatter(c_pos.T[0], c_pos.T[1], color = col)
            
            report = report+'\nCluster n. '+str(cluster)+'\n'
            
            for label in np.unique(m_predictions):
                p_mask = m_predictions[cl_mask] == label
                labelled = np.count_nonzero(p_mask)
                
                a_predict.scatter(c_pos.T[0][p_mask], c_pos.T[1][p_mask], color = col, 
                         marker = markers[label])
                
                perc = np.float(labelled)/np.count_nonzero(cl_mask)
                report = report + label+' : '+str(perc*100)+'\n'
        
        
        a_cluster.legend(np.unique(cluster_labels))
        fname = os.path.join(path, name+'_'+condition+'_mds_clusters.png')
        f_cluster.savefig(fname)
        
        a_predict.legend(np.unique(predictions))
        fname = os.path.join(path, name+'_'+condition+'_mds_predictions.png')
        f_predict.savefig(fname)
    
    
    rep_txt = name+'_stats.txt'
    
    rep = open(os.path.join(path, rep_txt), 'w')
    rep.write(report)
    rep.close()
    
def load_wu_fmri_data(path, name, task, el_vols=None, **kwargs):
    """
    returns imgList
    
    @param path: 
    @param name:
    @param task:
    @param el_vols: 
    """
    
    analysis = 'single'
    #sub_dirs = ['']
    img_pattern=''
    
    for arg in kwargs:
        if (arg == 'img_pattern'):
            img_pattern = kwargs[arg] 
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        if (arg == 'runs'):
            runs = int(kwargs[arg])

    path_file_dirs = []

    
    for dir in sub_dirs:
        if dir == 'none':
            dir = ''
        path_file_dirs.append(os.path.join(path,name,dir))
    
   
    print 'Loading...'
    
    fileL = []
    #Verifying which type of task I've to classify (task or rest) and loads filename in different dirs
    for path in path_file_dirs:
        fileL = fileL + os.listdir(path)
   
    
    #Verifying which kind of analysis I've to perform (single or group) and filter list elements   
    if cmp(analysis, 'single') == 0:
        fileL = [elem for elem in fileL if (elem.find(img_pattern) != -1) and (elem.find(task) != -1) and (elem.find('mni') == -1)]
    else:
        fileL = [elem for elem in fileL if elem.find(img_pattern) != -1 and elem.find(task) != -1 and elem.find('mni') != -1]

    
    #if no file are found I perform previous analysis!        
    if (len(fileL) < runs and len(fileL) >= 0):
        
        print ' **** File corrected not found... ****'
        if cmp(analysis, 'single') == 0:
            mc_wu_data(path, name, task = 'task')
            mc_wu_data(path, name, task = 'rest')
        else:
            wu_to_mni(path, name, task = 'rest')
            wu_to_mni(path, name, task = 'task')
        
        if cmp(task, 'task') == 0:
            mc_wu_data(path, name, task)
            fileL = os.listdir(path_file_dirs[0])
        else:
            fileL = os.listdir(path_file_dirs[0]) + os.listdir(path_file_dirs[1])
        
        if cmp(analysis, 'single') == 0:
            fileL = [elem for elem in fileL if (elem.find(img_pattern) != -1) and (elem.find(task) != -1) and (elem.find('mni') == -1)]
        else:
            fileL = [elem for elem in fileL if (elem.find(img_pattern) != -1) and (elem.find(task) != -1) and (elem.find('mni') != -1)]
        
        
    else:
        print 'File corrected found ....'   
    
        
    ### Data loading ###
    fileL.sort()
    
    imgList = []
    
    for img in fileL:
         
        if os.path.exists(os.path.join(path_file_dirs[0], img)):
            pathF = os.path.join(path_file_dirs[0], img)
        else:
            pathF = os.path.join(path_file_dirs[1], img)
            
            
        print 'Now loading... '+pathF        
        im = ni.load(pathF)
        
        data = im.get_data()
        
        print data.shape
        
        new_im = ni.Nifti1Image(data[:,:,:,el_vols:], affine = im.get_affine(), header = im.get_header()) 
        
        del data, im
        imgList.append(new_im)
        
    print 'The image list is of ' + str(len(imgList)) + ' images.'
        
    del fileL
    return imgList

def load_dataset(path, subj, type, **kwargs):
    '''
    @param mask: mask is a string indicating the area to be analyzed
            - total: the entire brain voxels;
            - visual: the visual cortex;
            - ll : Lower Left Visual Quadrant
            - lr : Lower Right Visual Quadrant
            - ul : Upper Left Visual Quadrant
            - ur : Upper Right Visual Quadrant
    '''
    
    niftiFilez = load_wu_fmri_data(path, name = subj, task = type, el_vols = 0, **kwargs)
    

    ### Code to substitute   
    [code, attr] = load_attributes(path, type, subj, **kwargs)        
    if code == 0:
        del niftiFilez
        return code
        
    files = len(niftiFilez)  
    #Mask issues     
  
    mask = load_mask(path, subj, **kwargs)        
    
    #print 'Mask used: '+ mask
    
    volSum = 0;
        
    for i in range(len(niftiFilez)):
            
        volSum += niftiFilez[i].shape[3]

    if volSum != len(attr.targets):
        del niftiFilez
        print subj + ' *** ERROR: Attributes Length mismatch with fMRI volumes! ***'
        return 0;       
        
        
    try:
        print 'Loading dataset...'
        ds = fmri_dataset(niftiFilez, targets = attr.targets, chunks = attr.chunks, mask = mask) 
        print  'Dataset loaded...'
        del niftiFilez
            
    except ValueError, e:
        print subj + ' *** ERROR: '+ str(e)
        del niftiFilez
        return 0;
    
    
    ev_list = []
    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)
    for i in range(len(events)):
        duration = events[i]['duration']
        for j in range(duration):
            ev_list.append(i+1)               
    ds.a['events'] = events
    ds.sa['events_number'] = ev_list
    
    f_list = []
    for i in range(files):
        for j in range(len(ds)/files):
                f_list.append(i+1)

    ds.sa['file'] = f_list
    return ds    


def load_mask(path, subj, **kwargs):
    '''
    @param mask_type: indicates the type of atlas you want to use:
            - wu : Washington University ROIs extracted during experiments
            - juelich : FSL standard Juelich Maps.
    @param mask_area: mask is a string indicating the area to be analyzed
            - total: the entire brain voxels;
            - visual or other (broca, ba1 ecc.): other masks (with mask_type = 'wu' the only field could be 'visual')
            - ll : Lower Left Visual Quadrant (it could be applied only with mask_type = 'wu')
            - lr : Lower Right Visual Quadrant (it could be applied only with mask_type = 'wu')
            - ul : Upper Left Visual Quadrant (it could be applied only with mask_type = 'wu')
            - ur : Upper Right Visual Quadrant (it could be applied only with mask_type = 'wu') 
    @param mask_space: Coordinate space of the mask used. (Needed only for Juelich Maps!)
            - mni: The standard MNI space. 
            - wu: The Washington University space.
    '''
    for arg in kwargs:
        if (arg == 'mask_atlas'):
            mask_type = kwargs[arg]
        if (arg == 'mask_area'):
            mask_area = kwargs[arg]    
            
            
    if (mask_type == 'wu'):
        mask = load_mask_wu(path, subj, **kwargs)
    else:
        if (mask_area == 'total'):
            mask = load_mask_wu(path, subj, **kwargs)
        else:
            mask = load_mask_juelich(**kwargs)
        
    return mask



def load_mask_wu(path, subj, **kwargs):
    

    mask_area = ['total']
    
    for arg in kwargs:
        if (arg == 'mask_area'):
            mask_area = kwargs[arg].split(',')
  
    
    #Mask issues    
    roi_folder = '1_single_ROIs'
    isScaled = False
    sub_dir = ['none']
    
    for arg in kwargs:
        if (arg == 'roi_folder'):
            roi_folder = kwargs[arg]
        if (arg == 'coords'): #To be implemented
            coords = kwargs[arg]
        if (arg == 'sub_dir'):
            sub_dir = kwargs[arg].split(',')
        if (arg == 'scaled'):
            isScaled = kwargs[arg]
            
            
    '''
    for m in mask_list:
        if coords in locals():
            return
    '''
    mask_to_find = ''

    mask_path = os.path.join(path, roi_folder)
    
    scaled = ''
    
    if isScaled == 'True':
        scaled = 'scaled'
    
    if (mask_area == ['visual']):
        mask_list = os.listdir(mask_path)
        mask_list = [m for m in mask_list if m.find(scaled) != -1 and m.find('hdr')!=-1 ]
                                 
    elif (mask_area == ['total']):
        mask_path = os.path.join(path, subj)
        mask_list = os.listdir(mask_path)
        mask_to_find = subj+'_mask_mask'
        mask_list = [m for m in mask_list if m.find(mask_to_find) != -1]
          
    else:
        mask_list_1 = os.listdir(mask_path)
        mask_list = []
        for m_ar in mask_area:
            mask_list = mask_list + [m for m in mask_list_1 if m.find(scaled) != -1 and m.find('hdr')!=-1 
                     and (m[:3].find(m_ar) != -1 or m[-15:].find(m_ar) !=-1)]
      
    
    print 'Mask searched in '+mask_path
    
    files = []
    if len(mask_list) == 0:
        mask_path = os.path.join(path, subj)
        dir = sub_dir[0]
        if dir == 'none':
            dir = ''
        
        files = files + os.listdir(os.path.join(path, subj, dir))
            
        first = files.pop()
        maskExtractor( os.path.join(path, subj, dir, first), 
                                        os.path.join(path, subj, subj+'_mask.nii.gz'))       
        mask_list = [subj+'_mask_mask.nii.gz']
    
    
    data = 0
    for m in mask_list:
        img = ni.load(os.path.join(mask_path,m))
        data = data + img.get_data() 
        print 'Mask used: '+img.get_filename()

    mask = ni.Nifti1Image(data.squeeze(), img.get_affine())
        
    return mask

   
def load_mask_juelich(**kwargs):

    mask_space = 'wu'
    mask_area = ['total']
    mask_excluded = 'none'
    for arg in kwargs:
        if (arg == 'mask_area'):
            mask_area = kwargs[arg].split(',')
        if (arg == 'mask_space'):
            mask_space = kwargs[arg]
        if (arg == 'mask_excluded'):
            mask_excluded = kwargs[arg]   
            
            
    mask_path = os.path.join('/media/DATA/fmri/ROI_MNI',mask_space)

    mask_list_1 = os.listdir(mask_path)
    mask_list = []
    for m_ar in mask_area:
        mask_list = mask_list + [m for m in mask_list_1 if m.find(m_ar) != -1 
                                 and m.find(mask_excluded) == -1]
    data = 0
    
    for m in mask_list:
        img = ni.load(os.path.join(mask_path,m))
        data = data + img.get_data() 
        print 'Mask used: '+img.get_filename()

    mask = ni.Nifti1Image(data, img.get_affine())

    
    return mask    
    

def load_attributes (path, task, subj, **kwargs):
    ## Should return attr and a code to check if loading has been exploited #####
    
    
    for arg in kwargs:
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        if (arg == 'event_file'):
            event_file = kwargs[arg]
        if (arg == 'fidl_type'):
            fidl_type = int(kwargs[arg])  
            
    completeDirs = []
    for dir in sub_dirs:
        if dir == 'none':
            dir = ''
        completeDirs.append(os.path.join(path,subj,dir))
    
    completeDirs.append(path)
    
    attrFiles = []
    for dir in completeDirs:
        attrFiles = attrFiles + os.listdir(dir)

    attrFiles = [f for f in attrFiles if f.find(event_file) != -1]
    
    if len(attrFiles) == 0:
        print ' *******       ERROR: No attribute file found!        *********'
        print ' *** Check in '+os.path.join(path, subj)+' or '+os.path.join(path)+' ******'
        return [0, None]
    
    
    txtAttr = [f for f in attrFiles if f.find('.txt') != -1]
    
    
    attrFilename = ''
    if len(txtAttr) > 0:
        
        for dir in completeDirs:
            if (os.path.exists(os.path.join(dir, txtAttr[0]))):
                attrFilename = os.path.join(dir, txtAttr[0])               
       
    else:
        
        for dir in completeDirs:
            if (os.path.exists(os.path.join(dir, attrFiles[0]))):
                fidl_convert(os.path.join(dir, attrFiles[0]), os.path.join(dir, attrFiles[0][:-5]+'.txt'), type=fidl_type)
                attrFilename = os.path.join(dir, attrFiles[0][:-5]+'.txt')

    
    attr = SampleAttributes(attrFilename)
    return [1, attr]
    

def read_configuration (path, experiment, type):
    
    import ConfigParser
    
    config = ConfigParser.ConfigParser()
    
    config.read(os.path.join(path,experiment))
    
    
    print 'Reading config file '+os.path.join(path,experiment)
    
    types = config.get('path', 'types').split(',')
    
    if types.count(type) > 0:
        types.remove(type)
    
    for typ in types:
        config.remove_section(typ)
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            print item
    
    return dict(configuration)   


def save_results(path, results, configuration):

    datetime = get_time()
    analysis = configuration['analysis_type']
    mask = configuration['mask_area']
    task = configuration['analysis_task']
    
    new_dir = datetime+'_'+analysis+'_'+mask+'_'+task
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    parent_dir = os.path.join(path, '0_results', new_dir)
    
    if analysis == 'searchlight':
        save_results_searchlight(parent_dir, results)
    elif analysis == 'transfer_learning':
        save_results_transfer_learning(parent_dir, results)
    elif analysis == 'clustering':
        save_results_clustering(parent_dir, results)
    else:
        save_results_basic(parent_dir, results)  
    '''
    for key in results:
        
        name = key
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        for key in results[name]:
            
            if key == 'map':
                map = results[name][key]
                fname = name+'_map.nii.gz'
                ni.save(map, os.path.join(results_dir,fname))
            elif key == 'stats':
                stats = results[name][key]
                fname = name+'_stats.txt'
                file = open(os.path.join(results_dir,fname), 'w')
                file.write(stats)
                file.close()
            else:
                obj = results[name][key]
                if key == 'classifier':
                    obj = results[name][key].ca
                fname = name+'_'+key+'.pyobj'          
                file = open(os.path.join(results_dir,fname), 'w')
                pickle.dump(obj, file)
                file.close()
    ''' 
    ###################################################################        
    import csv
    w = csv.writer(open(os.path.join(parent_dir, 'configuration.csv'), "w"))
    for key, val in configuration.items():
        w.writerow([key, val])   
    
    print 'Result saved in '+parent_dir
    
    return 'OK' 

def save_results_searchlight (path, analysis, type, mask, results):
    
    
    datetime = get_time()
    configuration = results[0]['configuration']
    
    new_dir = datetime+'_'+analysis.func_name+'_'+type+'_'+mask 
    
    command = 'mkdir '+os.path.join(path, '0_results', new_dir)
    os.system(command)
    
    dir = os.path.join(path, '0_results', new_dir)

    if (analysis.func_name == 'searchlight'):
        for res in results:
            subj = res['name']
            radius = res['results']['radius']
            filename = datetime+'_'+subj+'_searchlight_'+type+'_'+mask+'_rad_'+radius
            ni.save(results['results']['map'], 
                    os.path.join(path, subj, filename+'_.nii.gz'))
    
    else:      
           
        import csv
        w = csv.writer(open(os.path.join(dir, 'configuration.csv'), "w"))
        for key, val in results[0]['configuration'].items():
            w.writerow([key, val])
            
        pickle.dump(results, open(os.path.join(dir, new_dir+'_results.pyobj'), 'w'))
        

    print 'Results writed in '+path    
    return new_dir

def save_results_basic(path, results):
    
    parent_dir = path 
    
    for key in results:
        
        name = key
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        for key in results[name]:
            
            if key == 'map':
                m_mean = results[name]['map'].pop()
                fname = name+'_mean_map.nii.gz'
                ni.save(m_mean, os.path.join(results_dir,fname))
                
                for map, t in zip(results[name][key], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
                    ni.save(map, os.path.join(results_dir,fname))
            elif key == 'stats':
                stats = results[name][key]
                fname = name+'_stats.txt'
                file = open(os.path.join(results_dir,fname), 'w')
                file.write(str(stats))
                file.close()
            else:
                obj = results[name][key]
                if key == 'classifier':
                    obj = results[name][key].ca
                fname = name+'_'+key+'.pyobj'          
                file = open(os.path.join(results_dir,fname), 'w')
                pickle.dump(obj, file)
                file.close()
     
    ###################################################################          
    
    print 'Result saved in '+parent_dir
    
    return 'OK' 


def save_results_transfer_learning(path, results):
    
    for name in results:
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name)        
        
        m_mean = results[name]['map'].pop()
        fname = name+'_mean_map.nii.gz'
        ni.save(m_mean, os.path.join(results_dir,fname))
        
        for map, t in zip(results[name]['map'], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
                    ni.save(map, os.path.join(results_dir,fname))

        
        stats = results[name]['stats']
        fname = name+'_stats.txt'
        file = open(os.path.join(results_dir,fname), 'w')
        file.write(str(stats))
        file.close()
        
        obj = results[name]['classifier'].ca
        fname = name+'_'+'classifier'+'.pyobj'          
        file = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file)
        file.close()
        
        obj = results[name]['targets']
        fname = name+'_'+'targets'+'.pyobj'          
        file = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file)
        file.close()
        
        plot_transfer_graph(results_dir, name, results[name])
        
    return


def save_results_clustering(path, results):
    
    
    for name in results:
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name) 
        
        for key in results[name]:
            obj = results[name][key]
            fname = name+'_'+key+'.pyobj'          
            file = open(os.path.join(results_dir,fname), 'w')
            pickle.dump(obj, file)
            file.close()
            
            if key == 'clusters':
                plot_clusters_graph(results_dir, name, results[name])
  
    return


    

