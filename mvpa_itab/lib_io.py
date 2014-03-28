#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

import nibabel as ni
import os
from main_wu import *
from utils import *
import matplotlib.pyplot as plt
from mvpa2.suite import find_events, fmri_dataset, SampleAttributes
from itertools import cycle
import cPickle as pickle

#from memory_profiler import profile

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

def plot_transfer_graph_prob_fitted(path, name, analysis_folder):
    
    from sklearn.linear_model import Ridge
    from scipy.interpolate import UnivariateSpline
    result_file = open(
                       os.path.join(path, 
                               '0_results', 
                               analysis_folder, 
                               name, 
                               name+'_classifier.pyobj')
                       , 'r')
    
    results = pickle.load(result_file)
    
    runs = 12
    
    probabilities = results.probabilities
    prob = np.array([p[1][p[0]] for p in probabilities])
    pred = np.array([p[0] for p in probabilities])
    lab = np.unique(results.predictions)
    run_length = len(prob)/runs
       
    ridge = Ridge()
    
    f = plt.figure(figsize=(11,8))
    f2 = plt.figure(figsize=(11,8))
    data_sm = dict()
    data_or = dict()
    for c in np.unique(lab):
        data_sm[c] = []
        data_or[c] = []
    for i in range(12):
        if i < 6:
            add = 1
            l = '_pre'
        else:
            add = 2
            l = '_post'
        avg = []
        for c in np.unique(pred):
            a = f.add_subplot(3,2,(c*2)+add)
            a2 = f2.add_subplot(3,2,(c*2)+add)
            a.set_title(lab[c]+l)
            #v = prob[i*run_length:(i+1)*run_length]
            v = prob[i*run_length:(i+1)*run_length] * (pred[i*run_length:(i+1)*run_length] == c)
            v[len(v)-1] = 0
            yy = v.copy()
            
            
            
            xx = np.linspace(0, len(v), len(v))
            s = UnivariateSpline(xx, yy, s=5)
            ys = s(xx)
            try:
                ridge.fit(np.vander(xx, 7), yy)
                y_fit = ridge.predict(np.vander(xx, 7))
            except LinAlgError,err:
                ridge.fit(np.vander(xx, 9), yy)
                y_fit = ridge.predict(np.vander(xx, 9))
            
            data_sm[lab[c]].append(ys)
            data_or[lab[c]].append(v)
            
            
            
            a.plot(y_fit)
            a2.plot(ys)
            
            
            a.set_ybound(upper=1.1, lower=-0.1)
            a2.set_ybound(upper=1.1, lower=-0.1)
            
    f.legend(a.get_lines()[:],range(runs), loc=0)
    fname = os.path.join(path,'0_results', 
                               analysis_folder, 
                               name, 
                               name+'_values_fitted_ov.png')
    f.savefig(fname)
    f2.savefig(fname[:-3]+'smooth.png')
    return data_sm, data_or
    


def plot_transfer_graph_fitted(path, name, analysis_folder):
    
    from sklearn.linear_model import Ridge
    from scipy.interpolate import UnivariateSpline
    result_file = open(
                       os.path.join(path, 
                               '0_results', 
                               analysis_folder, 
                               name, 
                               name+'_classifier.pyobj')
                       , 'r')
    
    results = pickle.load(result_file)
    
    runs = 6
    
    values = results.estimates
    run_length = len(values)/runs
    
    ridge = Ridge()
    
    f = plt.figure()
    a = f.add_subplot(111)
    for i in range(runs):
        
        v = values[i*run_length:(i+1)*run_length]
        yy = v.copy()
        
        xx = np.linspace(0, len(v), len(v))
        
        try:
            ridge.fit(np.vander(xx, 12), yy)
            y_fit = ridge.predict(np.vander(xx, 12))
        except LinAlgError,err:
            ridge.fit(np.vander(xx, 9), yy)
            y_fit = ridge.predict(np.vander(xx, 9))
        
        a.plot(y_fit)
    
    a.legend(range(runs))
    fname = os.path.join(path,'0_results', 
                               analysis_folder, 
                               name, 
                               name+'_values_fitted.png')
    f.savefig(fname)
    


def plot_transfer_graph(path, name, results):
    
    r_targets = np.array(results['targets'])
    r_prediction = np.array(results['classifier'].ca.predictions)
    if str(results['classifier'].find('SVM') != -1):
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
        #a_prob.legend(np.unique(prediction))
        #a_pred.legend(np.unique(prediction))

        for axes in f_pred.get_axes():
            max = np.max(axes.get_lines()[0].get_data()[1])
            for i in range(3):
                axes.fill_between(np.arange(2*i*125, (2*i*125)+125), 
                                  -max, max,facecolor='yellow', alpha=0.2)
        
        
        fname = os.path.join(path, name+'_'+target+'_probabilities_values.png')
        f_pred.savefig(fname)
    
    
    rep_txt = name+'_stats_probab.txt'   
    
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

def plot_cv_results(cv, err, title):
    # make new figure
    pl.figure()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # null distribution samples
    dist_samples = np.asarray(cv.null_dist.ca.dist_samples)
    for i in range(len(err)):
        c = colors.next()
        # histogram of all computed errors from permuted data per CV-fold
        pl.hist(np.ravel(dist_samples[i]), bins=20, color=c,
                label='CV-fold %i' %i, alpha=0.5,
                range=(dist_samples.min(), dist_samples.max()))
        # empirical error
        pl.axvline(np.asscalar(err[i]), color=c)

    # chance-level for a binary classification with balanced samples
    pl.axvline(0.5, color='black', ls='--')
    # scale x-axis to full range of possible error values
    pl.xlim(0,1)
    pl.xlabel(title)


def plot_scatter_2d(ds_merged, method='mds', fig_number = 1):
    
    from sklearn import decomposition, manifold, lda, ensemble
    """
    methods: 'mds', 'pca', 'iso', 'forest', 'embedding'
    """
    
    data = ds_merged.samples
    
    stringa=''
    if method == 'pca':
        clf = decomposition.RandomizedPCA(n_components=2)
        stringa = 'Principal Component Analysis'
    ########    
    elif method == 'iso':
        clf = manifold.Isomap(30, n_components=2)
        stringa = 'Iso surfaces '
    #########    
    elif method == 'forest':
        hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
        data = hasher.fit_transform(data)
        clf = decomposition.RandomizedPCA(n_components=2)
        stringa = 'Random Forests'
    ########
    elif method == 'embedding':
        clf = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
        stringa = 'Spectral Embedding'
    #########
    else:
        clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
        stringa = 'Multidimensional scaling'
        
        
    ###########################
    #dist_matrix = squareform(pdist(data, 'euclidean'))

    print stringa+' is performing...'

    pos = clf.fit_transform(data)

    colors = cycle('bgrymkybgrcmybgrcmybgrcmy')
    
    f = plt.figure()
    a = f.add_subplot(111)
    a.set_title(stringa)
    for label in np.unique(ds_merged.targets):
        m = ds_merged.targets == label
        data_m = pos[m]
        c = colors.next()
        a.scatter(data_m.T[0].mean(), data_m.T[1].mean(), label=label, color=c, s=120)
        a.scatter(data_m.T[0][::2], data_m.T[1][::2], color=c)
        '''
        cov_ = np.cov(data_m.T)
        v, w = np.linalg.eigh(cov_)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 0.5
        ell = mpl.patches.Ellipse(np.mean(data_m, axis=0), v[0], v[1],
                                  180 + angle, color=c)
        ell.set_clip_box(a.bbox)
        ell.set_alpha(0.2)
        a.add_artist(ell)
        '''
    a.legend()
    return


def load_conc_fmri_data(conc_file_list, el_vols = 0, **kwargs):
        
        imgList = []
        
        for file in conc_file_list:
            
            print 'Now loading... '+file        
            
            im = ni.load(file)
            data = im.get_data()
        
            print data.shape
        
            new_im = ni.Nifti1Image(data[:,:,:,el_vols:], 
                                    affine = im.get_affine(), 
                                    header = im.get_header()) 
        
            del data, im
            imgList.append(new_im)
        
        return imgList

#@profile    
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
        if (arg == 'use_task_name'):
            use_task_name = kwargs[arg]

    path_file_dirs = []
    
    if use_task_name == 'False':
        task = ''
    
    for dir in sub_dirs:
        if dir == 'none':
            dir = ''
        if dir.find('/') == -1:
            path_file_dirs.append(os.path.join(path,name,dir))
    
   
    #print 'Loading...'
    
    fileL = []
    #Verifying which type of task I've to classify (task or rest) and loads filename in different dirs
    for path in path_file_dirs:
        fileL = fileL + os.listdir(path)

    #Verifying which kind of analysis I've to perform (single or group) and filter list elements   
    if cmp(analysis, 'single') == 0:
        fileL = [elem for elem in fileL if (elem.find(img_pattern) != -1) and (elem.find(task) != -1) and (elem.find('mni') == -1)]
    else:
        fileL = [elem for elem in fileL if elem.find(img_pattern) != -1 and elem.find(task) != -1 and elem.find('mni') != -1]

    print fileL
    #if no file are found I perform previous analysis!        
    if (len(fileL) <= runs and len(fileL) == 0):
        """
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
        
        """
        raise OSError('Files not found, please check if data exists in '+str(path_file_dirs))
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
            
        #if (str(len(imgList))) < 1:
        print 'Now loading... '+pathF        
        
        im = ni.load(pathF)
        data = im.get_data()
        
        #if (str(len(imgList))) < 1:
        print data.shape
        
        new_im = im.__class__(data[:,:,:,el_vols:], affine = im.get_affine(), header = im.get_header()) 
        new_im.set_filename(pathF)
        
        del data
        del im
        imgList.append(new_im)
        
    #print 'The image list is of ' + str(len(imgList)) + ' images.'
        
    del fileL
    return imgList

def load_beta_dataset(path, subj, type, **kwargs):
    
    for arg in kwargs:
        if arg == 'runs':
            runs = np.int(kwargs[arg])

    dataset = []
    targets = []
    for r in range(runs):
        filename = 'avg_stats_for_regions_'+subj+'_'+str(runs)+'run_'+subj+'_run'+str(r+1)+'_vox.txt'
        f = open(os.path.join(path, subj, filename), 'r')
        run_data = read_beta_file(f, r, **kwargs)
        dataset.append(run_data[0])
        targets.append(run_data[1])
        
    dataset = np.vstack(dataset)
    targets = np.vstack(targets)
    
    ds = dataset_wizard(dataset, targets=targets[:,0], chunks=targets[:,1])
    
    return ds

def read_beta_file(f, run, time_begin=1, time_end=14, **kwargs):
    #print kwargs
    for arg in kwargs:
        #print arg
        if arg == 'mask_area':
            mask = kwargs[arg].split(',')
        if arg == 'time_begin':
            time_begin = np.int(kwargs[arg])
        if arg == 'time_end':
            time_end = np.int(kwargs[arg])
            
    if 'time_begin' not in locals():
        time_begin = 0
    if 'time_end' not in locals():
        time_end = 600   
        
    file_lines = f.readlines()
        
    data_flag = False
    data = []
    condition = []
    runs = []
    line_ctr = 0
    for line in file_lines:
        
        s_line = line.split()
        
        if len(s_line) == 0:
            data_flag = False
            line_ctr = 0
            continue
          
        #if line.find('TIMECOURSE : ') != -1:
        if s_line[0] == 'TIMECOURSE':
            cond, time_points = decode_line(line, 'timecourse')
            voxel_per_condition = 0
            
            interval = time_end - time_begin
            if interval >= time_points:
                interval = time_points
            if time_end > time_points:
                time_end = time_points
            if time_begin > time_points:
                time_begin = 0
            if time_begin >= time_end:
                interval = time_points
                
            for i in range(int(interval)):
                condition.append(cond)
                runs.append(run)
            
        #if line.find('REGION : ') !=-1:
        if s_line[0] == 'REGION':
            voxel_num, roi_name = decode_line(line, 'region')
            for area in mask:
                if roi_name.find(area) != -1:
                    voxel_per_condition += int(voxel_num)
                    data_flag = True
                    line_ctr = 0
                    
                                        
        #print line
        if data_flag == True:
            #print line
            try:
                int(s_line[0])
            except ValueError, err:
                continue
            
            line_ctr += 1
            
            data_begin = 3 + time_begin
            data_end = data_begin +interval
            #print 'voxel_num='+str(voxel_num)
            #print 'line_ctr='+str(line_ctr)
            if line_ctr <= voxel_num:
                #print line.split()[data_begin:data_end]
                data.append(np.float_(np.array(line.split()[data_begin:data_end])))
    
    data_div = []
    ndata = np.array(data)
    for block in range(len(np.unique(condition))):
        data_div.append(ndata[block * voxel_per_condition:(block+1) * voxel_per_condition])
    
    ndata = np.hstack(data_div).T
    targets = np.vstack((condition, runs)).T
    
    return ndata, targets

def decode_line(line, keyword):
    
    if keyword == 'timecourse':
        
        condition = line.split()[2]
        time_points = line.split()[-1]
        return condition, time_points
    
    elif keyword == 'data':
        
        data = np.array(line)
        return data
    
    elif keyword == 'region':
        
        vox_num = line.split()[-1]
        roi_name = line.split()[-2]
        return vox_num, roi_name


def read_beta_data(line, voxel_num, time_points):
    
    
    
    return
    
    
#@profile
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
    
    use_conc = 'False'
    skip_vols = 0
    
    #print kwargs
    
    for arg in kwargs:
        if arg == 'skip_vols':
            skip_vols = np.int(kwargs[arg])
        if arg == 'use_conc':
            use_conc = kwargs[arg]
        if arg == 'conc_file':
            conc_file = kwargs[arg]
    
    
    
    if use_conc == 'False':      
        try:
            niftiFilez = load_wu_fmri_data(path, name = subj, task = type, el_vols = skip_vols, **kwargs)
        except OSError, err:
            print err
            return 0
    else:
        conc_file_list = read_conc(path, subj, conc_file)
        conc_file_list = modify_conc_list(path, subj, conc_file_list)
        try:
            niftiFilez = load_conc_fmri_data(conc_file_list, el_vols = skip_vols, **kwargs)
        except IOError, err:
            print err
            return 0
        
        kwargs = update_subdirs(conc_file_list, subj, **kwargs)
        #print kwargs

    ### Code to substitute   
    [code, attr] = load_attributes(path, type, subj, **kwargs)        
    if code == 0:
        del niftiFilez
        raise IOError('Attributes file not found')
        
    files = len(niftiFilez)  
    #Mask issues     
  
    mask = load_mask(path, subj, **kwargs)        
    
    #print 'Mask used: '+ mask
    
    volSum = 0;
        
    for i in range(len(niftiFilez)):
            
        volSum += niftiFilez[i].shape[3]

    if volSum != len(attr.targets):
        del niftiFilez
        print subj + ' *** ERROR: Attributes Length mismatches with fMRI volumes! ***'
        raise ValueError('Attributes Length mismatches with fMRI volumes!')       
        
        
    try:
        print 'Loading dataset...'
        ds = fmri_dataset(niftiFilez, targets = attr.targets, chunks = attr.chunks, mask = mask) 
        print  'Dataset loaded...'
        
            
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
    ds.sa['name'] = [subj for i in range(len(ds.sa.chunks))]
    
    f_list = []
    for i in range(files):
        for j in range(niftiFilez[i].shape[-1:][0]):
                f_list.append(i+1)

    ds.sa['file'] = f_list
    
    del niftiFilez
    
    return ds    

def load_spatiotemporal_dataset(ds, **kwargs):
    
    onset = 0
    
    for arg in kwargs:
        if (arg == 'onset'):
            onset = kwargs[arg]
        if (arg == 'duration'):
            duration = kwargs[arg]
        if (arg == 'enable_results'):
            enable_results = kwargs[arg]
        
        
        
    events = find_events(targets = ds.sa.targets, chunks = ds.sa.chunks)   
    
    #task_events = [e for e in events if e['targets'] in ['Vipassana','Samatha']]
    
    if 'duration' in locals():
        events = [e for e in events if e['duration'] >= duration]
    else:
        duration = np.min([ev['duration'] for ev in events])

    for e in events:
        e['onset'] += onset           
        e['duration'] = duration
        
    evds = eventrelated_dataset(ds, events = events)
    
    return evds


#@profile
def load_mask(path, subj, **kwargs):
    '''
    @param mask_type: indicates the type of atlas you want to use:
            - wu : Washington University ROIs extracted during experiments
            - juelich : FSL standard Juelich Maps.
    @param mask_area: mask is a string indicating the area to be analyzed
            - total: the entire brain voxels;
            - searchlight_3: mask from searchlight analysis exploited with radius equal to 3
            - searchlight_5: mask from searchlight analysis exploited with radius equal to 5
            - visual or other (broca, ba1 ecc.): other masks (with mask_type = 'wu' the only field could be 'visual')
            - v3a7 : Visual Cortex areas V3 V3a V7
            - intersect : intersection of v3a7 and searchlight 5
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
        if (arg == 'mask_dir'):
            path = kwargs[arg]  
    
    if mask_area == 'none':
        return None        
            
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
    #print mask_area
    if isScaled == 'True':
        scaled = 'scaled'
    
    if (mask_area == ['visual']):
        mask_list = os.listdir(mask_path)
        mask_list = [m for m in mask_list if m.find(scaled) != -1 and m.find('hdr')!=-1 ]
                                 
    elif (mask_area == ['total']):
        mask_path = os.path.join(path, subj)
        mask_list = os.listdir(mask_path)
        mask_to_find1 = subj+'_mask_mask'
        mask_to_find2 = 'mask_'+subj+'_mask'
        mask_to_find3 = '_mask.nii.gz'
        mask_list = [m for m in mask_list if m.find(mask_to_find1) != -1 or m.find(mask_to_find2) != -1 \
                     or m.find(mask_to_find3) != -1]
    
    elif (mask_area == ['searchlight_3'] or mask_area == ['searchlight_5']):
        mask_list = os.listdir(mask_path)
        if mask_area == ['searchlight_3']:
            mask_to_find = 'mask_sl_r_3_t_54'
        else:
            mask_to_find = 'mask_sl_r_5_t_54'
        mask_list = [m for m in mask_list if m.find(mask_to_find) != -1]      
    else:
        mask_list_1 = os.listdir(mask_path)
        mask_list = []
        for m_ar in mask_area:
            mask_list = mask_list + [m for m in mask_list_1 if #(m.find('nii.gz') != -1 and m.find(m_ar)!=-1) or
                      ((m[:].find(m_ar) != -1 or m[-15:].find(m_ar) !=-1) and m.find('nii.gz') != -1) or
                      ((m[:].find(m_ar) != -1 or m[-15:].find(m_ar) !=-1) and m.find('hdr') != -1)]
      
    
    print 'Mask searched in '+mask_path+' Mask(s) found: '+str(len(mask_list))
    
    files = []
    if len(mask_list) == 0:
        mask_path = os.path.join(path, subj)
        dir = sub_dir[0]
        if dir == 'none':
            dir = ''
        
        files = files + os.listdir(os.path.join(path, subj, dir))
            
        first = files.pop()
        
        bet_wu_data_(path, subj, dir)
    
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
        if dir.find('/') != -1:
            completeDirs.append(dir)
            
        completeDirs.append(os.path.join(path,subj,dir))
    
    completeDirs.append(path)
    completeDirs.append(os.path.join(path,subj))
    
    attrFiles = []
    for dir in completeDirs:
        attrFiles = attrFiles + os.listdir(dir)

    attrFiles = [f for f in attrFiles if f.find(event_file) != -1]
    print attrFiles
    if len(attrFiles) > 2:
        attrFiles = [f for f in attrFiles if f.find(subj) != -1]
        
    
    if len(attrFiles) == 0:
        print ' *******       ERROR: No attribute file found!        *********'
        print ' ***** Check in '+str(completeDirs)+' ********'
        return [0, None]
    
    
    #txtAttr = [f for f in attrFiles if f.find('.txt') != -1]
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

def modify_conc_list(path, subj, conc_filelist):
    """
    Function used to internally modify conc path, if remote directory is mounted at different
    location, the new mounting directory is passed as parameter.
    """
    new_list = []
    for file in conc_filelist:
        
        file = file[file.find(subj):-3]+'hdr'
        new_list.append(os.path.join(path,file))
        
    del conc_filelist
    return new_list


def read_conc(path, subj, conc_file_patt):
    
    conc_file_list = os.listdir(os.path.join(path, subj))
    conc_file_list = [f for f in conc_file_list if f.find('.conc') != -1 and f.find(conc_file_patt) != -1]
    
    c_file = conc_file_list[0]
    
    conc_file = open(os.path.join(path, subj, c_file), 'r')
    s = conc_file.readline()
    n_files = np.int(s.split(':')[1])
    
    i = 0
    filename_list = []
    while i < n_files:
        name = conc_file.readline()
        filename_list.append(name[name.find('/'):-1])
        i = i + 1
        
    return filename_list



def read_remote_configuration(path):
        
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(path,'remote.conf'))
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            print item
    
    return dict(configuration) 
    
    
    print 'Reading remote config file '+os.path.join(path,'remote.conf')

#@profile
def read_configuration (path, experiment, section):
    
    import ConfigParser
    
    config = ConfigParser.ConfigParser()
    
    config.read(os.path.join(path,experiment))
    
    
    print 'Reading config file '+os.path.join(path,experiment)
    
    types = config.get('path', 'types').split(',')
    
    if types.count(section) > 0:
        types.remove(section)
    
    for typ in types:
        config.remove_section(typ)
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            #print item
    
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
        write_all_subjects_map(path, new_dir)
    elif analysis == 'clustering':
        save_results_clustering(parent_dir, results)
    else:
        save_results_basic(parent_dir, results)
        write_all_subjects_map(path, new_dir)  

    ###################################################################        
    import csv
    w = csv.writer(open(os.path.join(parent_dir, 'configuration.csv'), "w"))
    for key, val in configuration.items():
        w.writerow([key, val])
        
    #######################################################################
    
    if analysis == 'searchlight':
        print 'Result saved in '+parent_dir
        return 'OK'
    else:
        file_summary = open(os.path.join(parent_dir, 'analysis_summary_'+mask+'_'+task+'.txt'), "w")
        res_list = []
    
        for subj in results.keys():
        
            list = [subj]
            file_summary.write(subj+'\t')
            
            r = results[subj]
        
            list.append(r['stats'].stats['ACC'])
            list.append(r['p'])
            
            file_summary.write(str(r['stats'].stats['ACC'])+'\t')
            file_summary.write(str(r['p'])+'\t')
            
            if analysis == 'transfer_learning':
                list.append(r['confusion_total'].stats['ACC'])
                list.append(r['d_prime'])
                list.append(r['beta'])
                list.append(r['c'])
                
                file_summary.write(str(r['confusion_total'].stats['ACC'])+'\t')
                file_summary.write(str(r['d_prime'])+'\t')
                file_summary.write(str(r['beta'])+'\t')
                file_summary.write(str(r['c'])+'\t')
                
            file_summary.write('\n')
            
        res_list.append(list)
        
        file_summary.close()
    
    print 'Result saved in '+parent_dir
    
    return 'OK' 

def save_results_searchlight (path, results):
    
    parent_dir = path 
    
    total_map = []
    
    for key in results:
        
        name = key
        command = 'mkdir '+os.path.join(parent_dir, name)
        os.system(command)
        
        results_dir = os.path.join(parent_dir, name)
        
        map = results[name]['map']
        
        radius = np.int(results[name]['radius'])
        
        
        if len(map.get_data().shape) > 3:
            mean_map = map.get_data().mean(axis=3)
            mean_img = ni.Nifti1Image(mean_map, affine=map.get_affine())
            fname = name+'_radius_'+str(radius)+'_searchlight_mean_map.nii.gz'
            ni.save(mean_img, os.path.join(results_dir,fname))
        else:
            mean_map = map.get_data()
            
        fname = name+'_radius_'+str(radius)+'_searchlight_map.nii.gz'
        ni.save(map, os.path.join(results_dir,fname))
        
        total_map.append(mean_map)
    
    total_map = np.array(total_map).mean(axis=0)
    total_img = ni.Nifti1Image(total_map, affine=map.get_affine())
    fname = 'accuracy_map_radius_'+str(radius)+'_searchlight_all_subj.nii.gz'
    ni.save(total_img, os.path.join(path,fname))
                   
    print 'Results writed in '+path    
    return path

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
                m_mean._data = (m_mean._data - np.mean(m_mean._data))/np.std(m_mean._data)
                ni.save(m_mean, os.path.join(results_dir,fname))
                
                for map, t in zip(results[name][key], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
                    map._data = (map._data - np.mean(map._data))/np.std(map._data)
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
    
    p = results[results.keys()[0]]['mahalanobis_similarity'][0].T[1]
    r = results[results.keys()[0]]['mahalanobis_similarity'][0].T[0]
    
    hist_sum = dict()
    
    means_s = dict()
    for t in np.unique(r):
        hist_sum[t] = dict()
        for l in np.unique(p):
            hist_sum[t][l] = dict()
            hist_sum[t][l]['dist'] = []
            hist_sum[t][l]['p'] = []
            means_s[l+'_'+t] = []
    
    
    for name in results:
        command = 'mkdir '+os.path.join(path, name)
        os.system(command)
        
        results_dir = os.path.join(path, name)        

        print '----------------------------'
                
        
        stats = results[name]['stats']
        fname = name+'_stats.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        file_.write(str(stats))
        p_value = results[name]['p-value']
        file_.write('\n\n p-values for each fold \n')
        for v in p_value:
            file_.write(str(v)+'\n')
        file_.write('\n\n Mean each fold p-value: '+str(p_value.mean()))
        file_.write('\n\n Mean null dist total accuracy value: '+str(results[name]['p']))
        file_.write('\nd-prime coefficient: '+str(results[name]['d_prime']))
        file_.write('\nbeta coefficient: '+str(results[name]['beta']))
        file_.write('\nc coefficient: '+str(results[name]['c']))
        #file.write('\n\nd-prime mahalanobis coeff: '+str(results[name]['d_prime_maha']))
        file_.close()
        
        if name == 'group':
            fname = name+'_fold_stats.txt'
            file_ = open(os.path.join(results_dir,fname), 'w')
            for m in stats.matrices:
                file_.write(str(m.stats['ACC']))
                file_.write('\n')
            file_.close()
                
        obj = results[name]['classifier'].ca
        fname = name+'_'+'classifier'+'.pyobj'          
        file_ = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file_)
        file_.close()
        
        obj = results[name]['targets']
        fname = name+'_'+'targets'+'.pyobj'          
        file_ = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file_)
        file_.close()
        
        obj = results[name]['predictions']
        fname = name+'_'+'predictions'+'.pyobj'          
        file_ = open(os.path.join(results_dir,fname), 'w')
        pickle.dump(obj, file_)
        file_.close()
        #plot_transfer_graph(results_dir, name, results[name])
        
        c_m = results[name]['confusion_target']
        fname = name+'_confusion_target.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        file_.write(str(c_m))
        file_.close()
        
        c_m = results[name]['confusion_total']
        fname = name+'_confusion_total.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        file_.write(str(c_m))
        file_.close()        
        
        full_data = results[name]['mahalanobis_similarity'][0]
        true_pred = results[name]['mahalanobis_similarity'][1]
        threshold = results[name]['mahalanobis_similarity'][2]
        distances = results[name]['mahalanobis_similarity'][4]
        
        t_mahala = full_data[true_pred]
        fname = name+'_mahalanobis_data.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        
        n_src_label = len(np.unique(full_data.T[1]))
        n_tar_label = len(np.unique(full_data.T[0]))
        
        plot_list = dict()
        
        
        for t in np.unique(full_data.T[1]):
            f, ax = plt.subplots(2, 1)
            plot_list[t] = [f, ax]
        
        #for each label of the target dataset (target classes)
        for tar in np.unique(full_data.T[0]): 
            
            #Rest mask
            t_pred_mask = full_data.T[0] == tar
            t_m_data = full_data[t_pred_mask * true_pred]
            
            #for each predicted label
            for lab in np.unique(full_data.T[1]):
                
                histo_fname = name+'_histo_'+lab+'_'+tar+'_dist.txt'
                histo_p_fname = name+'_histo_'+lab+'_'+tar+'_p.txt'
                
                
                
                all_vec = full_data[(full_data.T[1] == lab) * t_pred_mask]
                
                if len(t_m_data) != 0:
                    m_maha = t_m_data.T[1] == lab
                    true_vec = t_m_data[m_maha]
                    num = len(true_vec)
                    mean_maha = np.mean(np.float_(true_vec.T[2]))
                    mean_p = np.mean(np.float_(true_vec.T[3]))
                else:
                    num = 0
                    mean_maha = np.mean(np.float_(all_vec.T[2]))
                    mean_p = np.mean(np.float_(all_vec.T[3]))
                
                tot_mean = np.mean(np.float_(all_vec.T[2]))
                tot_p = np.mean(np.float_(all_vec.T[3]))
                               
                file_.write(tar+' '+lab+' '+str(num)+' '+str(mean_maha)+' '+str(mean_p)+' '+str(tot_mean)+' '+str(tot_p)+'\n')
                
                np.savetxt(os.path.join(results_dir,histo_fname), np.float_(all_vec.T[2]))
                np.savetxt(os.path.join(results_dir,histo_p_fname), np.float_(all_vec.T[3]))
                
                #bin_d = np.linspace(mn_d, mx_d, 25)
                #bin_p = np.linspace(0, 1, 25)
                
                plot_list[lab][1][0].hist(np.float_(all_vec.T[2]), bins=35, label=tar, alpha=0.5)
                plot_list[lab][1][1].hist(np.float_(all_vec.T[3]), bins=35, label=tar, alpha=0.5)          
                
                hist_sum[tar][lab]['dist'].append(np.float_(all_vec.T[2]))
                hist_sum[tar][lab]['p'].append(np.float_(all_vec.T[3]))
                
            
            histo_full_fname = name+'_'+tar+'_histo_.txt'
            histo_full_p_fname = name+'_'+tar+'_histo_p.txt'
            
            l = 0
            pred_array = np.zeros(full_data.T[1].shape)
            for lab in np.unique(full_data[t_pred_mask].T[1]):
                
                pred_array[full_data.T[1] == lab] = l
                
                l = l + 1
                
            #np.savetxt(os.path.join(results_dir,histo_full_fname), np.vstack((full_data[t_pred_mask].T[2:], pred_array)))


        for c in distances.keys():
            for tar in np.unique(full_data.T[0]):
                data = distances[c][full_data.T[0] == tar]
                f_d = plt.figure()
                a_d = f_d.add_subplot(111)
                a_d.plot(data)
                a_d.set_ylim(1000, 3000)
                step = data.__len__() / 6.
                for j in np.arange(6)+1:#n_runs
                    a_d.axvline(x = step * j, ymax=a_d.get_ylim()[1], color='y', linestyle='-', linewidth=1)
                a_d.axhline(y = threshold, color='r', linestyle='--', linewidth=2)
                
                means_s[c+'_'+tar].append(np.mean(data))
                a_d.axhline(y = np.mean(data), color='black', linestyle=':', linewidth=2)
                f_d.savefig(os.path.join(results_dir,name+'_distance_plot_'+c+'_'+tar+'_.png'))
            
                np.savetxt(os.path.join(results_dir,name+'_distance_txt_'+c+'_'+tar+'_.txt'), 
                           distances[c][full_data.T[0] == tar], fmt='%.4f')               
                
                
                
            
        
        for k in plot_list.keys():
            ax1 = plot_list[k][1][0]
            ax2 = plot_list[k][1][1]
            ax1.legend()
            ax2.legend()
            ax1.axvline(x=threshold, ymax=ax1.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            ax2.axvline(x=0.99, ymax=ax2.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            fig_name = os.path.join(results_dir,name+'_histogram_'+k+'.png')
            plot_list[k][0].savefig(fig_name)
        
        
            
        file_.write('\nthreshold '+str(threshold))       
        file_.close()
        
        cmatrix_mahala = results[name]['confusion_mahala']
        fname = name+'_confusion_mahala.txt'
        file_ = open(os.path.join(results_dir,fname), 'w')
        try:
            file_.write(str(cmatrix_mahala))
        except ValueError,err:
            file_.write('None')
            print err
        '''
        cmatrix_mahala = results[name]['confusion_tot_maha']
        fname = name+'_confusion_total_mahala.txt'
        file = open(os.path.join(results_dir,fname), 'w')
        try:
            file.write(str(cmatrix_mahala))
        except ValueError,err:
            file.write('None')
            print err  
        '''
        
        file_.close()
        
    
        if results[name]['map'] != None:
            
            m_mean = results[name]['map'].pop()
            fname = name+'_mean_map.nii.gz'
            mask_ = m_mean._data != 0
            m_mean._data[mask_ ] = (m_mean._data[mask_ ] - np.mean(m_mean._data[mask_ ]))/np.std(m_mean._data[mask_ ])
            ni.save(m_mean, os.path.join(results_dir,fname))
        
            for map, t in zip(results[name]['map'], results[name]['sensitivities'].sa.targets):
                    cl = '_'.join(t)
                    fname = name+'_'+cl+'_map.nii.gz'
    
                    map._data = (map._data - np.mean(map._data))/np.std(map._data)
                    ni.save(map, os.path.join(results_dir,fname))
    
    
    '''
    End of main for
    '''
                
    from pprint import pprint
    filename_ = open(os.path.join(path, 'distance_means.txt'), 'w')
    pprint(means_s, filename_)
    filename_.close()
    
    plot_list = dict()
    for t in hist_sum[hist_sum.keys()[0]].keys():
        f, ax = plt.subplots(2, 1)
        plot_list[t] = [f, ax]
    
    for k1 in hist_sum:
        
        for k2 in hist_sum[k1]:
            
            for measure in hist_sum[k1][k2]:
                hist_sum[k1][k2][measure] = np.hstack(hist_sum[k1][k2][measure])
                
        
            plot_list[k2][1][0].hist(hist_sum[k1][k2]['dist'], bins=35, label = k1, alpha=0.5)
            plot_list[k2][1][1].hist(hist_sum[k1][k2]['p'], bins=35, label = k1, alpha=0.5)
   
    for k in hist_sum[hist_sum.keys()[0]].keys():
            ax1 = plot_list[k][1][0]
            ax2 = plot_list[k][1][1]
            ax1.legend()
            ax2.legend()
            ax1.axvline(x=threshold, ymax=ax1.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            ax2.axvline(x=0.99, ymax=ax2.get_ylim()[1], color='r', linestyle='--', linewidth=3)
            plot_list[k][0].savefig(os.path.join(path,'total_histogram_'+k+'.png'))
        
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

def write_all_subjects_map(path, dir):
    
    res_path = os.path.join(path, '0_results', dir)
    
    subjects = os.listdir(res_path)
    subjects = [s for s in subjects if s.find('.') == -1]
    
    img_list = []
    i = 0
    list_t = []
    
    min_t = 1000
    max_t = 0
    
    #Get data from path
    for s in subjects:
        
        s_path = os.path.join(res_path, s)
        map_list = os.listdir(s_path)
        
        map_list = [m for m in map_list if m.find('mean_map') != -1]
        
        img = ni.load(os.path.join(s_path, map_list[0]))
        
        #Store maximum and minimum if maps are dimension mismatching
        
        if img.get_data().shape[-1] < min_t:
            min_t = img.get_data().shape[-1]
        
        if img.get_data().shape[-1] > max_t:
            max_t = img.get_data().shape[-1]
            
        img_list.append(img)
        list_t.append(img.get_data().shape[-1])
        i = i + 1
        
    dim_mask = np.array(list_t) == min_t
    img_list_ = np.array(img_list)
    
    #compensate for dimension mismatching
    n_list = []
    if dim_mask.all() == False:
        img_m = img_list_[dim_mask]
        n_list = []
        for img in img_m:
            data = img.get_data()
            for i in range(max_t - data.shape[-1]):
                add_img = np.mean(data, axis=len(data.shape)-1)
                add_img =  np.expand_dims(add_img, axis=len(data.shape)-1)
                data = np.concatenate((data, add_img), axis=len(data.shape)-1)
            n_list.append(data)   
        v1 = np.expand_dims(img_list_[~dim_mask][0].get_data(), axis=0)
    else:
        v1 = np.expand_dims(img_list_[dim_mask][0].get_data(), axis=0)
    
    for img in img_list_[~dim_mask][1:]:
        v1 = np.vstack((v1,np.expand_dims(img.get_data(), axis=0)))
    
    for img_data in n_list:
        v1 = np.vstack((v1,np.expand_dims(img_data, axis=0)))
        
    m_img = np.mean(v1, axis=0)
    
    img_ni = ni.Nifti1Image(m_img.squeeze(), img.get_affine())
    
    filename = os.path.join(res_path, 'all_subjects_mean_map_prova.nii.gz')
    ni.save(img_ni, filename)
    
    return


def update_subdirs(conc_file_list, subj, **kwargs):
    
    for arg in kwargs:
        if (arg == 'sub_dir'):
            sub_dirs = kwargs[arg].split(',')
        
    i = 0
    for directory in conc_file_list:
        s_dir = directory[directory.find(subj)+len(subj)+1:directory.rfind('/')]
        if sub_dirs[i].find('/') != -1 or i > len(sub_dirs):
            sub_dirs.append(s_dir)
        else:
            sub_dirs[i] = s_dir
        i = i + 1
        
    kwargs['sub_dir'] = ','.join(sub_dirs)
    
    return kwargs
            

