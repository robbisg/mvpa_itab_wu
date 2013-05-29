from main_wu import *
from io import *
path = '/media/DATA/fmri/learning'
conf = 'learning.conf'

subjects = os.listdir(path)
subjects = [s for s in subjects if s.find('_') == -1 and s.find('.') == -1]

masks = ['visual', 'll', 'ul', 'lr', 'ur', 'total']

analysis = ['spatial', 'spatiotemporal']

tasks = ['rest', 'task']

results = []

for mask in masks:
    res = analyze(path, subjects, spatiotemporal, tasks[1], conf, mask_area=mask)
    results.append(dict({'mask': mask,
                         'result': res
                         }))
    
##########################################################################

    for subj in subjects:
        try:
            ds_src = load_dataset(data_path, subj, source, **conf_src)
            ds_tar = load_dataset(data_path, subj, target, **conf_tar)
        except Exception, err:
            print err
            continue
        ds_src = preprocess_dataset(ds_src, source, **conf_src)
        ds_tar = preprocess_dataset(ds_tar, target, **conf_tar)

from scipy.spatial.distance import *
from scipy import linalg

conf_task = read_configuration(path, 'learning.conf', 'task')
conf_rest = read_configuration(path, 'learning.conf', 'rest')

for s in subjects:
    ds_task = load_dataset(path, s, 'task', **conf_task)
    ds_rest = load_dataset(path, s, 'rest', **conf_rest)
    
    ds_task = preprocess_dataset(ds_task, 'task', **conf_task)
    ds_rest = preprocess_dataset(ds_rest, 'rest', **conf_rest)
    
    for label in ds_task.targets:
        samples = ds_task.samples[ds_task.targets == label]
        mean = np.mean()
    
###########################################
t_list = []
for s in subjects:
    f_list = os.listdir(os.path.join(data_path, s))
    f_list = [f for f in f_list if f.find('ffa') != -1 and f.find('coll.txt') != -1]
    #t_list.append([s, f_list])
    if len(f_list) == 0:
        continue
    
    if f_list[0].find('3runs_short') != -1:
        exp_end = 815.
    elif f_list[0].find('3runs.') != -1:
        exp_end = 849.
    else:
        exp_end = 498.#FFA
        #exp_end = 566.#LIP
        
    outfile = path+s+'_'+f_list[0][:f_list[0].find('onset')]+'attr.txt'
    fidlfile = os.path.join(data_path, s, f_list[0])
    print exp_end
    fidl2txt_2(fidlfile, outfile, exp_end)
    

    
##########################################################à
for label in np.unique(ds.targets):
    pos_f = pos[ds.targets == label]
    mean = np.mean(pos_f, axis = 0)
    cov = np.cov(pos_f.T)
    robust_cov = MinCovDet().fit(pos_f)
    
    
    a.scatter(pos[ds.targets == label].T[0], pos[ds.targets == label].T[1], color=color[label])
    xx, yy = np.meshgrid(np.linspace(-30, 50, 100),
                     np.linspace(-60, 60, 100))
    
    zz = np.c_[xx.ravel(), yy.ravel()]
    
    mahal_robust_cov = robust_cov.mahalanobis(zz)
    mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
    robust_contour = a.contour(xx[:], yy[:], np.sqrt(mahal_robust_cov)[:],
                                 color=color[label], linestyles='dashed')
    #robust_contour_2 = a.contour(xx, yy, np.sqrt(cov),
    #                             cmap=pl.cm.YlOrBr_r, linestyles='dotted')
    
    
######################## Mean functional images script ######################################

s_tot = 0
l_tot = 0

for exp in ['Carlo_MDM', 'Annalisa_DecisionValue']:
    
    if exp == 'Carlo_MDM':
        path = '/media/DATA/fmri/buildings_faces/Carlo_MDM/'
        
        subjects = os.listdir('/media/DATA/fmri/buildings_faces/Carlo_MDM/0_results/20130309_052128_transfer_learning_L_PPA_saccade/')
    else:
        path = '/media/DATA/fmri/buildings_faces/Annalisa_DecisionValue/'
        subjects = os.listdir('/media/DATA/fmri/buildings_faces/Annalisa_DecisionValue/0_results/20130131_124058_transfer_learning_FFA,PPA_ffa/')
    conf_file = exp+'.conf'
    
    subjects = [s for s in subjects if s.find('.') == -1]
    
    for task in ['face', 'saccade']:
        conf = read_configuration(path, conf_file, task)
    
        for arg in kwargs:
            conf[arg] = kwargs[arg]  
         
        for arg in conf:
            if arg == 'skip_vols':
                skip_vols = np.int(conf[arg])
            if arg == 'use_conc':
                use_conc = conf[arg]
            if arg == 'conc_file':
                conc_file = conf[arg]    
    
            data_path = conf['data_path']    
    
        tot = 0
    
        for subj in subjects:
        
        
            conc_file_list = read_conc(data_path, subj, conc_file)
            conc_file_list = modify_conc_list(data_path, subj, conc_file_list)
            try:
                nifti_list = load_conc_fmri_data(conc_file_list, el_vols = skip_vols, **kwargs)
            except IOError, err:
                print err
                #return 0
            i = 0
            for n in nifti_list:
                if i == 0:
                    a = n.get_data()
                else:
                    a = np.concatenate((a, n.get_data()), axis=3)
                i = i + 1
        
            a = a.mean(axis=3)
        
            s_tot = s_tot + a
        
        l_tot = l_tot + len(subjects)

s_tot = s_tot / l_tot
