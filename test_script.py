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
