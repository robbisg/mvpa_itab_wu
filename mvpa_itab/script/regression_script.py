import numpy as np
import scipy as sp
from mvpa_itab.stats import Correlation, CrossValidation
from mvpa_itab.conn.utils import ConnectivityTest
from sklearn.metrics.metrics import mean_squared_error, r2_score
from mvpa_itab.stats import RegressionPermutation
from mvpa_itab.measure import correlation
from mvpa_itab.results import RegressionAnalysis


class RegressionRunner(object):
    
    def setup(self):
        
        self.analysis = RegressionAnalysis()
    
    def run(self):
        
        self.analysis.setup_analysis(path, roi_list)
        
        

subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)
path = '/media/robbis/DATA/fmri/monks/'
conditions = ['Samatha', 'Vipassana']
group_ = 'E'

## Analysis setup ##
r = '20151030_141350_connectivity_filtered_first_no_gsr_findlab_fmri'
roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                      delimiter=',',
                      dtype=np.str)

style_ = 'Samatha'

cv_repetitions = 250
cv_fraction = 0.5


num_exp_subjects = subjects[subjects.T[1] == group_].shape[0]
cv = ShuffleSplit(num_exp_subjects, n_iter=cv_repetitions, test_size=cv_fraction)
algorithm = SVR(kernel='linear', C=1)


# Load data
conn = ConnectivityTest(path, subjects, r, roi_list)
conn.get_results(conditions)
ds = conn.get_dataset()
ds = ds[np.logical_and(ds.sa.meditation == style_, ds.sa.groups == group_)]

# Select data
X = ds.samples
y = np.float_(ds.sa.expertise)*0.01

# preprocess
X_ = zscore(X, axis=1) # Sample-wise
y_ = zscore(y)

c = Correlation(X_)
corr = c.run(X_, y_)[0]

arg_ = np.argsort(np.abs(corr))[::-1]
arg_ = arg_[:500] # Focusing on first 500 features


# Cross validate
feat_ = []
for i in range(len(arg_)):
    
    X_sel = X_[:,arg_[:i]]
    
    cv_ = CrossValidation(cv, algorithm, error_fx=[mean_squared_error, correlation])
    err = cv_.run(X_sel, y_)
    

    rp = RegressionPermutation(cv_)
    rp.run(X_sel, y_)
    
    feat_.append([err, rp.null_dist])

#####################################
color = np.array(['blue', 
         'red', 
         'gold', 
         'yellow', 
         'violet', 
         'c', 
         'skyblue'])


list_nogsr = os.listdir('/media/robbis/DATA/fmri/monks/0_results/')
list_nogsr = [d for d in list_nogsr if d.find('connectivity') != -1]

list_nogsr_ = [d for d in list_nogsr if d.find('no_gsr') != -1] + \
             [d for d in list_nogsr if d.find('20150323_115') != -1] + \
             [d for d in list_nogsr if d.find('20150414_1837') != -1] + \
             [d for d in list_nogsr if d.find('20150415_1419') != -1] + \
             [d for d in list_nogsr if d.find('20150415_1728') != -1] + \
             [d for d in list_nogsr if d.find('20150415_181') != -1] + \
             [d for d in list_nogsr if d.find('20150427_125') != -1]
             
color_dict = dict(zip(list_nogsr, range(len(list_nogsr))))             
             
             

color_med = {'Vipassana': 'gold', 'Samatha':'silver'}

color_dir = ['blue', 'red'] 


alg = np.array(iterator_setup['learner'], np.str_)
        
fig1 = pl.figure()
a1 = fig1.add_subplot(211)
a2 = fig1.add_subplot(212)

json_format = dict()

lista_res = [np.array(r[1]).squeeze() for r in results]
mega_arr = []
labels = []
for r in results:
    m = alg == np.str(r[0]['learner'])
    ind = np.nonzero(m)
    #if color[ind][0] in ['blue', 'yellow', 'violet']:
    if 'kernel' in r[0]['learner'].get_params().keys():
        arr_ = np.array(r[1]).squeeze()
        
        
        #if r[0]['directory'].find('findlab') == -1 and r[0]['directory'].find('2014') == -1:
            #labels.append(r[0]['directory'])
            #mega_arr.append(arr_[:,:,0].mean(1)[:200])
            
            
        a2.plot(arr_[:,:,1].mean(1), c=colors.cnames.values()[color_dict[r[0]['directory']]],
                alpha=0.7)
        a1.plot(arr_[:,:,0].mean(1), c=colors.cnames.values()[color_dict[r[0]['directory']]],
                alpha=0.7)
            
            #json_format[r[0]['directory']] = arr_[:,:,0].mean(1).tolist()
            
            
fname = os.path.join(path,
                    'result.json')
        
with open(fname, 'w') as fp:
            json.dump(json_format, fp)