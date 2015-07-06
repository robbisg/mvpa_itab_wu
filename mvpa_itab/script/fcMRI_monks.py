from scipy.io import savemat
import numpy as np
import os
from mvpa_itab.connectivity import load_matrices, z_fisher, plot_matrix
from nitime.analysis import SeedCorrelationAnalyzer
from nitime.timeseries import TimeSeries
from scipy.stats import ttest_ind
from mvpa_itab.conn.plot import plot_circle
from nipy.algorithms.statistics.empirical_pvalue import fdr_threshold

import itertools
from numpy.random.mtrand import permutation
from mvpa_itab.stats import TTest, permutation_test
from scipy.optimize.minpack import curve_fit
from scipy.stats.stats import zscore
from sklearn.metrics.metrics import mean_squared_error

path = '/media/robbis/DATA/fmri/monks/0_results/'

results_dir = os.listdir(path)
 
results_dir = [r for r in results_dir if r.find('connectivity') != -1 
               and r.find('20150427_19')!=-1]

roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_AAL/atlas90.cod', 
                      delimiter='=',
                      dtype=np.str)

subjects = np.loadtxt('/media/robbis/DATA/fmri/monks/attributes_struct.txt',
                      dtype=np.str)



p_value = np.float(0.05)
for r in results_dir:
    results = load_matrices(os.path.join(path,r), ['Samatha', 'Vipassana', 'Rest'])
    nan_mask = np.isnan(results)
     
    for i in range(len(results.shape) - 2):
        nan_mask = nan_mask.sum(axis=0)
        
    results = results[:,:,:,~np.bool_(nan_mask)]
    rows = np.sqrt(results.shape[-1])
    shape = list(results.shape[:-1])
    shape.append(int(rows))
    shape.append(-1)
    
    results = results.reshape(shape)
    zresults = z_fisher(results)
    zresults[np.isinf(zresults)] = 1
    
    roi_mask = ~np.bool_(np.diagonal(nan_mask))
    
    fields = dict()
    fields['z_matrix'] = zresults
    #fields['network'] = list(roi_list[roi_mask].T[0])
    #fields['roi_name'] = list(roi_list[roi_mask].T[2])
    fields['roi_name'] = list(roi_list[roi_mask].T[1])
    fields['groups'] = list(subjects.T[1])
    fields['level'] = list(np.int_(subjects.T[-1]))
    
    #savemat(os.path.join(path,r,'zcorrelation_matrix.mat'), fields)
    
    ################### Tests ###########################
    
    roi_names = np.array(fields['roi_name'])
    #networks = roi_list[roi_mask].T[-2]   
    networks = ['AAL Atlas 90']
    zmean = zresults.mean(axis=2)
    
    labels = []
    for l in ['Samatha', 'Vipassana', 'Rest']:
        labels += [l for i in range(zmean.shape[1])]
    labels = np.array(labels)
    
    group = []
    for i in range(3):
        group+=fields['groups']
    group = np.array(group)
    
    new_shape = list(zmean.shape[-2:])
    new_shape.insert(0, -1)
    
    zreshaped = zmean.reshape(new_shape)
    
    upper_mask = np.ones_like(zreshaped[0])
    upper_mask[np.tril_indices(zreshaped[0].shape[0])] = 0
    upper_mask = np.bool_(upper_mask)
    
    vipassana = zmean[1]
    samatha = zmean[0]
    rest = zmean[2]
    
    mask_v = labels=='Vipassana'
    #vt, vp = TTest(conditions=['E', 'N']).run(zreshaped[mask_v], group[mask_v])
    
    
    tv, pv = ttest_ind(vipassana[subjects.T[1] == 'E'], 
                     vipassana[subjects.T[1] == 'N'],
                     #equal_var=False,
                     axis=0)
    pv_corrected = fdr_threshold(pv[upper_mask], alpha=p_value)
    
    
    ts, ps = ttest_ind(samatha[subjects.T[1] == 'E'], 
                     samatha[subjects.T[1] == 'N'],
                     #equal_var=False,
                     axis=0)       
    ps_corrected = fdr_threshold(ps[upper_mask], alpha=p_value)
    
    fields['ttest_vipassana_t'] = tv
    fields['ttest_vipassana_p'] = pv
    
    #f = plot_matrix(tv * (pv < 0.01), roi_names, networks)
    f, _ = plot_circle(tv * (pv < pv_corrected), roi_names, None, 
                       n_lines=np.count_nonzero(pv < pv_corrected))
    f.savefig(os.path.join(path,r,'vipassana_t_test_corr_.png'))
    

    fields['ttest_samatha_t'] = ts
    fields['ttest_samatha_p'] = ps
    
    #f = plot_matrix(ts * (ps < 0.01), roi_names, networks)
    f, _ = plot_circle(ts * (ps < ps_corrected), roi_names, None, 
                       n_lines=np.count_nonzero(ps < ps_corrected))
    f.savefig(os.path.join(path,r,'samatha_t_test_corr_.png'))
    
    for level in ['N', 'E']:
        tvr, pvr = ttest_ind(vipassana[subjects.T[1] == level], 
                         rest[subjects.T[1] == level],
                         #equal_var=False,
                         axis=0)
        pvr_corrected = fdr_threshold(pvr[upper_mask], alpha=p_value)
        
        tsr, psr = ttest_ind(samatha[subjects.T[1] == level], 
                         rest[subjects.T[1] == level],
                         #equal_var=False,
                         axis=0)    
        psr_corrected = fdr_threshold(psr[upper_mask], alpha=p_value)
        
        tsv, psv = ttest_ind(samatha[subjects.T[1] == level], 
                         vipassana[subjects.T[1] == level],
                         #equal_var=False,
                         axis=0)
        psv_corrected = fdr_threshold(psv[upper_mask], alpha=p_value)
        
        fields['ttest_samatha_rest_t'] = tsr
        fields['ttest_samatha_rest_p'] = psr
        
        fields['ttest_vipassana_rest_t'] = tvr
        fields['ttest_vipassana_rest_p'] = pvr    
        
        fields['ttest_samatha_vipassana_t'] = tsv
        fields['ttest_samatha_vipassana_p'] = psv
        
        f, _ = plot_circle(tsr * (psr < psr_corrected), roi_names, None, 
                           n_lines=np.count_nonzero(psr < psr_corrected))
        f.savefig(os.path.join(path,r,'samatha_rest_t_test_corr_'+level+'.png'))
        
        f, _ = plot_circle(tvr * (pvr < pvr_corrected), roi_names, None, 
                           n_lines=np.count_nonzero(pvr < pvr_corrected))
        f.savefig(os.path.join(path,r,'vipassana_rest_t_test_corr_'+level+'.png'))
        
        f, _ = plot_circle(tsv * (psv < psv_corrected), roi_names, None, 
                           n_lines=np.count_nonzero(psv < psv_corrected))
        f.savefig(os.path.join(path,r,'samatha_vipassana_t_test_corr_'+level+'.png'))
    pl.close('all')
    ############### Behavioral correlation ###############
    
    bh = TimeSeries(np.int_(subjects[subjects.T[1] == 'E'].T[-1]), sampling_interval=1.)
    ts_s = TimeSeries(samatha[subjects.T[1] == 'E'].T, sampling_interval=1.)
    ts_v = TimeSeries(vipassana[subjects.T[1] == 'E'].T, sampling_interval=1.)
    
    S_s = SeedCorrelationAnalyzer(bh, ts_s)
    S_v = SeedCorrelationAnalyzer(bh, ts_v)
    
    fields['vipassana_expertise_corr'] = S_v.corrcoef
    fields['samatha_expertise_corr'] = S_s.corrcoef       
    
    #f = plot_matrix(S_s.corrcoef * (np.abs(S_s.corrcoef) > 0.6), roi_names, networks)
    f, _ = plot_circle(S_s.corrcoef * (np.abs(S_s.corrcoef) > 0.7), roi_names, None)
    f.savefig(os.path.join(path,r,'samatha_correlation_expertise_0.7.png'))
    
    #f = plot_matrix(S_v.corrcoef * (np.abs(S_v.corrcoef) > 0.6), roi_names, networks)
    f, _ = plot_circle(S_v.corrcoef * (np.abs(S_v.corrcoef) > 0.7), roi_names, None)
    f.savefig(os.path.join(path,r,'vipassana_correlation_expertise_0.7.png'))
    
    savemat(os.path.join(path,r,'all_analysis.mat'), fields)
    pl.close('all')



full_matrices = np.vstack((samatha[subjects.T[1] == level], 
                           vipassana[subjects.T[1] == level]))
labels = np.array(['S' for i in range(full_matrices.shape[0])])
labels[samatha.shape[0]:]='V'

p_ = []
t_ = []
for i in range(1000):
    labels_ = permutation(labels)
    
    samatha_ = full_matrices[labels_ == 'S']
    vipassana_ = full_matrices[labels_ == 'V']
    
    tp, pp = ttest_ind(samatha_, vipassana_, axis=0)
    
    t_.append(tp)
    p_.append(pp)

t_ = np.array(t_)
p_ = np.array(p_)

t_true, p_true = ttest_ind(samatha, vipassana, axis=0)

#print np.count_nonzero(p_true > p_)

## Curve fitting

def sigmoid(x, k, c):
    y = 1 / (1 + np.exp(-k*(x))) + c
    return y


def logarithm(x, a, b):
    y = a + np.log(x + b)
    return y

def exponential(x, k):
    y = np.exp(k * x)
    return y

X = X/X.std(axis=0)
y = y/y.std()

X = zscore(X, axis=0)
y = zscore(y, axis=0)

x_ = np.linspace(-1.5, 1.5, 100)
error_conn = []
mse_ = []
func = [exponential, sigmoid]
shift = 200
for i in range(100):
    mse__ = []
    pl.figure()
    pl.scatter(X[:,shift+i], y)
    for f in func:
        try:
            popt, pcov = curve_fit(f, X[:,shift + i], y)
        except RuntimeError:
            error_conn.append(shift + i)
            continue        
        y_ = f(x_, *popt)
        y_pred = f(X[:,shift + i], *popt)
        
        mse = mean_squared_error(y, y_pred)
        
        mse__.append(mse)
    
        lb = str(f)[str(f).find(' ')+1:str(f).rfind(' ')-3]

        pl.plot(x_, y_, label=lb)
    
    mse_.append(mse__)
    pl.legend()
