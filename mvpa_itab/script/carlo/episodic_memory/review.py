from pyitab.analysis.results import get_results, df_fx_over_keys,  \
             get_permutation_values, filter_dataframe

import statsmodels.api as sm

dataframe = get_results('/media/robbis/DATA/fmri/carlo_ofp/0_results/review/', 
                          dir_id="across", 
                          field_list=['sample_slicer'],
                          )


df_lateral_ips = filter_dataframe(dataframe, permutation=[0], roi=['lateral_ips'])

key_across = ['roi_value', 'evidence', 'fold']
key_within = ['roi_value', 'evidence','subject']


df_lips_avg = df_fx_over_keys(df_lateral_ips, 
                              keys=key_across,
                              attr=['score_accuracy'],
                              fx=np.mean)


key_value = 'accuracy'

results = []

dataframe_test = df_lips_avg
#dataframe_test = df_old

for roi in np.unique(dataframe_test['roi_value'].values):
    df_roi = filter_dataframe(dataframe_test, roi_value=[roi])
    variables = np.array([df_roi[key].values for key in ['evidence', 'fold']]).T
    X, factors, n_factor = design_matrix(variables)  
    evidence = (np.int_(df_roi['evidence'].values[:,np.newaxis]) - 1.5)
    X_mixed = np.hstack((X, evidence))
    X_mixed = X_mixed[:,3:]
    y = (df_roi[key_value].values - .48)
    res_omnibus = sm.OLS(y, X).fit()
    res_linear = sm.OLS(y, evidence.flatten()).fit()
    res_mixed = sm.OLS(y, X_mixed).fit()

    contrast_omnibus = build_contrast(n_factor, 0, const_value=0.)
    #n_factor_mixed = np.append(n_factor, 1)
    #contrast_mixed = build_contrast(n_factor_mixed, 0, const_value=0.5)

    t_contrast_mixed = np.zeros_like(X_mixed[0])
    t_contrast_mixed[-1] = 1

    t_contrast_linear = [1]


    test_omnibus = res_omnibus.f_test(contrast_omnibus)
    #test_mixed = res_mixed.f_test(contrast_mixed)


    test_linear_mixed = res_mixed.t_test(t_contrast_mixed)
    test_linear_linear = res_linear.t_test(t_contrast_linear)

    r = {'roi': roi,
         'p_omnibus_plain':test_omnibus.pvalue,
         #'p_omnibus_mixed':test_mixed.pvalue,
         'p_linear_plain' :test_linear_linear.pvalue,
         'p_linear_mixed' :test_linear_mixed.pvalue,
         }

    results.append(r)



####### old results ######

import pickle
with open(os.path.join("/home/robbis/mount/permut1/fmri/carlo_ofp/0_results/within_detrending_total_zscore_roi_5_fold.pickle"), 'rb') as input_:
    result_dict = pickle.load(input_, encoding='latin1')

df_array = []
for subject, results in result_dict.items():
    r = dict()
    r['subject'] = subject.decode('ascii')
    for j, ev in enumerate([1,2,3]):
        r['evidence'] = ev
        for kroi, data in results[j].items():
            roi = np.float(kroi.split("_")[-1])
            accuracy = np.mean(data[0]['test_accuracy'])

            r['roi_value'] = roi
            r['accuracy'] = accuracy

            df_array.append(r.copy())





with open(os.path.join("/home/robbis/mount/permut1/fmri/carlo_ofp/0_results/across_lateral_ips_balance_subject.pickle"), 'rb') as input_:
    result_dict = pickle.load(input_, encoding='latin1')

df_array = []
for j, ev in enumerate([1,2,3]):
    r = dict()
    r['evidence'] = ev
    for kroi, data in result_dict[j].items():
        data = data[0]
        roi = np.float(kroi.split("_")[-1])
        r['roi_value'] = roi
        for i, accuracy in enumerate(data['test_accuracy']):
            r['subject'] = i+1
            r['accuracy'] = accuracy
            
            df_array.append(r.copy())

df_old = pd.DataFrame(df_array)

######################
# B - 1x within / across conjunction ROI > chance
from scipy.stats import ttest_1samp

#analysis = 'across'
analysis = 'within'

dataframe = get_results('/home/robbis/mount/permut1/fmri/carlo_ofp/0_results/review/', 
                          dir_id=analysis, 
                          field_list=['sample_slicer'],
                          filter={'roi':['%s_conjunction'%(analysis)]}
                          )


df_conjunction = filter_dataframe(dataframe, 
                                  permutation=[0], 
                                  roi=['%s_conjunction'%(analysis)])


key_across = ['roi_value', 'fold']
#key_within = ['roi_value', 'subject']
key_value = 'score_accuracy'

df_conj_avg = df_fx_over_keys(df_conjunction, 
                              keys=key_within,
                              attr=['score_accuracy'],
                              fx=np.mean)
results = []
for roi in np.unique(df_conj_avg['roi_value'].values):

    df_roi = filter_dataframe(df_conj_avg, 
                              roi_value=[roi])

    variables = np.array([df_roi[key].values for key in ['fold']]).T
    X, factors, n_factor = design_matrix(variables)
    mean_ = np.ones_like(X[:,0])[:,np.newaxis]
    X = np.hstack((X, mean_)) 
    y = (df_roi[key_value].values)
    res_test = sm.OLS(y, mean_).fit()

    t_contrast_mixed = np.zeros_like(X[0])
    t_contrast_mixed[-1] = 1

    t, p = ttest_1samp(y, 0.48)
    #test_ttest = res_test.t_test(t_contrast_mixed)

    r = {'roi':roi,
         'p_ols':res_test.pvalues,
         'p_ttest': p,
         'avg':y.mean(0)
         }

    results.append(r)


#####








