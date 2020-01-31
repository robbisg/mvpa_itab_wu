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

    r = {'roi':roi,
         'p_omnibus_plain':test_omnibus.pvalue,
         #'p_omnibus_mixed':test_mixed.pvalue,
         'p_linear_plain' :test_linear_linear.pvalue,
         'p_linear_mixed' :test_linear_mixed.pvalue,
         }

    results.append(r)