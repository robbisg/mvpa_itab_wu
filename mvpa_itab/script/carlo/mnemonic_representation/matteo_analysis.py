import pandas as pd

data = pd.read_excel('/home/robbis/Dropbox/PhD/experiments/memory_movie/VAS_Roberto.xlsx', 
                     sheet_name=None)

########## Statistical tests ###############
from pyitab.analysis.results.base import filter_dataframe
from scipy.stats import ttest_1samp

df = filter_dataframe(data, corresp=[1])

subjects = np.unique(df['Subject'])
movie_clips = np.unique(df['VAS_Corr']) / 6.

timewise_tests = []
for clip in np.unique(df['VAS_Corr']):
    df_ = filter_dataframe(df, VAS_Corr=[clip])

    values = df_['VAS sec'].values
    values = values[np.logical_not(np.isnan(values))]

    popmean = np.mean(df_['VAS_Corr sec'])

    t, p = ttest_1samp(values, popmean)
    timewise_tests.append({'clip':clip, 
                           't':t, 
                           'p':p, 
                           'mean':values.mean(), 
                           'attendend':popmean})


df_test = pd.DataFrame(timewise_tests)

##################### Fit ##################

from scipy.optimize import curve_fit
from sklearn.metrics import *

def linear(x, a):
    return a*x

def linear_intercept(x, a, b):
    return a*x + b

def logaritmic(x, a, b):
    return a*np.log(x)+b

def exponential(x, a, b):
    return -a*np.exp(b*x)

functions = {
    'linear': linear,
    'intercept': linear_intercept,
    'logarithmic': logaritmic,
    #'exponential': exponential,
}

metrics = {
    'r2': r2_score,
    'ev': explained_variance_score,
    'mae': mean_absolute_error
}


import pandas as pd

x_column = 'VAS_Corr sec'
y_column_list = ['VAS_sec', 'DIST sec', 'DIST(ABS) sec']

data = pd.read_excel('/home/robbis/Dropbox/PhD/experiments/memory_movie/VAS_Roberto.xlsx', 
                     sheet_name=None)
fitted_parameters= []
df_total = {}
for y_column in y_column_list:
    fitted_parameters = []
    for experiment, df in data.items():
        
        subjects = np.unique(df['Subject'])

        for subj in subjects:
            item = {}
            df_ = filter_dataframe(df, Subject=[subj])

            args = np.argsort(df_['VAS_Corr'].values)
            
            xdata = df_[x_column].values[args]
            ydata = df_[y_column].values[args]

            mask = np.logical_not(np.isnan(ydata))

            ydata = ydata[mask]
            xdata = xdata[mask]

            item['subject'] = subj
            item['experiment'] = experiment

            for name, fx in functions.items():
                popt, pcov = curve_fit(fx, xdata, ydata)
                item['params_%s' % (name)] = popt

                for metric, score in metrics.items():
                    pp = [ydata, *popt]
                    fdata = fx(*pp)
                    mask = np.logical_or(np.isnan(fdata), 
                                         np.isinf(fdata))
                    mask = np.logical_not(mask)
                    fdata = fdata[mask]
                    ydata = fdata[mask]
                    try:
                        item['%s_%s' % (metric, name)] = score(ydata, fdata)
                    except Exception as _:
                        item['%s_%s' % (metric, name)] = np.nan

            fitted_parameters.append(item.copy())

    df_total[y_column] = pd.DataFrame(fitted_parameters)




import seaborn as sns
colors = sns.color_palette("cubehelix", 20)

############### Plot #######################

fig3d = [pl.subplots(3, 6, figsize=(18,13)) for _ in range(20)]

for i, y_column in enumerate(y_column_list):

    for j, (experiment, df) in enumerate(data.items()):
        #axes[i].plot()
        df_fit = df_total[y_column]
        df_ = filter_dataframe(df_fit, experiment=[experiment])
        
        for k, subj in enumerate(np.unique(df_['subject'])):
            df_data = filter_dataframe(df, Subject=[subj])

            args = np.argsort(df_data['VAS_Corr'].values)
            xdata = df_data[x_column].values[args]
            ydata = df_data[y_column].values[args]
            
            fig, axes = fig3d[k]
            for name, fx in functions.items():

                if name=='exponential':
                    continue

                popt = df_['params_%s'%(name)].values[subj-1]
                pp = [xdata, *popt]
                fdata = fx(*pp)
                if y_column == 'VAS_sec':
                    axes[i, j].plot(xdata, xdata, c='k', lw=5.)
                else:
                    axes[i, j].plot(xdata, np.zeros_like(xdata), c='k', lw=5.)

                
                #
                axes[i, j].set_title("%s | %s" % (y_column, experiment))
                #raise Exception()
                c=[colors[subj-1] for _ in xdata]
                axes[i, j].scatter(xdata, ydata, c=c)
                axes[i, j].plot(xdata, fdata, lw=3)
                #axes[i, j].set_xlim(0)
            
for i, (fig, _) in enumerate(fig3d):
    fig.savefig('/home/robbis/Dropbox/PhD/experiments/memory_movie/fit_subject_%01d.png' % (i+1))


writer = pd.ExcelWriter('/home/robbis/Dropbox/PhD/experiments/memory_movie/fit_data.xlsx', 
                        engine='xlsxwriter')
for key, df in df_total.items():
    df.to_excel(writer, sheet_name=key)



y_column_list = ['DIST sec']

for i, y_column in enumerate(y_column_list):
    
    for j, (experiment, df) in enumerate(list(data.items())[::-1]):
        
        if experiment == 'intero':
            continue
        fig, axes = pl.subplots(4, 5) 
        fig.suptitle("%s | %s" % (experiment, y_column))
        for k, subj in enumerate(np.unique(df['Subject'])):
            df_data = filter_dataframe(df, Subject=[subj])
            
            args = np.argsort(df_data['VAS_Corr'].values)
            xdata = df_data[x_column].values[args]
            ydata = df_data[y_column].values[args]
            c = [colors[j] for _ in xdata]
            axes[int(k/5), k%5].scatter(xdata, ydata, c=c, alpha=0.5)
            

            df_intero = filter_dataframe(data['intero'], Subject=[subj])
            args = np.argsort(df_intero['VAS_Corr'].values)
            xdata = df_intero[x_column].values[args]
            ydata = df_intero[y_column].values[args]            
            c = ['gray' for _ in xdata]
            axes[int(k/5), k%5].scatter(xdata, ydata, c=c, alpha=0.5)

            axes[int(k/5), k%5].vlines(3600, np.nanmin(ydata), np.nanmax(ydata))