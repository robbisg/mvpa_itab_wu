from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.svm.classes import SVR, SVC

from pyitab.io.loader import DataLoader
from pyitab.io.bids import load_bids_dataset
from pyitab.preprocessing.pipelines import PreprocessingPipeline, \
    StandardPreprocessingPipeline
from pyitab.analysis.searchlight import SearchLight

from pyitab.analysis.iterator import AnalysisIterator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.configurator import AnalysisConfigurator

from imblearn.over_sampling import SMOTE
import numpy as np

from imblearn.under_sampling import *
from imblearn.over_sampling import *

conf_file = "/home/carlos/mount/megmri03/fmri/carlo_ofp/ofp.conf"
conf_file = "/media/robbis/DATA/fmri/carlo_ofp/ofp.conf"
#conf_file = "/home/carlos/fmri/carlo_ofp/ofp_new.conf"

if conf_file[1] == 'h':
    from mvpa_itab.utils import enable_logging
    root = enable_logging()


loader = DataLoader(configuration_file=conf_file, task='OFP')
ds = loader.fetch()


return_ = True
ratio = 'auto'

_default_options = {
                       'sample_slicer__evidence' : [[1]],
                       'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                       'balancer__balancer': [AllKNN(return_indices=return_, ratio=ratio),
                                              CondensedNearestNeighbour(return_indices=return_, ratio=ratio),
                                              EditedNearestNeighbours(return_indices=return_, ratio=ratio),
                                              InstanceHardnessThreshold(return_indices=return_, ratio=ratio),
                                              NearMiss(return_indices=return_, ratio=ratio),
                                              OneSidedSelection(return_indices=return_, ratio=ratio),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=5),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=15),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=25),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=35),
                                              RandomUnderSampler(return_indices=return_, ratio=ratio, random_state=45),
                                              RepeatedEditedNearestNeighbours(return_indices=return_, ratio=ratio),
                                              TomekLinks(return_indices=return_, ratio=ratio),
                                              #SMOTE(),
                                              #RandomOverSampler(),
                                            ],                          
                       'cv__n_splits': [7],
                       'analysis__radius':[9],
                        }


_default_config = {
               
                       'prepro':[
                                 'sample_slicer', 
                                 'feature_norm', 
                                 'target_trans', 
                                 'balancer'
                                 ],
                       'target_trans__target':"decision",
                       'balancer__attr':'subject',
                       'imbalancer__attr':'subject',
                       'estimator': [('clf', SVC(C=1, kernel='linear'))],
                       'estimator__clf__C':1,
                       'estimator__clf__kernel':'linear',

                       'cv': StratifiedKFold,
                       'cv__n_splits': 5,

                       'scores' : ['accuracy'],

                       'analysis': SearchLight,
                       'analysis__n_jobs': 15,
                       'cv_attr': 'subject'

                    }


iterator = AnalysisIterator(_default_options, AnalysisConfigurator(**_default_config))
#conf = iterator.next()

for conf in iterator:
    kwargs = conf._get_kwargs()
    a = AnalysisPipeline(conf, name="permut1_balancer_7").fit(ds, **kwargs)
    a.save()


#############################################################
# Haxby dataset #

conf_file = "/media/robbis/DATA/fmri/ds105/ds105.conf"


loader = DataLoader(configuration_file=conf_file, 
                    task='objectviewing',
                    loader='bids',
                    bids_derivatives='fmriprep',
                    )

ds = loader.fetch(prepro=StandardPreprocessingPipeline())
ds = MemoryReducer().transform(ds)

return_ = True
ratio = 'auto'

## Undersamplers
_default_options = {
                       'imbalancer__sampling_strategy' : [0.1, 0.2, 0.3, 0.4, 0.5, 1.],
                       
                       'balancer__balancer': [
                                              AllKNN(return_indices=return_, sampling_strategy=ratio),
                                              CondensedNearestNeighbour(return_indices=return_, sampling_strategy=ratio),
                                              EditedNearestNeighbours(return_indices=return_, sampling_strategy=ratio),
                                              InstanceHardnessThreshold(return_indices=return_, sampling_strategy=ratio),
                                              NearMiss(return_indices=return_, 
                                                       sampling_strategy=ratio,
                                                       n_neighbors=2,
                                                       ),
                                              NearMiss(return_indices=return_, 
                                                       sampling_strategy=ratio,
                                                       n_neighbors=2, 
                                                       version=2,),
                                              NearMiss(return_indices=return_, 
                                                       sampling_strategy=ratio,
                                                       n_neighbors_ver3=2, 
                                                       version=3),                                              
                                              OneSidedSelection(return_indices=return_, sampling_strategy=ratio),
                                              RandomUnderSampler(return_indices=return_, sampling_strategy=ratio, random_state=45),
                                              RepeatedEditedNearestNeighbours(return_indices=return_, sampling_strategy=ratio),
                                              NeighbourhoodCleaningRule(return_indices=return_, sampling_strategy=ratio, n_neighbors=2),
                                              TomekLinks(return_indices=return_, ratio=ratio),
                                              #SMOTE(),
                                              #RandomOverSampler(),
                                            ],                          
                       #'cv__n_splits': [5],
                       
                       'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                       'analysis__radius':[9],
                        }
## Only weights
_default_options = {
                       'imbalancer__sampling_strategy' : [0.1, 0.2, 0.3, 0.4, 0.5, 1.],
                       'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                       #'analysis__radius':[9],
                        }

_default_options = {
                       'imbalancer__sampling_strategy' : [0.3, 0.4, 0.5, 1.],
                       
                       'balancer__balancer': [
                                              SMOTE(k_neighbors=2),
                                              #SMOTENC(categorical_features=[], k_neighbors=2),
                                              ADASYN(n_neighbors=2),
                                              BorderlineSMOTE(k_neighbors=2)
                                              #RandomOverSampler(),
                                            ],                          
                       #'cv__n_splits': [5],
                       
                       'sample_slicer__subject': [[s] for s in np.unique(ds.sa.subject)],
                       'analysis__radius':[9],
                        }


_default_config = {
               
                       'prepro':[
                                 'sample_slicer', 
                                 #'feature_normalizer', 
                                 'imbalancer', 
                                 'balancer'
                                 ],
                       'sample_slicer__targets':["face", "house"],
                       #'balancer__attr':'chunks',
                       'imbalancer__attr': 'chunks',
                       'estimator': [('clf', SVC(C=1, kernel='linear'))],
                       'estimator__clf__C':1,
                       'estimator__clf__kernel':'linear',
                       #'estimator__clf__class_weight': 'balanced',

                       'cv': GroupShuffleSplit,
                       #'cv__n_splits': 5,

                       'scores' : ['accuracy'],

                       'analysis': SearchLight,
                       'analysis__n_jobs': -1,
                       'cv_attr': 'chunks'

                    }

from pyitab.analysis import run_analysis
"""run_analysis(ds, _default_config, _default_options, 
             name="ds105_balancer", subdir="derivatives/mvpa")"""

name="over_balancing"
subdir="derivatives/mvpa"

iterator = AnalysisIterator(_default_options, 
                            AnalysisConfigurator(**_default_config))

errs = [] 
for conf in iterator:
    kwargs = conf._get_kwargs()
    try:
        a = AnalysisPipeline(conf, name=name).fit(ds, **kwargs)
        a.save()
    except Exception as err:
        errs.append(err)

    

    

from pyitab.analysis.results.base import get_searchlight_results, filter_dataframe

path = "/media/robbis/DATA/fmri/ds105/derivatives/mvpa"
name = "ds105_balancer"
df = get_searchlight_results(path, name, field_list=['sample_slicer'])

for balancer in np.unique(df['balancer']):
    for ratio in np.unique(df['sampling_strategy'])[:-1]:
        command = "3dttest++ -setA %s_%s " % (balancer, str(ratio))
        prefix = " -paired -prefix %s/%s_%s" %(path, balancer, str(ratio))
        df_filtered = filter_dataframe(df, balancer=[balancer], sampling_strategy=[ratio])
        
        set_a = " "
        set_b = " -setB base "
        for i, sub in enumerate(np.unique(df_filtered['name'])):
            df_base = filter_dataframe(df, balancer=[balancer], sampling_strategy=[1.0], name=[sub])

            set_a += "sub%s %s'[0]' " %(sub, df_filtered['map'].values[i])
            set_b += "sub%s %s'[0]' " %(sub, df_base['map'].values[0])

        sys_command = command+set_a+set_b+prefix
        print(sys_command)
        os.system(sys_command)



path = "/media/robbis/DATA/fmri/ds105/derivatives/"
name = "ds105_only_weights"
df = get_searchlight_results_bids(path, 
                            field_list=['sample_slicer', 'imbalancer'], 
                            pipeline=name, 
                            filetype="mean")

df = filter_dataframe(df, filetype=["mean"])

for ratio in np.unique(df['sampling_strategy'])[:-1]:
    command = "3dttest++ -setA %s_%s " % ("clf_weight", str(ratio))
    prefix = " -paired -prefix %s/%s_%s" %(path, "clf_weight", str(ratio))
    df_filtered = filter_dataframe(df, sampling_strategy=[ratio])
    
    set_a = " "
    set_b = " -setB base "
    for i, sub in enumerate(np.unique(df_filtered['subject'])):
        df_base = filter_dataframe(df, sampling_strategy=[1.0], subject=[sub])

        set_a += "sub%s %s'[0]' " %(sub, df_filtered['filename'].values[i])
        set_b += "sub%s %s'[0]' " %(sub, df_base['filename'].values[0])

    sys_command = command+set_a+set_b+prefix
    print(sys_command)
    os.system(sys_command)



path = "/media/robbis/DATA/fmri/ds105/derivatives/"
name = "over+balancing"
df = get_searchlight_results_bids(path, 
                                  field_list=['sample_slicer', 'imbalancer', 'balancer'], 
                                  pipeline=name, 
                                  filetype="mean")

df = filter_dataframe(df, filetype=["mean"])

for balancer in np.unique(df['balancer']):
    balancer_txt = balancer[:balancer.find('(')]
    for ratio in np.unique(df['sampling_strategy'])[:-1]:

        command = "3dttest++ -setA %s_%s " % (balancer_txt, str(ratio))
        prefix = " -paired -prefix %s/stats/%s_%s" %(path, balancer_txt, str(ratio))
        df_filtered = filter_dataframe(df, balancer=[balancer], sampling_strategy=[ratio])
        
        set_a = " "
        set_b = " -setB base "
        for i, sub in enumerate(np.unique(df_filtered['subject'])):
            df_base = filter_dataframe(df, balancer=[balancer], sampling_strategy=[1.0], subject=[sub])

            set_a += "sub%s %s'[0]' " %(sub, df_filtered['filename'].values[i])
            set_b += "sub%s %s'[0]' " %(sub, df_base['filename'].values[0])

        sys_command = command+set_a+set_b+prefix
        print(sys_command)



path = "/media/robbis/DATA/fmri/ds105/derivatives/"
name = "unbalanced"
df = get_searchlight_results_bids(path, 
                            field_list=['sample_slicer', 'imbalancer'], 
                            pipeline=name, 
                            filetype="mean")

df = filter_dataframe(df, filetype=["mean"])

for ratio in np.unique(df['sampling_strategy'])[:-1]:
    command = "3dttest++ -setA %s_%s " % ("unbalanced", str(ratio))
    prefix = " -paired -prefix %s/stats/%s_%s" %(path, "unbalanced", str(ratio))
    df_filtered = filter_dataframe(df, sampling_strategy=[ratio])
    
    set_a = " "
    set_b = " -setB base "
    for i, sub in enumerate(np.unique(df_filtered['subject'])):
        df_base = filter_dataframe(df, sampling_strategy=[1.0], subject=[sub])

        set_a += "sub%s %s'[0]' " %(sub, df_filtered['filename'].values[i])
        set_b += "sub%s %s'[0]' " %(sub, df_base['filename'].values[0])

    sys_command = command+set_a+set_b+prefix
    print(sys_command)
    os.system(sys_command)

#################################

df = pd.read_excel("./p-values-table.xlsx", sheet_name="Sheet1")
df_melt = pd.melt(df, id_vars=['Algorithm', 'Type'], 
                  value_vars=[0.1, 0.2, 0.3, 0.4, 0.5], 
                  var_name='ratio', 
                  value_name='pvalue')
df_ = filter_dataframe(df_melt, Type=['B', 'O', 'U', 'W'])
df = filter_dataframe(df, Type=['U', 'O', 'B', 'W'])
df['avg'] = df.mean(numeric_only=True, axis=1)


colors = {
    'O': 'red',
    'W': 'limegreen',
    'U': 'deepskyblue',
    'B': 'gray',
}

colors_list = [colors[v] for v in df.sort_values("avg", ascending=False)['Type'].values]

g = sns.PairGrid(df.sort_values("avg", ascending=False),
                 x_vars=['avg', 0.1, 0.2, 0.3, 0.4, 0.5], 
                 y_vars=["Algorithm"])

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette=colors_list, linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlabel="t-value", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Average", "ratio = 0.1", "ratio = 0.2",
          "ratio = 0.3", "ratio = 0.4", "ratio = 0.5"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_xlim(2., 7.05)

sns.despine(left=True, bottom=True)
pl.tight_layout()
                 
#################################

from collections import Counter

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn import svm
from imblearn.datasets import make_imbalance

print(__doc__)


def plot_decoration(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([-4, 4])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# Generate the dataset
X, y = make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=15)

# Two subplots, unpack the axes array immediately
f, axs = plt.subplots(2, 3)

axs = [a for ax in axs for a in ax]


def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}


multipliers = [1, 0.5, 0.4, 0.3, 0.2, 0.1]
for i, multiplier in enumerate(multipliers):
    ax = axs[i]

    X_, y_ = make_imbalance(X, y, sampling_strategy=ratio_func,
                            **{"multiplier": multiplier,
                               "minority_class": 1})
    ax.scatter(X_[y_ == 0, 0], 
               X_[y_ == 0, 1], 
               label="Class #0", 
               #alpha=0.5, 
               color='c')
    ax.scatter(X_[y_ == 1, 0], 
               X_[y_ == 1, 1], 
               label="Class #1", 
               #alpha=0.5,
               color='salmon')
    
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_, y_)

    xlim = X.min(0)[0], X.max(0)[0]
    ylim = X.min(0)[1], X.max(0)[1]
    xx = np.linspace(xlim[0], xlim[1], 40)
    yy = np.linspace(ylim[0], ylim[1], 40)

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='dimgray', levels=[ 0], alpha=0.7,
           linestyles=['-'])

    ax.set_title('ratio = {}'.format(multiplier))
    if i == 0:
        ax.set_title('Original dataset')

    plot_decoration(ax)

plt.tight_layout()
plt.show()

################################################à
