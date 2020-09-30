from nilearn.plotting import *
from matplotlib.colors import LinearSegmentedColormap 
from nilearn import surface
from scipy.stats import ttest_1samp
from scipy.signal import find_peaks
import seaborn as sns
import nibabel as ni

from pyitab.analysis.results import *

path = '/media/robbis/Seagate_Pt1/data/carlo_mdm/derivatives/'

dataframe = get_results_bids(path=path,  
                             pipeline="temporal+decoding+across+fsel",
                             field_list=['sample_slicer'], 
                             )

dataframe['value'] = np.int_(dataframe['value'].values)

tasks = np.unique(dataframe['attr'].values)
masks = np.unique(dataframe['mask'].values)



df_diagonal = apply_function(dataframe=df, 
                                keys=['value', 'fold', 'mask', 'attr'], 
                                attr='score_score', 
                                fx=lambda x: np.diagonal(np.mean(np.dstack(x), axis=2)))

df_diagonal = df_diagonal.loc[df_diagonal['mask'] == df_diagonal['attr']]

df_diagonal['peaks'] = [argrelmax(x, mode='wrap')[0] for x in df_diagonal['score_score'].values]

df_diagona 

df_counter = apply_function(df_diagonal, 
                            keys=['value', 'attr', 'mask'], 
                            attr='peaks', 
                            fx= lambda x: np.hstack(x))


#################################################à

dataframe_across = get_results_bids(path=path,  
                             pipeline="temporal+decoding+across+fsel",
                             field_list=['sample_slicer'], 
                             )

dataframe_across['value'] = np.int_(dataframe_across['value'].values)
dataframe_across['experiment'] = ['across' for i in dataframe_across['value'].values]


dataframe_within = get_results_bids(path=path,  
                                    pipeline="temporal+decoding+mdm",
                                    field_list=['sample_slicer'], 
                                    )
dataframe_within['value'] = np.int_(dataframe_within['value'].values)
dataframe_within['experiment'] = ['within' for i in dataframe_within['value'].values]

attr = np.zeros_like(dataframe_within['mask'].values, dtype='U24')


attr_mapper = {
    'resp': 'motor+resp',
    'side': 'target+side',
    'type': 'image+type',
    'decision': 'decision'
}



for k in ['resp', 'side', 'decision', 'type']:
    mask_ = np.logical_not([isinstance(v, float) for v in dataframe_within[k].values])
    attr[mask_] = attr_mapper[k]
    dataframe_within=dataframe_within.drop(k, axis=1)


##########
for strategy in ['within', 'across']:
    if strategy == 'across':
        subject = 'fold'
    else:
        subject = 'subject'
    for task in ['motor+resp', 'decision']:

        df_within = filter_dataframe(dataframe, 
                                    attr=[task], mask=[task],
                                    experiment=[strategy])

        
        df_matrix =  apply_function(dataframe=df_within, 
                                    keys=['value'], 
                                    attr='score_score', 
                                    fx=lambda x: np.mean(np.dstack(x), axis=2))

        data = np.dstack(df_matrix['score_score'].values)
        data = np.diagonal(data)

        for value, datum in enumerate(data):
            print(task, value, argrelmax(datum, order=2, mode='wrap'))

