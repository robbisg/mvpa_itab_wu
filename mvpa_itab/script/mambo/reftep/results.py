from pyitab.analysis.results import get_results_bids, filter_dataframe, apply_function
import seaborn as sns
import h5py

pipeline = 'reftep+iplv+singleregression'
pipeline = 'reftep+power+singleregression'

path = "/media/robbis/DATA/meg/reftep/derivatives/"
dataframe = get_results_bids(path, 
                field_list=['sample_slicer','estimator__clf', 'target_transformer__attr'], 
                pipeline=['reftep+aal+singleregression'],
                scores=['r2', 'explained_variance'])

mask = np.logical_not(np.isnan(dataframe['score_explained_variance'] ))
dataframe['score_r2'].loc[mask] = np.nan


average_df = apply_function(dataframe, ['win', 'band', 'target_transformer__attr', 'estimator__clf'], 
                            attr='score_r2', fx= lambda x:np.dstack(x).nanmean(2))

grid = sns.FacetGrid(dataframe, col="estimator__clf", hue="target_transformer__attr", palette="tab20c",
                     col_wrap=4, height=1.5)
grid.map(pl.plot, "fold", 'score_r2')