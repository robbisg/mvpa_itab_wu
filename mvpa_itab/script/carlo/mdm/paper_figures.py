from nilearn.plotting import *
from matplotlib.colors import LinearSegmentedColormap 
from nilearn import surface
from scipy.stats import ttest_1samp
import seaborn as sns
import nibabel as ni

tasks = ['decision', 'image+type', 'motor+resp', 'target+side']

colors = {
    'decision': '#0B775E',
    'motor+resp':'#F2300F',
    'target+side':'#35274A',
    'image+type': '#F2AD00'
    }

cmaps = {
    k:LinearSegmentedColormap.from_list('mdm', ['#696969', v, '#FFFFFF']) for k, v in colors.items()
}

#big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

path = '/media/robbis/DATA/fmri/carlo_mdm/'
atlas = os.path.join(path, '0_results', 'derivatives', 'atlas_3mm_nifti.nii')
atlas = '/media/robbis/DATA/fmri/template_4dfp/4dfp_refdir/nifti/711-2C_111.4dfp.nii'
for task in tasks:
    stats_map_fname = os.path.join(path, '0_results', 'derivatives', task+'_conjunction.nii.gz')

    colormap = ['#696969', colors[task], '#FFFFFF']
    cmap_ = LinearSegmentedColormap.from_list('mdm', colormap)

    stats_map_nii = ni.load(stats_map_fname)
    stats_map = (stats_map_nii.get_data() > 0) * stats_map_nii.get_data()

    stats_img = ni.Nifti1Image(stats_map, stats_map_nii.affine)

    
    f = plot_stat_map(stat_map_img=stats_img, 
                        #bg_img=ni.load(atlas), 
                        cut_coords=np.linspace(-33, 69, 10),
                        threshold=0.,
                        symmetric_cbar=False,
                        black_bg=False,
                        cmap=sns.light_palette(colors[task], reverse=True),
                        display_mode='z',
                        )
    
    """
    big_texture = surface.vol_to_surf(stats_img, big_fsaverage.pial_right)
    

    plot_surf_stat_map(big_fsaverage.infl_right, big_texture, 
                       hemi='left', colorbar=True, colormap=cmap_,
                       title='Surface right hemisphere: fine mesh', 
                       threshold=1., bg_map=big_fsaverage.sulc_right)

    big_texture = surface.vol_to_surf(stats_img, big_fsaverage.pial_left)
    plot_surf_stat_map(big_fsaverage.infl_left, big_texture,
                       colormap=cmap_, hemi='left', colorbar=True, 
                       title='Surface left hemisphere: fine mesh',
                       threshold=1., bg_map=big_fsaverage.sulc_left) 

    """
    f.savefig(os.path.join(path, 'figures', task+'_white.svg'), dpi=200)

########################## Figures evidence #################################

from pyitab.analysis.results import *
path = '/home/robbis/mount/permut1/fmri/carlo_mdm/derivatives/' 

dataframe = get_results_bids(path=path,   
                             pipeline="roi+decoding+across+full", 
                             field_list=['sample_slicer', 'ds.sa.evidence'],  
                            )

dataframe['evidence'] = np.int_([s[1:-1] for s in dataframe['ds.sa.evidence'].values])

tasks = ['decision', 'image+type', 'motor+resp', 'target+side']

colors = {
    'decision': '#0B775E',
    'motor+resp':'#F2300F',
    'target+side':'#35274A',
    'image+type': '#F2AD00'
    }

cmaps = {
    k:LinearSegmentedColormap.from_list('mdm', ['#696969', v, '#FFFFFF']) for k, v in colors.items()
}

fig, axes = pl.subplots(2,2, figsize=(15,15))

dataframe['value'] = np.int_(dataframe['value'].values)

for i, task in enumerate(tasks):
    df = filter_dataframe(dataframe, attr=[task], mask=[task])#, value=[1,2,3,4])
    
    ax = axes[np.int(i/2), i%2]
    ax.axhline(y=0.5, ls=':', c='.5', lw=2.5)
    data = df['score_score'].values
    evidences = df['evidence'].values
    values = df['value'].values
    quote = np.linspace(0.45, 0.6,len(np.unique(values))) [::-1]

    palette = sns.light_palette(colors[task], reverse=True, n_colors=4+len(np.unique(values)))[:-4]
    for j, mask in enumerate(np.unique(values)):
        df_roi = filter_dataframe(df, value=[mask])
        df_avg = df_fx_over_keys(df_roi, attr="score_score", keys=['evidence'], fx=np.mean)
        #ax.scatter(evidences+(0.02*j), data, alpha=0.8, c=np.array([palette[j]]))
        evidences = df_avg['evidence'].values
        scores = df_avg['score_score'].values
        ax.plot(evidences, 
                scores,
                marker='o',
                markersize=12, 
                linewidth=3, 
                color=palette[j])
        area = big_table[task][j][3].strip().replace("  ", " ").split("(")[0]
        ax.text(evidences[-1]+0.2, 
                scores[-1], 
                area, 
                color=palette[j], 
                fontsize=10)
    
    ax.set_title(task)
    ax.set_ylim([0.475, 0.62])
    if task == 'target+side':
        ax.set_ylim([0.475, 0.75])
    else:
        ax.set_yticks(np.arange(0.45, 0.65, 0.05))

    
    ax.set_xlim([0.75, 9.5])
    ax.set_xticks(evidences)
    ax.set_xticklabels(evidences)

    ax.set_xlabel("Evidence")
    ax.set_ylabel("Classification Accuracy")

fig.savefig("/media/robbis/DATA/fmri/carlo_mdm/figures/evidence.svg", dpi=200)

######################### Temporal profiles ##################################
pl.style.use('seaborn-white')
pl.style.use('seaborn-paper')

colors = {
    'decision': '#0B775E',
    'motor+resp':'#F2300F',
    'target+side':'#35274A',
    'image+type': '#F2AD00'
    }

roi = {
    'decision': 10,
    'motor+resp': 6,
    'target+side':5,
    'image+type': 6
}

path = '/home/robbis/mount/permut1/fmri/carlo_mdm/derivatives/'

dataframe = get_results_bids(path=path,  
                             pipeline="temporal+decoding+across+fsel",
                             field_list=['sample_slicer'], 
                             )

dataframe['value'] = np.int_(dataframe['value'].values)

tasks = np.unique(dataframe['attr'].values)
masks = np.unique(dataframe['mask'].values)

statistics = []

for task in tasks:

    df = filter_dataframe(dataframe, attr=[task], mask=[task])
    if df.size == 0:
        continue

    df_diagonal = df_fx_over_keys(dataframe=df, 
                                    keys=['value', 'fold'], 
                                    attr='score_score', 
                                    fx=lambda x: np.diagonal(np.mean(np.dstack(x), axis=2)))



    df_exploded = df_diagonal.explode('score_score')
    n_roi = len(np.unique(df_diagonal['value'])) * len(np.unique(df_diagonal['fold']))
    frames = np.hstack([np.arange(7) for _ in range(n_roi)])

    df_exploded['value'] = np.int_(df_exploded['value'])
    df_exploded['frame'] = frames

    rois = [big_table[task][value-1][3].strip().replace("  ", " ") for value in df_exploded['value'].values]
    df_exploded['roi'] = rois

    nrois = len(np.unique(df_exploded['value'].values))

    #fig, axes = pl.subplots(2, int(np.ceil(nrois/2)), figsize=(15,9))
    fig, axes = pl.subplots(1, nrois, figsize=(nrois*4,4))
    for i, value in enumerate(np.unique(df_exploded['value'].values)):
        #ax = axes[int((i*2)/nrois), i%(int(np.ceil(nrois/2)))]
        ax = axes[i]

        for subj in np.unique(df_exploded['fold'].values):
            
            df_roi = filter_dataframe(df_exploded, value=[value], fold=[subj])
            ax.plot(df_roi['frame'].values, 
                    df_roi['score_score'].values, 
                    c=colors[task], 
                    marker='o',
                    lw=3.,
                    alpha=0.1)
            ax.set_ylim(0.35, 0.75)
        
        df_mean = filter_dataframe(df_exploded, value=[value])
        df_mean = df_fx_over_keys(dataframe=df_mean, 
                                    keys=['value', 'frame'], 
                                    attr='score_score', 
                                    fx=np.mean)
        ax.axhline(y=0.5, ls=':', c='.5', lw=2.5)
        ax.plot(df_mean['frame'].values, 
                df_mean['score_score'].values, 
                c=colors[task], 
                marker='o',
                markersize=10,
                lw=3.)
        
        ax.set_title(df_roi['roi'].values[0], fontfamily='Arial')

        test_values = []
        frame_values = np.unique(df_exploded['frame'].values)
        for f in frame_values:
            df_frame = filter_dataframe(df_exploded, frame=[f], value=[value])
            t, p = ttest_1samp(df_frame['score_score'].values, 0.5)
            test_values.append([t, p])
            if (p*(7*roi[task]) < 0.05 and t > 0):
                key = "task-%s_roi-%s_frame-%s" % (task, str(value), str(f))
                statistics+= [key, t, p*(7*roi[task])]

        test_values = np.array(test_values)

        # Uncorrected
        """
        sign_values = np.logical_and(test_values[:,1] < 0.01,
                                    test_values[:,0] > 0)
        ax.plot(frame_values[sign_values], 
                df_mean['score_score'].values[sign_values], 'o', 
                color='lightgray', markersize=5)  
        """


        # Corrected
        sign_values = np.logical_and(test_values[:,1] < 0.05/(7*roi[task]),
                                    test_values[:,0] > 0)


        ax.plot(frame_values[sign_values], 
                df_mean['score_score'].values[sign_values], 'o', 
                color='white', markersize=5)
        
                         
        
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(np.arange(7))

        ax.set_xlabel("Frame")
        ax.set_ylabel("Classification Accuracy")

    fig.savefig("/media/robbis/DATA/fmri/carlo_mdm/temporal+across+fsel+full_%s_s1.svg" %(task), dpi=200)

#####################################
path = '/home/robbis/mount/permut1/fmri/carlo_mdm/derivatives/'

cmaps = {
    k:LinearSegmentedColormap.from_list('mdm', [v, '#FFFFFF']) for k, v in colors.items()
}

cmaps['decision'] = LinearSegmentedColormap.from_list('mdm', ["#777777", '#FFFFFF'])


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

dataframe_within['attr'] = attr

dataframe = pd.concat([dataframe_across, dataframe_within], axis=0)

selected = [
    {'task':'decision', 'value':3},
    {'task':'decision', 'value':5},
    {'task':'decision', 'value':6},
    {'task':'decision', 'value':7},
    {'task':'decision', 'value':8},
    {'task':'decision', 'value':9},
    {'task':'decision', 'value':10},
    #{'task':'motor+resp', 'value':1}
]
"""
selected = [
    {'task':'decision', 'value':1},
    {'task':'decision', 'value':2},
    {'task':'decision', 'value':4},
]
"""


nroi = {
    'decision': 10,
    'motor+resp': 6,
    'target+side':5,
    'image+type': 6
}

fig, axes = pl.subplots(2, len(selected), figsize=(len(selected)*4,10))
color_within = "#346C73"

for i, roi in enumerate(selected):

    task = roi['task']
    value = roi['value']
    
    palette = sns.light_palette(colors[task], 
                                reverse=True, 
                                n_colors=10)

    df_task = filter_dataframe(dataframe, 
                              mask=[task],
                              value=[value])


    # Extract matrix
    df_within = filter_dataframe(df_task, 
                                 attr=[task],
                                 experiment=['within'])
    df_matrix =  df_fx_over_keys(dataframe=df_within, 
                                 keys=['value', 'subject'], 
                                 attr='score_score', 
                                 fx=lambda x: np.mean(np.dstack(x), axis=2))

    data = np.dstack(df_matrix['score_score'].values)

    t, p = ttest_1samp(data, 0.52, axis=2)
    

    ax = axes[0, i]
    ax.set_title(big_table[task][value-1][3].strip().replace("  ", " "))
    
    ax.axhline(y=0.5, ls=':', c='.5', lw=2.5)
    
    ax.plot(np.diagonal(data.mean(2)), 
            c=color_within, 
            marker='o',
            markersize=10,
            lw=3.)
 
    sign_values = np.logical_and(p < 0.05/(7*nroi[task]),
                                 t > 0)
    sign_diag = np.diagonal(sign_values)
    ax.plot(np.arange(7)[sign_diag],
            np.diagonal(data.mean(2))[sign_diag],
            'o', color='white', markersize=5

    )

    sign_matrix = np.logical_and(p < 0.05/(49*nroi[task]),
                                 t > 0)

    palette_within = sns.light_palette(color_within, 
                                #reverse=True, 
                                n_colors=256,
                                as_cmap=True)

    ax = axes[1, i]
    m = ax.imshow(data.mean(2), 
              origin='lower', 
              cmap=palette_within, 
              vmin=0.5, 
              vmax=0.6)
    coords = np.nonzero(sign_matrix)
    ax.scatter(coords[0], coords[1], color='#FFFFFF', s=40)
    ax.set_ylabel("Training Frame")
    ax.set_xlabel("Testing Frame")

    #fig.colorbar(m, ax=ax)


    df_across = filter_dataframe(df_task, 
                                attr=[task],
                                experiment=['across'])
    df_matrix =  df_fx_over_keys(dataframe=df_across, 
                                 keys=['value', 'fold'], 
                                 attr='score_score', 
                                 fx=lambda x: np.mean(np.dstack(x), axis=2))
    data = np.dstack(df_matrix['score_score'].values)
    
    ax = axes[0, i]
    ax.plot(np.diagonal(data.mean(2)),
            c=palette[5], 
            marker='o',
            markersize=10,
            lw=3.)
    
    t, p = ttest_1samp(data, 0.5, axis=2)
    sign_values = np.logical_and(p < 0.05/(7*nroi[task]),
                                 t > 0)
    sign_diag = np.diagonal(sign_values)
    ax.plot(np.arange(7)[sign_diag],
            np.diagonal(data.mean(2))[sign_diag],
            'o', color='white', markersize=5

    )

    mask = 'decision'
    if task == 'decision':
        mask = 'motor+resp'

    df_across = filter_dataframe(df_task, 
                                attr=[mask],
                                experiment=['across'])
    df_matrix =  df_fx_over_keys(dataframe=df_across, 
                                 keys=['value', 'fold'], 
                                 attr='score_score', 
                                 fx=lambda x: np.mean(np.dstack(x), axis=2))
    data = np.dstack(df_matrix['score_score'].values)
        


    ax = axes[0, i]

    palette = sns.dark_palette(colors[mask], 
                                reverse=True, 
                                n_colors=10)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Classification Accuracy')
    ax.plot(np.diagonal(data.mean(2)),
            c="#f99887", 
            marker='o',
            markersize=10,
            lw=3.)
    

    
    t, p = ttest_1samp(data, 0.5, axis=2)
    sign_values = np.logical_and(p < 0.05/(7*nroi[task]),
                                 t > 0)
    sign_diag = np.diagonal(sign_values)
    ax.plot(np.arange(7)[sign_diag],
            np.diagonal(data.mean(2))[sign_diag],
            'o', color='white', markersize=5

    )

    if task == 'decision':
        ax.set_ylim([0.4, 0.7])
    else:
        ax.set_ylim([0.4, 0.82])

fig.savefig("/media/robbis/DATA/fmri/carlo_mdm/figures/Figure_S2.svg", dpi=200)

    