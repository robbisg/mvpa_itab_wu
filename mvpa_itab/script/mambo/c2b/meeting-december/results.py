from pyitab.analysis.results import get_results_bids, filter_dataframe, apply_function
import seaborn as sns
import h5py




path = "/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/"
dataframe = get_results_bids(path, field_list=['sample_slicer','estimator__clf'], pipeline=['cross+session'])



average_df = apply_function(dataframe, ['targets', 'estimator__clf', 'ses'], 
                            attr='score_score', fx= lambda x:np.dstack(x).mean(2))






fname = '/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/sub-106521_ses-02_window-300_powerbox_beta.mat'


fname = '/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/power/sub-109123/sub-109123_ses-02_window-300_band-beta_power.mat'

mat = h5py.File(fname)
t = mat['timevec'][:].T[0]
idx = [0, 50, 100, 150, 200]
idx = [0, 20, 40, 60, 80, 100]
tt = t >= 0
xticklabels = np.array(["{:5.1f}".format(i) for i in t])

conditions = np.unique(average_df['targets'])
sessions = np.unique(average_df['ses'])
limits = (0.4, .95)

occurences = list(itertools.product(conditions, sessions))

for occurence in occurences:

    df = filter_dataframe(average_df, targets=[occurence[0]], ses=[occurence[1]])
    
    fig, axes = pl.subplots(2, 4, sharex=True)
    for r, (_, row) in enumerate(df.iterrows()):
        im = axes[0, r].imshow(row['score_score'], 
                          origin='lower',
                          cmap=pl.cm.magma,
                          vmin=limits[0],
                          vmax=limits[1]
                          )
        clf = row['estimator__clf']
        kernel = ""
        if r>2:
            kernel = clf[clf.find("kernel='"):clf.find(", p")]
        clf = clf[:clf.find("(")]+" "+kernel
        axes[0, r].set_title("%s | %s" % (clf, occurence[0].replace("+", "-")))
        axes[0, r].set_xticks(np.arange(58)[::8])
        axes[0, r].set_xticklabels(xticklabels[::8])
        axes[0, r].set_xlabel('Training time')

        axes[0, r].set_yticks(np.arange(58)[::8])
        axes[0, r].set_yticklabels(xticklabels[::8])
        axes[0, r].set_ylabel('Testing time')
        axes[0, r].vlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')
        axes[0, r].hlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')

        axes[1, r].plot(np.diag(row['score_score']))
        axes[1, r].set_xlabel('Training time')
        axes[1, r].set_ylabel('Classification accuracy')
        axes[1, r].set_ylim(limits)
        axes[1, r].vlines(7.5, limits[0], limits[1], colors='r', linestyles='dashed')

    fig.colorbar(im, ax=axes[:], location='right')


from matplotlib.colors import LinearSegmentedColormap
cmaps = {
    k:LinearSegmentedColormap.from_list('mdm', ['#FFFFFF', v]) for k, v in colors.items()
}


for condition in conditions:

    #condition = occurence[0]
    #session = occurence[1]

    #df = filter_dataframe(average_df, targets=[occurence[0]], ses=[occurence[1]])

    df1 = filter_dataframe(average_df, targets=[condition])
    fig, axes = pl.subplots(3, 4, sharex=True)

    for i, ses in enumerate(sessions):

        cmap = cmaps[ses]

        df = filter_dataframe(df1, ses=[ses])
    
        for r, (_, row) in enumerate(df.iterrows()):
            im = axes[i, r].imshow(row['score_score'], 
                                   origin='lower',
                                   cmap=pl.cm.magma,
                                   vmin=limits[0],
                                   vmax=limits[1])
            clf = row['estimator__clf']
            kernel = ""
            if r>2:
                kernel = clf[clf.find("kernel='"):clf.find(", p")]
            clf = clf[:clf.find("(")]+" "+kernel
            axes[i, r].set_title("%s | %s | %s" % (clf, occurence[0].replace("+", "-"), ses))
            axes[i, r].set_xticks(np.arange(58)[::9])
            axes[i, r].set_xticklabels(xticklabels[::9])
            axes[i, r].set_xlabel('Training time')

            axes[i, r].set_yticks(np.arange(58)[::9])
            axes[i, r].set_yticklabels(xticklabels[::9])
            axes[i, r].set_ylabel('Testing time')
            axes[i, r].vlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')
            axes[i, r].hlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')

            axes[2, r].plot(np.diag(row['score_score']), c=colors[ses])
            axes[2, r].set_xlabel('Training time')
            axes[2, r].set_ylabel('Classification accuracy')
            axes[2, r].set_ylim(limits)
            axes[2, r].vlines(7.5, limits[0], limits[1], colors='r', linestyles='dashed')

    fig.colorbar(im, ax=axes[:], location='right')






g = sns.FacetGrid(average_df, col="targets",  row="estimator__clf")
g.map(imshow, 'score_score', origin='lower')


def imshow(x, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    pl.imshow(x.values[0], origin='lower')
    pl.colorbar()



#############################
