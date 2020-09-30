import pandas as pd
from pyitab.analysis.results.base import filter_dataframe
from pyitab.analysis.results.dataframe import apply_function
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def find_distance_boundaries(data):

    scene_center = .5*(d['Scena_offset_sec'] - d['Scena_onset_sec'])
    distance_offset = scene_center - d['VAS sec']
    value_click = np.int_(np.sign(distance_offset) == 1)
    return value_click


def windowed_similarity(x, y, window):
    spearman = []
    for i in range(len(x) - window):
        s = spearmanr(x[i:i+window], y[i:i+window])
        spearman.append(s[0])

    return spearman


def bootstrap(x, y, n=100, fx=windowed_similarity, window=10):
    permutations = []
    for p in range(n):
        idx = np.sort(np.random.choice(len(x), size=len(x), replace=True))
        
        spearman = windowed_similarity(x[idx], y[idx], window)
        permutations.append(spearman)
    return permutations

def plot_fit(x, y, ax, linestyle='--', color='gray'):
    from scipy.stats import linregress
    m, b, r, p, s = linregress(x, y)
    ax.plot(x, m*x+b, linestyle=linestyle, c=color, label=r**2)
    ax.legend()



path = "/home/robbis/Dropbox/PhD/experiments/memory_movie/review/data/new/"

pl.style.use("seaborn")

fontsize = 15
style = {
    'figure.figsize': (19, 15),
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,
    'grid.color': 'white',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'ytick.labelsize': fontsize-2,
    'xtick.labelsize': fontsize-2,
    'legend.fontsize': fontsize-5,
    'legend.title_fontsize': fontsize-4,
    'font.size': fontsize,
    'axes.labelsize': fontsize-1,
    'axes.titlesize': fontsize,

    'svg.fonttype':'none'
}

pl.rcParams.update(style)

palette_scatter = LinearSegmentedColormap.from_list("scatter_click", ['#73a87c', '#eba2b6'], N=2)
#palette_scatter = sns.diverging_palette(150, 275, s=80, l=55, n=256, center='dark', as_cmap=True)

experiment_list = ["VAS60old_boundaries", "VAS_INTERO", "VAS_W&G", 
                   "VAS_NEWins", "VAS60new_boundaries"]
filetype = "svg"

spearman = {
    "VAS_INTERO": [0.715170, 0.721362, 0.083591, 0.758514, 0.713106, 0.446852],
    "VAS60old_boundaries": [0.661507, 0.533540, 0.723426, 0.545924, 0.153767, 0.135191],
    "VAS_W&G": [0.678019, 0.649123, 0.723426, 0.420021, 0.114551, 0.884417]
}

limits = {
    "VAS_INTERO": [5250, 20],
    "VAS60old_boundaries": [3600, 20],
    "VAS_W&G": [3600, 15]
}


for experiment in experiment_list[:3]:
    fig = pl.figure()
    grid = pl.GridSpec(8, 3, figure=fig)

    data = pd.read_excel(os.path.join(path, experiment+".xlsx"))

    d = filter_dataframe(data, corresp=[1], **{'IR.ACC':[1]})
    d = d.dropna()

    #### Click distribution ###
    value_click = np.int_(np.sign(d['DIST sec']) == 1)

    #grid = pl.GridSpec(4, 1, top=0.88, bottom=0.11, left=0.15,
    #                        right=0.85, hspace=0.2, wspace=0.2)
    
    ax1 = pl.subplot(grid[:3, 0])
    scatter = ax1.scatter(d['VAS sec'], d['Subject'], 
                        marker='|', 
                        c=value_click, 
                        cmap=palette_scatter)
    l = ax1.vlines(limits[experiment][0], 
                   0.5, limits[experiment][1]+0.5, 
                   linestyles='dashed', color='gray', alpha=0.5)
    handles = scatter.legend_elements()[0]
    labels = ['Anticipated', 'Posticipated']
    legend1 = ax.legend(handles, labels, loc=(.75,.95), title="Response")
    ax1.set_yticks(np.arange(1, 1+np.max(d['Subject'])))
    ax1.set_yticklabels(np.unique(d['Subject']))
    ax1.set_ylabel("Subject")
    ax1.set_title("Click distribution")
        
    ax2 = pl.subplot(grid[3:4, 0], sharex=ax)
    sns.distplot(d['VAS sec'], ax=ax2, bins=100, color='#205d89')
    ax2.set_xlim(-200, 200+np.max(d['VAS_Corr sec']))
    ax1.set_xlim(-200, 200+np.max(d['VAS_Corr sec']))
    #pl.savefig(os.path.join(path, experiment+"_clickdistribution.%s" % (filetype)), dpi=250)
    
    #pl.close()


    ### Distribution of errors ###
    drel_mean = apply_function(d, keys=['VAS_Corr sec'], attr='DIST sec', fx=np.nanmean)
    dabs_mean = apply_function(d, keys=['VAS_Corr sec'], attr='DIST(ABS) sec', fx=np.nanmean)

    color_rel = '#205d89'
    color_abs = '#cf784b'

    # Scatter
    ax3 = pl.subplot(grid[:4, 1])
    ax3.scatter(d['VAS_Corr sec'], d['DIST sec'], alpha=0.2, marker='.', color=color_rel)
    ax3.plot(drel_mean['VAS_Corr sec'], drel_mean["DIST sec"], '-o', c=color_rel, label="Relative")

    ax3.scatter(d['VAS_Corr sec'], d['DIST(ABS) sec'], alpha=0.2, marker='.', color=color_abs)
    ax3.plot(dabs_mean['VAS_Corr sec'], dabs_mean["DIST(ABS) sec"], '-o', c=color_abs, label="Absolute")
    ax3.hlines(0, 0, np.max(d['VAS_Corr sec']), color='black', linestyles="dashed")

    legend = pl.legend(loc=3)
    legend.set_title("Distance")

    #pl.savefig(os.path.join(path, experiment+"_errordistr_points.%s" % (filetype)), dpi=300)
    #pl.savefig(os.path.join(path, experiment+"_errordistr_points.svg"), dpi=150)
    #pl.close()


    # Lines

    # Anova
    dmelt = d.melt(id_vars=['Subject', 'Part'], 
                value_vars=['DIST sec', "DIST(ABS) sec"], 
                value_name='Distance (sec)',
                var_name="Distance"
                )

    ax3 = pl.subplot(grid[:4, 2])
    g = sns.boxenplot(x="Part", 
                y="Distance (sec)", 
                hue="Distance", 
                data=dmelt, 
                dodge=True,
                showfliers=False,
                palette=sns.color_palette([color_rel, color_abs], n_colors=2),
                ax=ax3
                )
    legend = g.axes.legend(loc=3)
    pl.hlines(0, -.5, 5.5, color='dimgray', zorder=5, linestyles="dashed")
    legend.set_title("Distance")
    texts = g.get_legend().get_texts()
    for t, l in zip(texts, ['Relative', 'Absolute']): t.set_text(l)
    #pl.savefig(os.path.join(path, experiment+"_anova_boxen.%s" % (filetype)), dpi=300)
    #pl.close()


    # Scatter distance
    drel_mean['Clip distance from end (sec)'] = np.max(drel_mean['VAS_Corr sec']) - drel_mean['VAS_Corr sec']
    dabs_mean['Clip distance from end (sec)'] = np.max(dabs_mean['VAS_Corr sec']) - dabs_mean['VAS_Corr sec']

    ax4 = pl.subplot(grid[4:,0])
    ax5 = pl.subplot(grid[4:,1])
    if experiment == "VAS_INTERO":
        d_bound = apply_function(d, keys=['DISTBOUNDminima_VASsec', 'testclip'], attr='DIST(ABS) sec', 
                    fx=np.nanmean)
        ax4.scatter(d_bound['DISTBOUNDminima_VASsec'], 
                   d_bound['DIST(ABS) sec'], 
                   marker='o', 
                   color=color_abs)

        plot_fit(d_bound['DISTBOUNDminima_VASsec'],
                 d_bound['DIST(ABS) sec'], ax4)
        ax4.set_xlabel("Distance from bounds (sec)")
        ax4.set_ylabel("Absolute positioning error (sec)")
        #pl.savefig(os.path.join(path, experiment+"_scatter_bound_abs.%s" % (filetype)), dpi=300)
        #pl.close()

        d_vas = apply_function(d, keys=['VAS_Corr sec', 'testclip'], attr='DIST(ABS) sec', fx=np.nanmean)
        d_vas['B/E'] = np.min(np.vstack((d_vas['VAS_Corr sec'], 5250-d_vas['VAS_Corr sec'])), axis=0)
        ax5.scatter(d_vas['B/E'], 
                   d_vas['DIST(ABS) sec'], 
                   marker='o', 
                   color=color_abs)
        plot_fit(d_vas['B/E'],
                 d_vas['DIST(ABS) sec'], ax5)
        ax5.set_xlabel("Distance from beginning/end (sec)")
        ax5.set_ylabel("Absolute positioning error (sec)")
        #pl.savefig(os.path.join(path, experiment+"_scatter_beg-end_abs.%s" % (filetype)), dpi=300)
        #pl.close()
    else:
        ax4.scatter(drel_mean['VAS_Corr sec'],
               drel_mean['DIST sec'],
               marker='o', 
               color=color_rel)

        plot_fit(drel_mean['VAS_Corr sec'],
                 drel_mean['DIST sec'], ax4)
        ax4.set_xlabel("Clip onset (sec)")
        ax4.set_ylabel("Relative positioning error (sec)")
        #pl.savefig(os.path.join(path, experiment+"_scatter_dist_rel.%s" % (filetype)), 
        #            dpi=300)
        #pl.close()


        ax5.scatter(dabs_mean['VAS_Corr sec'], 
                dabs_mean['DIST(ABS) sec'], 
                marker='o', 
                color=color_abs)
        plot_fit(dabs_mean['VAS_Corr sec'],
                 dabs_mean['DIST(ABS) sec'], ax5)
        ax5.set_xlabel("Clip onset (sec)")
        ax5.set_ylabel("Absolute positioning error (sec)")
        #pl.savefig(os.path.join(path, experiment+"_scatter_dist_abs.%s" % (filetype)), dpi=300)
        #pl.close()

    ax6 = pl.subplot(grid[4:,2])
    if experiment in spearman.keys():
        values = spearman[experiment]
        ax6.bar(np.arange(1,7), values, color="silver", width=0.7)
        ax6.set_xlabel("Part")
        ax6.set_ylabel("Spearman's correlation")
        ax6.set_ylim(0, 0.9)
        #pl.savefig(os.path.join(path, experiment+"_spearman.%s" % (filetype)), dpi=300)
        #pl.close()

    pl.tight_layout()
    pl.savefig(os.path.join(path, experiment+"_full.pdf"), dpi=300)

# Figure 3
data = pd.read_csv("/home/robbis/Dropbox/PhD/experiments/memory_movie/review/data/new/confronto.csv")

fig = pl.figure()
grid = pl.GridSpec(1, 2, figure=fig)

ax = pl.subplot(grid[:, 0])

ax.scatter(data['VAS_Corr_90'], data['VAS_90'], c='k')
ax.scatter(data['VAS_Corr_60'], data['VAS_60'], c='r')
plot_fit(data['VAS_Corr_90'],
         data['VAS_90'], ax, linestyle='-', color='k')
plot_fit(data['VAS_Corr_60'],
         data['VAS_60'], ax, linestyle='-', color='r')
data_60 = data.loc[data['VAS_Corr_90']<=3600]
plot_fit(data_60['VAS_Corr_90'],
         data_60['VAS_90'], ax, linestyle='-', color='limegreen')
ax.plot(data['VAS_Corr_90'], data['VAS_Corr_90'], c='tab:blue')
ax.set_xlabel("Actual time (sec)")
ax.set_ylabel("Subjective time (sec)")


data = pd.read_csv("/home/robbis/Dropbox/PhD/experiments/memory_movie/review/data/new/intercept.csv")
ax = pl.subplot(grid[:, 1])
d = data.melt(id_vars=['subject'], var_name='task', value_name='intercept') 
g = sns.barplot(data=d, x='task', y='intercept', palette=['k', 'limegreen', 'r'], ax=ax)
ax.set_xlabel("Task")
ax.set_ylabel("Slope coefficient (b)")
pl.tight_layout()
pl.savefig(os.path.join(path, "Figure3.svg"), dpi=300)

###################    AIP   ###################
experiment_ = "SHERLOCK_DOPPIAVAS"
data = pd.read_excel(os.path.join(path, experiment_+".xlsx"))

for session in [1, 2]:
    experiment = experiment_+"_session_"+str(session)
    d = filter_dataframe(data, corresp=[1], Session=[session], **{'IR.ACC':[1]})
    d = d.dropna()

    #### Click distribution ###
    value_click = np.int_(np.sign(d['DIST sec']) == 1)

    grid = pl.GridSpec(4, 1, top=0.88, bottom=0.11, left=0.15,
                            right=0.85, hspace=0.2, wspace=0.2)
    ax = pl.subplot(grid[:3, 0])
    scatter = ax.scatter(d['VAS sec'], d['Subject'], 
                        marker='|', c=value_click, cmap=palette_scatter)
    handles = scatter.legend_elements()[0]
    labels = ['Anticipated', 'Posticipated']
    legend1 = ax.legend(handles, labels, loc=(1.,.9), title="Response")
    ax.set_yticks(np.arange(1, 1+np.max(d['Subject'])))
    ax.set_yticklabels(np.unique(d['Subject']))
    ax.set_ylabel("Subject")
    ax.set_title("Click distribution")
        
    ax2 = pl.subplot(grid[3, 0], sharex=ax)
    sns.distplot(d['VAS sec'], ax=ax2, bins=100, color='#205d89')
    pl.xlim(-200, 200+np.max(d['VAS_Corr sec']))
    pl.savefig(os.path.join(path, experiment+"_clickdistribution.png"), dpi=250)

    pl.close()


    ### Distribution of errors ###
    drel_mean = apply_function(d, keys=['VAS_Corr sec'], attr='DIST sec', fx=np.nanmean)
    dabs_mean = apply_function(d, keys=['VAS_Corr sec'], attr='DIST(ABS) sec', fx=np.nanmean)

    color_rel = '#205d89'
    color_abs = '#cf784b'

    # Scatter
    pl.scatter(d['VAS_Corr sec'], d['DIST sec'], alpha=0.2, marker='.', color=color_rel)
    #sns.lineplot(x="VAS_Corr sec", y="DIST sec", c=color_rel, marker='o', lw=1.5, data=drel_mean, mew=None)
    pl.plot(drel_mean['VAS_Corr sec'], drel_mean["DIST sec"], '-o', c=color_rel, label="Relative")

    pl.scatter(d['VAS_Corr sec'], d['DIST(ABS) sec'], alpha=0.2, marker='.', color=color_abs)
    #sns.lineplot(x="VAS_Corr sec", y="DIST(ABS) sec", c=color_abs, marker='o', lw=1.5, data=dabs_mean, mew=None)
    pl.plot(dabs_mean['VAS_Corr sec'], dabs_mean["DIST(ABS) sec"], '-o', c=color_abs, label="Absolute")
    pl.hlines(0, 0, np.max(d['VAS_Corr sec']), color='black', linestyles="dashed")

    legend = pl.legend(loc=3)
    legend.set_title("Distance")

    pl.savefig(os.path.join(path, experiment+"_errordistr_points.png"), dpi=300)
    #pl.savefig(os.path.join(path, experiment+"_errordistr_points.svg"), dpi=150)
    pl.close()


        #### Analisi ####
    # Spearman's rank correlation window #
    from scipy.stats import *

    f = d.pivot(index='VAS_Corr sec', columns='Subject', values='VAS sec')
    fmean = f.mean(1)
    window_s = []
    palette = sns.cubehelix_palette(len(np.arange(18, 32, 2)), start=.5, rot=-.75)
    
    x = np.argsort(fmean.index.values)
    y = np.argsort(fmean.values)
    t = fmean.index.values
    
    for w, window in enumerate([20]):

        permutations = bootstrap(x, y, window=window)
        permutations = np.array(permutations)
        sd = np.sum((permutations - permutations.mean(0))**2, axis=0) / 499

        time = [0.5 * (t[i] + t[i+window-1]) for i in range(len(t) - window)]

        pl.plot(time, permutations.T, color=palette[3], label=window, alpha=0.3, lw=0.5)
        pl.plot(time, permutations.mean(0), color=palette[3], label=window, lw=1.9)
    
    pl.xlabel('Clip onset (sec)')
    pl.ylabel("Spearman's rank index")
    #pl.legend()
    #pl.savefig(os.path.join(path, experiment+"_spearman.png"), dpi=150)
    #pl.savefig(os.path.join(path, experiment+"_spearman.svg"), dpi=150)
    #pl.close()