skills = {
    'Techniques': {
        'fMRI': 40,
        'MEG': 20,
        'EEG': 25,
        'sMRI': 15,
        'dMRI': 1,
        'TMS': 1
    },
    'Analyses': {
        'Machine Learning': 55,
        'Connectiviy': 30,
        'Linear Models': 25,
        'Inverse Solution': 5,
        'Causal Inference': 5,
        'Nonlinear Dynamics': 5
    },
    'Programming': {
        'Python': 45,
        'Matlab': 20,
        'R': 10,
        'Javascript': 20,
        'C/C++': 20,
        'Java': 15
    },
    'Tools': {
        'Nipy': 50,
        'FSL': 30,
        'AFNI': 35,
        'FieldTrip': 15,
        'SPM': 15,
        'EEGLAB': 15
    }
}

fig, axes = pl.subplots(2, 2, subplot_kw={'polar':True})
theta = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
theta = np.hstack((theta, 0))

colors = pl.cm.get_cmap("Set2", 4)

for i, (skill, value) in enumerate(skills.items()):
    ax = axes[int(i/2), i%2]
    #ax = axes[i]
    values = list(value.values())
    values += values[:1]

    ax.plot(theta, values, '-o', c=colors(i))
    ax.fill(theta, values, alpha=0.4, c=colors(i))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ylabels = np.linspace(0, 1, 5) 
    yticks = ylabels * np.max(values)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(0, np.max(values))
    ax.set_rlabel_position(0)

    ax.set_xticks(theta)
    ax.set_xticklabels(list(value.keys()), fontsize=9)
    ax.set_title(skill, fontsize=14, position=(0.5,1.2))

pl.tight_layout()

#
fig, ax = plt.subplots()

size = 0.3
vals = np.array([[60., 10.], [25., 0.], [5., 0.]])
cmap = pl.get_cmap("tab20b")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

ax.pie(vals.sum(axis=1), radius=1-size, colors=outer_colors, 
       #explode=(0.5, 0.5, 0.5),
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals.flatten(), radius=1, colors=inner_colors, 
       #explode=(0.5, 0.1, 0.1, 0.5, 0, 0.5, 0),
       wedgeprops=dict(width=size, edgecolor='w'))

fig.savefig('prova--.svg')