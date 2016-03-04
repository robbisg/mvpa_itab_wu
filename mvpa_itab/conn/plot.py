from mne.viz import *
from mne.viz.circle import _plot_connectivity_circle_onpick
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(matrix, roi_names, networks):
    
    f = plt.figure(figsize=(12.0, 10.0))
    
    a = f.add_subplot(111)
    
    ax = a.imshow(matrix, interpolation='nearest')
    

    min_ = -0.5
    max_ = len(networks) + 0.5
    ### Draw networks separation lines ###
    network_ticks = [] 
    network_name, indices = np.unique(networks, return_index=True)
    counter = -0.5
    for net in np.unique(networks):
        
        items = np.count_nonzero(networks == net)
        counter = counter + items
        
        network_ticks.append((counter + 0.5) - (items * 0.5)) 
        
        a.axvline(x=counter, ymin=min_, ymax=max_)
        a.axhline(y=counter, xmin=min_, xmax=max_)
    
    
    a.set_yticks(network_ticks)
    a.set_yticklabels(np.unique(networks))
    
    a.set_xticks(network_ticks)
    a.set_xticklabels(np.unique(networks), rotation='vertical')
    
    f.colorbar(ax)
    return f



def plot_cross_correlation(xcorr, t_start, t_end, labels):


    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    dim = len(labels)
    
    
    fig = plt.figure()
    ax = plt.axes(xlim=(-0.5, dim-0.5), ylim=(dim-0.5, -0.5))
    
    
    #im = ax.imshow(xcorr.at(t_start), interpolation='nearest', vmin=-1, vmax=1)
    im = ax.imshow(np.eye(dim), interpolation='nearest', vmin=-4, vmax=4)
    title = ax.set_title('')
    xt = ax.set_xticks(np.arange(dim))
    xl = ax.set_xticklabels(labels, rotation='vertical')
    yt = ax.set_yticks(np.arange(dim))
    yl = ax.set_yticklabels(labels)
    fig.colorbar(im)

    l_time = np.arange(-50, 50, 1)
    mask = (l_time >= t_start) * (l_time<=t_end)
    
    def init():
        im.set_array(np.eye(dim))
        title.set_text('Cross-correlation at time lag of '+str(t_start)+' TR.')
        plt.draw()
        return im, title
        
    
    
    def animate(i):
        global l_time        
        j = np.int(np.rint(i/20))
        print l_time[mask][j]
        #im.set_array(xcorr.at(l_time[j]))
        im.set_array(x[mask][j])
        title.set_text('Cross-correlation at time lag of '+str(l_time[mask][j])+' TR.')
        plt.draw()
        return im, title

    ani = animation.FuncAnimation(fig, animate, 
                                  init_func=init, 
                                  frames=20*(t_end-t_start), 
                                  interval=10,
                                  repeat=False, 
                                  blit=True)
    plt.show()
    #ani.save('/home/robbis/xcorrelation_.mp4')
    
def plot_dendrogram(dendrogram, dissimilarity_matrix):
    
    return


def plot_connectivity_circle_edited(con, node_names, indices=None, n_lines=None,
                                     node_angles=None, node_width=None,
                                     node_size=200, con_thresh=None,
                                     node_colors=None, facecolor='black',
                                     textcolor='white', node_edgecolor='black',
                                     linewidth=1.5, colormap='hot', vmin=None,
                                     vmax=None, colorbar=True, title=None,
                                     colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                                     fontsize_title=12, fontsize_names=8,
                                     fontsize_colorbar=8, padding=6.,
                                     fig=None, subplot=111, interactive=True):
    """Visualize connectivity as a circular graph.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.loria.fr/~rougier/coding/recipes.html

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    n_lines : int | None
        If not None, only the n_lines strongest connections (strength=abs(con))
        are drawn.
    node_angles : array, shape=(len(node_names,)) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    node_width : float | None
        Width of each node in degrees. If None, "360. / len(node_names)" is
        used.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    interactive : bool
        When enabled, left-click on a node to show only connections to that
        node. Right-click shows all connections.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches
    from mne.externals.six import string_types
    from mne.fixes import normalize_colors
    from functools import partial
    
    n_nodes = len(node_names)

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)

    if node_width is None:
        node_width = 2 * np.pi / n_nodes
    else:
        node_width = node_width * np.pi / 180

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
    else:
        # assign colors using colormap
        node_colors = [plt.cm.spectral(i / float(n_nodes))
                       for i in range(n_nodes)]

    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part
        indices = np.tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')


    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        if con_thresh != None:
            con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        if con_thresh == None:
            con_thresh = 0.
    
    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    con_abs = con_abs[sort_idx]
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initalize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    
    
    # get the colormap
    if isinstance(colormap, string_types):
        colormap = plt.get_cmap(colormap)

    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(8, 8), facecolor=facecolor)

    # Use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, polar=True, axisbg=facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additonal space if requested
    #plt.ylim(0, 10 + padding)
    
    
    
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start])
                           / float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end])
                         / float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange

    # Finally, we draw the connections
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 1.2

        # End point
        t1, r1 = node_angles[j], 1.2

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), (t0, 0.5), (t1, 0.5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE4, m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=linewidth, alpha=1., zorder=0)
        axes.add_patch(patch)

    # Draw ring with colored nodes
    height = np.ones(n_nodes) * 1.2
    
    
    #mean_ = axes.scatter(node_angles, height, s=9, c='red', zorder=2, alpha=0.8, edgecolors='red')
    bars = axes.scatter(node_angles, height, s=node_size, c=node_colors, zorder=1, alpha=0.9, linewidths=2, facecolor='.9')
    axes.set_ylim(0, 2)
    
    '''
    bars = axes.bar(node_angles, height, width=node_width, bottom=9,
                    edgecolor=node_edgecolor, lw=2, facecolor='.9',
                    align='center')
    
    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)
    '''
    # Draw node labels
    angles_deg = 180 * node_angles / np.pi
    for name, angle_rad, angle_deg in zip(node_names, node_angles, angles_deg):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'

        axes.text(angle_rad, 1.25, name, size=fontsize_names,
                  rotation=angle_deg, rotation_mode='anchor',
                  horizontalalignment=ha, verticalalignment='center',
                  color=textcolor)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes)

    if colorbar:
        norm = normalize_colors(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array(np.linspace(vmin, vmax))
        cb = plt.colorbar(sm, ax=axes, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    #Add callback for interaction
    
    if interactive:
        callback = partial(_plot_connectivity_circle_onpick, fig=fig,
                           axes=axes, indices=indices, n_nodes=n_nodes,
                           node_angles=node_angles)

        fig.canvas.mpl_connect('button_press_event', callback)

    return fig, axes