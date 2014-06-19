from mne.viz import circular_layout, plot_connectivity_circle
import matplotlib.pyplot as pl
import numpy as np

def plot_matrix(matrix, roi_names, networks):
    
    f = pl.figure()
    
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

def plot_circle(matrix, roi_names, roi_color, n_lines=50):
    
    
    f, sp = plot_connectivity_circle(matrix, 
                                    roi_names, 
                                    n_lines=n_lines, 
                                    node_colors=roi_color, 
                                    facecolor='white', 
                                    textcolor='black',
                                    colormap='RdYlGn',
                                    fontsize_names=14,
                                    node_angles=circular_layout(roi_names, list(roi_names)),
                                    node_edgecolor='white',
                                    colorbar_size=0.5,
                                    fig=pl.figure(figsize=(13,13)),
                                    vmin=-4.5,
                                    vmax=4.5,                          
                                    )
    return f,sp


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