#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

import numpy as np
from matplotlib.pylab import *
import cPickle as pickle
import os


def findTrLearningRes(res, name):
    
    for i in range(len(res)):
        if res[i]['name'] == name:
            return res[i]
        
def extract_p(probabilities):
    
    p_list = []
    
    for p in probabilities:
        p_list.append(p[1][p[0]])
        
    return np.array(p_list)

        
res = pickle.load(open(os.path.join(path, '0_results', '20121114_171626_res_task_visual_transLearn_untrained.pyobj')))

res2 = pickle.load(open(os.path.join(path, '0_results', '20121114_165600_res_task_visual_transLearn_trained.pyobj')))

subplot_l = []
            
values_l = []
names = []
fig = figure(11)
fig12 = figure(12)
ax12 = fig12.add_subplot(111)
ax = fig.add_subplot(111)           
c = 0
        
for i in range(len(res)):
        
        r = res[i]
        
        name = r['name']
        predictions = r['predictions']      
        target = r['targets']
        values = r['values']
        probabilities = r['probabilities']
        
        p = []
              
        for pr,v in zip(probabilities,values):
            p_ = pr[1][pr[0]] * np.sign(v)
            p.append(p_)      
        p = np.array(p) 
        
        
        tar_label = 'RestPost'
        pred_label = 'untrained'
                
        names.append(name)       
        mask_fix = np.array(target == tar_label)        
        zipp = np.array(zip(target, predictions, values, p))        
        filteredPre = zipp[mask_fix]       
        pred = filteredPre.T[1]        
        nPre = np.count_nonzero(np.array(pred == pred_label, dtype = 'int'))        
        perc = float(nPre)/filteredPre.shape[0]        
        print 'untrained vols: '+name + ' ' + str(perc)
        
        
        #############################################################################
        '''Trained Rest Post volumes!'''
        
        r = res2[i]
        
        name = r['name']
        predictions = r['predictions']      
        target = r['targets']
        values = r['values']
        probabilities = r['probabilities']
        
        
        p = []
              
        for pr,v in zip(probabilities,values):
            p_ = pr[1][pr[0]] * np.sign(v)
            p.append(p_)      
        p = np.array(p)
        
        tar_label = 'RestPost'
        pred_label = 'trained'
                
        #names.append(name)       
        mask_fix = np.array(target == tar_label)        
        zipp = np.array(zip(target, predictions, values, p))        
        filteredPre_tr = zipp[mask_fix]       
        pred = filteredPre_tr.T[1]        
        nPre = np.count_nonzero(np.array(pred == pred_label, dtype = 'int'))        
        perc = float(nPre)/filteredPre_tr.shape[0]        
        print '  trained vols: '+name + ' ' + str(perc)
        
        ########################################### 
        '''Plot all subject all runs'''
        #Untrained                                    
        valFil = np.float_(filteredPre.T[2])
        ax.plot(valFil)
        
        #Trained
        valFil_tr = np.float_(filteredPre_tr.T[2])
        ax.plot(valFil_tr)
        
        
        ax.legend(['untrained', 'trained'])
        ############################################
        
        
        values_l.append(valFil)
        prob = np.float_(filteredPre.T[3])
               
        
        '''Plot code for single subject'''                 
        c = c + 1       
        f = figure(c)
        af = f.add_subplot(311)
        #Untrained
        af.plot(valFil)
        #Trained
        af.plot(valFil_tr)
        af.legend(['untrained', 'trained'])

        a40 = f.add_subplot(313)
        
        val_tr = np.maximum(valFil_tr,0)
        val_un = np.maximum(valFil,0)
        plot_v = np.maximum((val_tr - val_un), 0)
        a40.plot(plot_v)
        
        a42 = f.add_subplot(312)
        a42.plot(val_un)
        a42.plot(val_tr)
        
        
        
        
        max = np.max(np.abs(np.float_(valFil))) + 2
        
        '''Code for plotting results'''
        for i in range(3):
            af.fill_between(np.arange(2*i*125, (2*i*125)+125), -max, max,facecolor='yellow', alpha=0.2)
        af.set_title(names[c-1]+' '+str(perc))
        
        '''Plot each run all subjects'''
        for j in range(6):
            n = figure(20+j)
            a = n.add_subplot(111)
            firstVol = j * 125
            lastVol = (j+1) * 125
            a.plot(valFil[firstVol:lastVol])
            a.legend(names)
            a.set_title('Run '+str(j+1))
            
        '''Probabilities'''
        figure_p = figure(30+c)
        
        a_p = figure_p.add_subplot(211)
        mask_pre = target == 'RestPre'
        mask_post = target == 'RestPost'
        
        p_pre = np.float_(zipp[mask_pre].T[3])
        p_post = np.float_(zipp[mask_post].T[3])
        
        a_p.plot(p_pre)
        a_p.plot(p_post)
        a_p.legend(['RestPre', 'RestPost'])
        
        a_positive = figure_p.add_subplot(212)
        p_tr_pre = np.maximum(np.float_(p_pre),0.75)
        p_tr_post = np.maximum(np.float_(p_post),0.75)
        a_positive.plot(p_tr_pre)
        a_positive.plot(p_tr_post)
        a_positive.set_title(name)
           
for i in range(3):
                
    ax.fill_between(np.arange(2*i*125, (2*i*125)+125), -10, 10,facecolor='yellow', alpha=0.2)
            
ax.legend(names)
show()   






############################       MDS         ##############################
color = dict({'trained':'r', 
              'fixation':'b', 
              'RestPre':'y', 
              'RestPost':'g',
              'untrained': 'k'})


color = dict({'0':'r', '1':'b', '2':'y', '3':'g','4': 'k', '5': 'Violet','6': 'k','7': 'k','8': 'k','9': 'k',
              '10': 'k','4': 'k','4': 'k','4': 'k',})


color = ['r', 'b','y','g','k','Violet', 'Orange', 'Grey', 'Navy', 'YellowGreen', 'MidnightBlue', 'Purple ' ]

markers = dict({'trained':'p', 
              'fixation':'s', 
              'RestPre':'^', 
              'RestPost':'o',
              'untrained': '*'})
i = 0
for name in subjects:
    
    ds_task = load_dataset(path, name, 'task', **conf_src)
    ds_task = preprocess_dataset(ds_task, 'task', **conf_src)
    ds_rest = load_dataset(path, name, 'rest', **conf_tar)
    ds_rest = preprocess_dataset(ds_rest, 'rest', **conf_tar)
    
    ds_tot = vstack((ds_task, ds_rest))
    
    mask_trained = ds_tot.targets == 'trained'
    mask_post = ds_tot.targets == 'RestPost'
    mask_tot = mask_trained + mask_post
    
    ds_tot = ds_tot[mask_tot]
    
    dist = squareform(pdist(ds_tot.samples, 'euclidean'))
    pos = mds.fit_transform(dist)
    
    i = i + 1
    f = figure()
    a = f.add_subplot(111)
    
    a.set_title(name)
    j = 0
      
    
    
    for label in np.unique(ds_tot.targets):
        mask = ds_tot.targets == label
        c = pos[mask]
        cov = np.cov(c.T)
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1]/u[0])
        angle = 180*angle/np.pi
        ell = Ellipse(np.mean(c, axis=0), 2*2*np.sqrt(v[0]), 2*2*np.sqrt(v[1]), 180+angle, color=color[label], alpha=0.2)
        ell.set_clip_box(a.bbox)
        a.add_artist(ell)
        
        a.scatter(np.mean(c.T[0]), np.mean(c.T[1]), c=color[label], marker=markers[label], s=110)
        
        if(label == 'RestPost'):
            
            r = findTrLearningRes(res2, name)              
            
            predictions = r['predictions']      
            target = r['targets']
            values = r['values']
            probabilities = r['probabilities']
            
            mask = target == label
            predictions = np.array(predictions)[mask]
            values = values[mask]
            probabilities = np.array(probabilities)[mask]
            target = target[mask]

            for p_label in np.unique(predictions):
                mask_p = predictions == 'trained'
                points = c[mask_p]
                pr_masked = probabilities[mask_p]
                #sizes = np.power(250, 2*(extract_p(pr_masked) - 0.5))
                sizes = (2*(extract_p(pr_masked)-0.5))*100
                a.scatter(points.T[0], points.T[1], c=color[p_label], marker=markers[label], alpha=0.8, s=sizes)
                
        j = j + 1
        
        
    #a.legend(np.unique(ds.targets))
    show()
    #a.set_xlim(-50, 50)
    #a.set_ylim(-50, 50)
    
#------------------------------------------------------------------------------------------------   
i = 0
for name in subjects:
    
    ds_task = load_dataset(path, name, 'task', **conf_src)
    ds_task = preprocess_dataset(ds_task, 'task', **conf_src)
    ds_rest = load_dataset(path, name, 'rest', **conf_tar)
    ds_rest = preprocess_dataset(ds_rest, 'rest', **conf_tar)
    
    ds_tot = vstack((ds_task, ds_rest))   
    
    couples = itertools.combinations(np.unique(ds_tot.targets), r=2)
    dist = squareform(pdist(ds_tot.samples, 'euclidean'))
    pos = mds.fit_transform(dist)
    
    f_tot = figure()
    a_tot = f_tot.add_subplot(111)
    a_tot.imshow(dist)
    f_tot.savefig(name+'_dist_total.png')
    
    
    for pair in couples:
        mask_x = ds_tot.targets == pair[0]
        mask_y = ds_tot.targets == pair[1]
        
        d = dist[mask_y]
        d = d.T[mask_x].T
        
        f_dist = figure()
        af = f_dist.add_subplot(111)
        af.imshow(d)
        f_dist.colorbar(af.get_images()[0])
        af.set_xlabel(pair[0])
        af.set_ylabel(pair[1])
        
        fname = name+'_dist_'+pair[0]+'_vs_'+pair[1]+'.png'
        
        f_dist.savefig(fname)
    
    i = i + 1
    f = figure()
    a = f.add_subplot(111)
    
    a.set_title(name)
    j = 0
    
    
    for label in np.unique(ds_tot.targets):
        mask = ds_tot.targets == label
        c = pos[mask]
        cov = np.cov(c.T)
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1]/u[0])
        angle = 180*angle/np.pi
        ell = Ellipse(np.mean(c, axis=0), 2*2*np.sqrt(v[0]), 2*2*np.sqrt(v[1]), 180+angle, color=color[label], alpha=0.1)
        ell.set_clip_box(a.bbox)
        a.add_artist(ell)
        
        a.scatter(np.mean(c.T[0]), np.mean(c.T[1]), c=color[label], marker=markers[label], s=200)
        fname = name+'_mds_ellipses_'+str(j)+'.png'
        f.savefig(fname)
        a.scatter(c.T[0], c.T[1], c=color[label], marker=markers[label])
        
        f_dist_post = figure()
        a_dist_tr = f_dist_post.add_subplot(121)
        a_dist_fi = f_dist_post.add_subplot(122)
        
        if(label == 'RestPost'):
            
            mask_post = ds_tot.targets == label
            mask_train = ds_tot.targets == 'trained'            
            
            
            d = dist[mask_post]
            d = d.T[mask_train].T
            
            
            
            r = findTrLearningRes(res2, name)              
            
            predictions = r['predictions']      
            target = r['targets']
            values = r['values']
            probabilities = r['probabilities']
            
            mask = target == label
            predictions = np.array(predictions)[mask]
            values = values[mask]
            probabilities = np.array(probabilities)[mask]
            target = target[mask]

            for p_label in np.unique(predictions):
                mask_p = predictions == 'trained'
                points = c[mask_p]
                pr_masked = probabilities[mask_p]
                #sizes = np.power(250, 2*(extract_p(pr_masked) - 0.5))
                sizes = (2*(extract_p(pr_masked)-0.5))*100
                a.scatter(points.T[0], points.T[1], c=color[p_label], marker=markers[label], alpha=0.8, s=sizes)
        
            d_tr = d[mask_p]
            d_fi = d[1 - mask_p]
            
            a_dist_tr.imshow(d_tr)
            a_dist_fi.imshow(d_fi)
            
            f_dist_post.savefig(name+'distance_matrix_trained_fixation.png')
           
        j = j + 1 
        fname = name+'_mds_total.png'  
        f.savefig(fname)
