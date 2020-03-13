
import itertools
import numpy as np
import scipy as sp
from scipy import signal
from pyitab.simulations.models import *

n_nodes = 10
n_brain_states = 6
max_edges = 5
lifetime_range = [2.5, 3.5] # seconds
n_iteration = 1
sample_frequency = 256


model = PhaseDelayedModel(snr=1e10, delay=np.pi*0.5)

edges = [e for e in itertools.combinations(np.arange(n_nodes), 2)]
n_edges = len(edges)

n_seq_states = np.floor(60*sample_frequency*5/(sample_frequency*np.mean(lifetime_range)))
n_seq_states = np.int(60*5/np.mean(lifetime_range))

states = []
for i in range(max_edges):
    states += [e for e in itertools.combinations(np.arange(n_edges), i+1)]

states = np.array(states)


#for i in range(n_iteration):
selected_states = states[np.random.randint(0, len(states), n_brain_states)]
length_bs = np.random.randint(sample_frequency*lifetime_range[0], 
                                sample_frequency*lifetime_range[1], 
                                n_seq_states
                                )

# This is done using Hidden Markov Models but since 
# Transition matrix is uniformly distributed we can use random sequency
#     mc = mcmix(nBS,'Fix',ones(nBS)*(1/nBS));
#     seqBS= simulate(mc,nseqBS-1);
seqBS = np.random.randint(0, n_brain_states, n_seq_states)
bs_dynamics = [] 
for i, time in enumerate(length_bs): 
    for _ in range(time): 
        bs_dynamics.append(seqBS[i])
bs_dynamics = np.array(bs_dynamics)

full_bs_matrix = np.zeros((n_brain_states, n_nodes, n_nodes, model.order))

brain_matrices = []
for j in range(n_brain_states):
    matrix = np.eye(n_nodes)
    selected_edges = selected_states[j]
    for edge in selected_edges:
        matrix[edges[np.array(edge)]] = 1
    pl.figure()
    pl.imshow(matrix)
    brain_matrices.append(matrix)

brain_matrices = np.array(brain_matrices)

for j in range(n_brain_states):
    matrix = np.dstack([brain_matrices[j] for _ in range(model.order)])

    cycles = 0
    FA = np.zeros((n_nodes*model.order, n_nodes*model.order))
    eye_idx = n_nodes*model.order - n_nodes
    FA[n_nodes:, :eye_idx] = np.eye(eye_idx)
    for k in range(8000):
        A = 1/2.5 * np.random.rand(n_nodes, n_nodes, model.order) * matrix
        FA[:n_nodes,:] = np.reshape(A, (n_nodes, -1), 'F')

        eig, _ = sp.linalg.eig(FA)

        if np.all(np.abs(eig) < 1):
            print(k)
            break
    
    if k==8000-1:
        print("Solutions not found")


    full_bs_matrix[j] = A.copy()

data = model.fit(full_bs_matrix, seqBS, length_bs)




print("data "+str(data.shape))
           
"""    
% writing the structure with the useful info
structure.whichBS{iiter}=whichBS;
structure.seqBS{iiter}=seqBS;
structure.lengthBS{iiter}=lengthBS;
structure.probtrans{iiter}=mc.P;
structure.data{iiter}=data;

SNR = 10;
noise = randn(size(data));
datanoisy = data + sqrt(1/SNR)*(noise./std(noise)); 
structure.SNR = SNR;
structure.datanoisy{iiter} = datanoisy;       
"""
window_length = 1 * sample_frequency
print("data "+str(data.shape))

# Single iteration

# Butter filter
"""
iir_params = dict(order=8, ftype='butter')
filt = mne.filter.create_filter(data, sample_frequency, l_freq=6, h_freq=20,
                                method='iir', iir_params=iir_params,
                                verbose=True)

data_filtered = signal.sosfiltfilt(filt['sos'], data.T).T
"""
""" Create Filter Transformer"""
b, a = signal.butter(8, [6, 20], btype='bandpass', fs=sample_frequency)
data_filtered = signal.filtfilt(b, a, data.T)
data = data_filtered
print("data "+str(data.shape))


"""Connectivity"""
connectivity_lenght = data.shape[1] - window_length + 1
timewise_connectivity = np.zeros((n_edges, connectivity_lenght))
for w in range(connectivity_lenght):
    data_window = data[:, w:w+window_length]
    assert data_window.shape[1] == 256
    phi = np.angle(signal.hilbert(data_window))
    for e, (x, y) in enumerate(edges):
        coh = np.imag(np.exp(1j*(phi[x] - phi[y])))
        iplv = np.abs(np.mean(coh))
        timewise_connectivity[e, w] = iplv
    perc = w/connectivity_lenght * 100
    if perc % 1 == 0:
        print(perc)

# Storing structure n_edges x (time - window)
b, a = signal.butter(4, 2, btype='lowpass', fs=sample_frequency)
iplv_filtered = signal.filtfilt(b, a, timewise_connectivity).T
iplv_filtered = np.squeeze(iplv_filtered)
print("timewise_connectivity "+str(timewise_connectivity.shape))
print("iplv_filtered "+str(iplv_filtered.shape))
timewise_connectivity = iplv_filtered
print("timewise_connectivity "+str(timewise_connectivity.shape))
timewise_connectivity = np.expand_dims(timewise_connectivity, axis=0)
print("timewise_connectivity "+str(timewise_connectivity.shape))
#from mvpa_itab.conn.states.subsamplers import *

"""State"""
subsampler = VarianceSubsampler()
X = subsampler.fit_transform(timewise_connectivity)
print("X "+str(X.shape))
X, clustering_ = cluster_state(X, range(2, 10))
    
metrics_, k_step, metrics_keys = calculate_metrics(X, clustering_)

plot_metrics(metrics_, metrics_keys, k_step)
centroid_ = get_centroids(X, clustering_[4])

stat = [];
for i=1:length(structure.seqBS{1})
    stat = [stat (structure.seqBS{1}(i)-1)*ones(1,structure.lengthBS{1}(i))];
end
offset = winleng/2;
stat = stat((offset+1):end-offset);
stat = stat.';

conn = structure.cts{1};
nsec = 60;
n = 256*nsec; figure; plot([1:n]/256,[conn(:,1:n); stat(1:n).'/5] )


stat = [];
for i=1:length(structure.seqBS{1})
    stat = [stat (structure.seqBS{1}(i)-1)*ones(1,structure.lengthBS{1}(i))];
end
offset = winleng/2;
stat = stat((offset+1):end-offset);
stat = stat.'+1;

% find the right estimated state order
UNCI = unique(idx); % uniquely numbered cluster indexes
P = perms(UNCI);
ERR = zeros(size(UNCI,1),1);
for i=1:size(P,1) % for all permutations
    idx_loc = idx;
    for j=1:length(UNCI)
        idx_loc(idx==UNCI(j)) = P(i,j);
    end
    ERR(i)=sum(double( not(idx_loc==stat) ))/numel(idx_loc);
end

% choose the permutation which gives the minimum ERROR
[v,imin] = min(ERR);
idx_loc = idx;
for j=1:length(UNCI)
   idx_loc(idx==UNCI(j)) = P(imin,j);
end
    
figure
nsec = 120;
n = 256*nsec; figure; plot([1:n]/256,[idx_loc(1:n)'; stat(1:n)'] )





%% plots

% example of BSs considered
figure('Position', [100 100 900 900])
for ibs=1:nBS
    subplot(floor(nBS/2),2,ibs)
    WBS=whichBS(ibs,isnan(whichBS(ibs,:))==0);
    G = digraph(1,1);
    for inode=2:nNodes
        G = addedge(G,inode,inode);
    end
    if(length(WBS)>0)
      for il=1:length(WBS)
          G = addedge(G,Edges(WBS(il),1),Edges(WBS(il),2));
      end
    end
    plot(G,'Layout','force')
    axis off
    str = sprintf('Brain state %d', ibs);
    title(str)
end

% graph of the BSs and transition probability matrix
figure('Position', [100 100 900 900])
subplot(2,1,1)
graphplot(mc,'ColorEdges',true);
colormap parula
caxis([0 0.3])
subplot(2,1,2)
imagesc(mc.P);
xlabel('Brain state')
ylabel('Brain state')
title('Transition probability')
colormap parula
caxis([0 0.3])
colorbar

% example of BSs sequence 
figure('Position', [100 100 900 400])
seqplot=[];
for iBS=1:length(lengthBS)
    seqplot=[seqplot,seqBS(iBS)*ones(1,lengthBS(iBS))];
end
plot(1/256:1/256:60,seqplot(1:256*60))
ylabel('Brain state')
xlabel('Sec')

% example of time series realization
figure('Position', [100 100 900 900])
for iplot=1:nNodes
    subplot(nNodes,1,iplot)
    plot(1/256:1/256:60,data(1:256*60,iplot))
    ylabel('Simulated time series')
    xlabel('Sec')
end

% histogram of life time for each BS
figure('Position', [100 100 900 400])
hist(seqBS,10)
ylabel('Occurrency')
xlabel('Brain state')





stat = [];
for i=1:length(structure.seqBS{1})
    stat = [stat (structure.seqBS{1}(i)-1)*ones(1,structure.lengthBS{1}(i))];
end
offset = winleng/2;
stat = stat((offset+1):end-offset);
stat = stat.';

conn = structure.cts{1};
nsec = 60;
n = 256*nsec; figure; plot([1:n]/256,[conn(:,1:n); stat(1:n).'/5] )


stat = [];
for i=1:length(structure.seqBS{1})
    stat = [stat (structure.seqBS{1}(i)-1)*ones(1,structure.lengthBS{1}(i))];
end
offset = winleng/2;
stat = stat((offset+1):end-offset);
stat = stat.'+1;

% find the right estimated state order
UNCI = unique(idx); % uniquely numbered cluster indexes
P = perms(UNCI);
ERR = zeros(size(UNCI,1),1);
for i=1:size(P,1) % for all permutations
    idx_loc = idx;
    for j=1:length(UNCI)
        idx_loc(idx==UNCI(j)) = P(i,j);
    end
    ERR(i)=sum(double( not(idx_loc==stat) ))/numel(idx_loc);
end

% choose the permutation which gives the minimum ERROR
[v,imin] = min(ERR);
idx_loc = idx;
for j=1:length(UNCI)
   idx_loc(idx==UNCI(j)) = P(imin,j);
end
    
figure
nsec = 120;
n = 256*nsec; figure; plot([1:n]/256,[idx_loc(1:n)'; stat(1:n)'] )





%% plots

% example of BSs considered
figure('Position', [100 100 900 900])
for ibs=1:nBS
    subplot(floor(nBS/2),2,ibs)
    WBS=whichBS(ibs,isnan(whichBS(ibs,:))==0);
    G = digraph(1,1);
    for inode=2:nNodes
        G = addedge(G,inode,inode);
    end
    if(length(WBS)>0)
      for il=1:length(WBS)
          G = addedge(G,Edges(WBS(il),1),Edges(WBS(il),2));
      end
    end
    plot(G,'Layout','force')
    axis off
    str = sprintf('Brain state %d', ibs);
    title(str)
end

% graph of the BSs and transition probability matrix
figure('Position', [100 100 900 900])
subplot(2,1,1)
graphplot(mc,'ColorEdges',true);
colormap parula
caxis([0 0.3])
subplot(2,1,2)
imagesc(mc.P);
xlabel('Brain state')
ylabel('Brain state')
title('Transition probability')
colormap parula
caxis([0 0.3])
colorbar

% example of BSs sequence 
figure('Position', [100 100 900 400])
seqplot=[];
for iBS=1:length(lengthBS)
    seqplot=[seqplot,seqBS(iBS)*ones(1,lengthBS(iBS))];
end
plot(1/256:1/256:60,seqplot(1:256*60))
ylabel('Brain state')
xlabel('Sec')

% example of time series realization
figure('Position', [100 100 900 900])
for iplot=1:nNodes
    subplot(nNodes,1,iplot)
    plot(1/256:1/256:60,data(1:256*60,iplot))
    ylabel('Simulated time series')
    xlabel('Sec')
end

% histogram of life time for each BS
figure('Position', [100 100 900 400])
hist(seqBS,10)
ylabel('Occurrency')
xlabel('Brain state')
