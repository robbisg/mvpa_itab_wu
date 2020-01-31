import h5py 
data = {} 
f = h5py.File('/home/robbis/Dropbox/C2B_brain_states/presentations/newstruct.mat') 
for k, v in f.items(): 
    if k!='#refs#': 
        for kk, vv in v.items(): 
            data[kk] = np.array(vv)    
         
f.close()

bs_dynamics = np.hstack([np.full(int(data['lengthBS'].T[i]), int(data['seqBS'].T[i])) for i in range(300)])


bs_matrix = []
for j in range(10):
    bss = data['whichBS'].T[j]
    matrix = np.eye(4)
    for bs in bss:
        if not np.isnan(bs):
            x, y = np.int_(data['edges'].T[int(bs-1)])-1
            matrix[x,y] = 1
    
    bs_matrix.append(matrix)




