import numpy as np, h5py, json, glob
R='/iopsstor/scratch/cscs/sfedorchenko/real_events/ncaltech_converted'
H,W=180,240
vids=sorted(json.load(open(f'{R}/meta_expressions.json'))['videos'])
rng=np.random.RandomState(0); pick=rng.choice(len(vids),30,replace=False)
rows=[]
for i in pick:
    v=vids[i]
    with h5py.File(f'{R}/{v}/events.h5','r') as f:
        x,y=f['x'][:].astype(int),f['y'][:].astype(int)
    c=np.bincount(y*W+x,minlength=H*W)
    act=c[c>0]
    rows.append((c.max(), np.percentile(act,50), np.percentile(act,99), (c>50).sum(), (c>100).sum(), (c>200).sum(), (c>400).sum(), len(x), (c[c>200]).sum()/len(x)))
a=np.array(rows)
print('per-pixel total event count over the 300 ms recording, 30 random samples (mean over samples):')
for n,col in zip(['max','median active px','p99 active px','#px>50','#px>100','#px>200','#px>400','n_events','frac of events from px>200'],a.T):
    print(f'  {n:28s} mean={col.mean():10.2f}  min={col.min():10.2f}  max={col.max():10.2f}')
