import numpy as np, h5py, json
from PIL import Image, ImageDraw
R='/iopsstor/scratch/cscs/sfedorchenko/real_events/ncaltech_converted'
H,W=180,240
vids=['kangaroo_0001','elephant_0001','panda_0001','butterfly_0003']
tiles=[]
for v in vids:
    with h5py.File(f'{R}/{v}/events.h5','r') as f:
        x,y,p=f['x'][:].astype(int),f['y'][:].astype(int),f['p'][:]
    on=np.bincount((y*W+x)[p>0],minlength=H*W).reshape(H,W).astype(float)
    off=np.bincount((y*W+x)[p<=0],minlength=H*W).reshape(H,W).astype(float)
    n=lambda a:a/a.max()*255 if a.max()>0 else a
    img=np.stack([n(off),n(on+off),n(on)],-1).astype(np.uint8)
    cont=np.load(f'{R}/gt/{v}_contour.npy')
    for swap in (False,True):
        im=Image.fromarray(img).resize((W*3,H*3),Image.NEAREST)
        d=ImageDraw.Draw(im)
        pts=[((b if swap else a)*3,(a if swap else b)*3) for a,b in cont]
        d.line(pts+[pts[0]],fill=(255,255,0),width=2)
        d.text((4,2),f'{v} {"(y,x)" if swap else "(x,y)"}  cont x[{cont[:,0].min()},{cont[:,0].max()}] y[{cont[:,1].min()},{cont[:,1].max()}]',fill=(255,255,255))
        tiles.append(np.array(im))
rows=[np.concatenate(tiles[i:i+2],1) for i in range(0,len(tiles),2)]
Image.fromarray(np.concatenate(rows,0)).save('/iopsstor/scratch/cscs/sfedorchenko/real_events/ncalt_gt_check.png')
print('ok')
