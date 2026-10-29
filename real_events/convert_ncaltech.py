"""N-Caltech101 (real ATIS recordings of monitor-displayed images, Orchard et al. 2015)
-> events.h5 in the Event-SAMA eval layout + expression json + GT contour masks.

Event format (README): 40 bits/event: x (8) | y (8) | polarity (1) | timestamp us (23);
y == 240 marks a timestamp overflow (adds 2**13 us to all later events) and is removed.
Classes/samples are fixed a priori (animals, first 8 samples by file name).
Each sample gets 2 expressions: 'the <class>' and 'the <other class>' (cyclic next class).
"""
import json
import os
import zipfile

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = '/iopsstor/scratch/cscs/sfedorchenko/real_events/ncaltech101'
OUT = '/iopsstor/scratch/cscs/sfedorchenko/real_events/ncaltech_converted'
H, W = 180, 240
FPS = 100.0
FRAMES = [f'{k:05d}' for k in range(3, 31, 3)]          # t = 30..300 ms, 10 frames
CLASSES = ['beaver', 'butterfly', 'crab', 'crocodile', 'dalmatian', 'dolphin', 'dragonfly',
           'elephant', 'emu', 'flamingo', 'hedgehog', 'ibis', 'kangaroo', 'llama', 'lobster',
           'octopus', 'okapi', 'panda', 'pigeon', 'platypus', 'rhino', 'rooster', 'scorpion',
           'starfish', 'Leopards']
N_PER_CLASS = 8


def pretty(c):
    return c.replace('_', ' ').lower().rstrip('s') if c == 'Leopards' else c.replace('_', ' ')


def read_events(buf):
    raw = np.frombuffer(buf, dtype=np.uint8).astype(np.uint32)
    n = len(raw) // 5
    raw = raw[:n * 5]
    x, y = raw[0::5], raw[1::5]
    p = (raw[2::5] & 128) >> 7
    ts = ((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]
    ov = np.where(y == 240)[0]
    for i in ov:
        ts[i:] += 2 ** 13
    keep = y != 240
    return x[keep], y[keep], p[keep], ts[keep].astype(np.int64)


def read_annotation(buf):
    w = np.frombuffer(buf, dtype='<i2')
    n = int(w[1])
    box = w[2:2 + 2 * n].reshape(-1, 2)
    m = int(w[2 + 2 * n + 1])
    cont = w[2 + 2 * n + 2:2 + 2 * n + 2 + 2 * m].reshape(-1, 2)
    return box, cont


def main():
    zev = zipfile.ZipFile(f'{ROOT}/Caltech101.zip')
    zan = zipfile.ZipFile(f'{ROOT}/Caltech101_annotations.zip')
    an_names = {n for n in zan.namelist() if n.endswith('.bin')}
    print('annotation example names:', sorted(an_names)[:2], flush=True)

    def an_path(cls, idx):
        for pre in ('', 'Caltech101_annotations/', 'Caltech101/'):
            n = f'{pre}{cls}/annotation_{idx}.bin'
            if n in an_names:
                return n
        return None

    os.makedirs(f'{OUT}/gt', exist_ok=True)
    videos, meta, bad = {}, {}, 0
    for ci, cls in enumerate(CLASSES):
        other = CLASSES[(ci + 1) % len(CLASSES)]
        names = sorted(n for n in zev.namelist() if n.startswith(f'Caltech101/{cls}/image_') and n.endswith('.bin'))
        for n in names[:N_PER_CLASS]:
            idx = n.rsplit('_', 1)[1][:-4]
            vid = f'{cls}_{idx}'
            x, y, p, ts = read_events(zev.read(n))
            if len(x) < 1000:
                bad += 1
                continue
            order = np.argsort(ts, kind='stable')
            x, y, p, ts = x[order], y[order], p[order], ts[order]
            ok = (x < W) & (y < H)
            x, y, p, ts = x[ok], y[ok], p[ok], ts[ok]
            ts = ts - ts[0]
            os.makedirs(f'{OUT}/{vid}', exist_ok=True)
            with h5py.File(f'{OUT}/{vid}/events.h5', 'w') as g:
                kw = dict(compression='gzip', compression_opts=4)
                g.create_dataset('x', data=x.astype(np.uint16), **kw)
                g.create_dataset('y', data=y.astype(np.uint16), **kw)
                g.create_dataset('t', data=ts.astype(np.int32), **kw)
                g.create_dataset('p', data=np.where(p > 0, 1, -1).astype(np.int8), **kw)
                g.attrs['native_height'] = H
                g.attrs['native_width'] = W
                g.attrs['native_fps'] = FPS
            ap = an_path(cls, idx)
            if ap:
                box, cont = read_annotation(zan.read(ap))
                m = Image.new('L', (W, H), 0)
                ImageDraw.Draw(m).polygon([(int(a), int(b)) for a, b in cont], fill=255)
                m.save(f'{OUT}/gt/{vid}.png')
                np.save(f'{OUT}/gt/{vid}_contour.npy', cont)
            videos[vid] = dict(frames=FRAMES, expressions={
                '0': {'exp': f'the {pretty(cls)}'},
                '1': {'exp': f'the {pretty(other)}'}})
            meta[vid] = dict(cls=cls, other=other, n_events=int(len(x)), dur_ms=float(ts[-1] / 1000),
                             has_gt=bool(ap))
    json.dump({'videos': videos}, open(f'{OUT}/meta_expressions.json', 'w'), indent=1)
    json.dump(meta, open(f'{OUT}/samples_meta.json', 'w'), indent=1)
    ne = np.array([m['n_events'] for m in meta.values()])
    print(f'samples: {len(videos)} (skipped {bad}); events/sample median {int(np.median(ne))}, '
          f'dur_ms median {np.median([m["dur_ms"] for m in meta.values()]):.0f}; '
          f'with GT: {sum(m["has_gt"] for m in meta.values())}', flush=True)


if __name__ == '__main__':
    main()
