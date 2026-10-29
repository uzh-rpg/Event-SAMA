"""Convert DSEC real-event slices into the events.h5 layout the Event-SAMA eval
pipeline reads (flat x/y/t/p, native_height/width/fps attrs), and write the
MeViS-style expression json. Events are rectified with DSEC's rectify_map.

Frame k (1-based, name '%05d') sits at t = k * 50 ms in the clip time base,
so a 50 ms pre-roll of events precedes frame 1 (keeps its 10 ms window full).
"""
import json
import os

import h5py
import hdf5plugin  # noqa: F401  (DSEC uses Blosc-compressed datasets)
import numpy as np

SEQ_ROOT = '/iopsstor/scratch/cscs/sfedorchenko/real_events/dsec'
OUT = '/iopsstor/scratch/cscs/sfedorchenko/real_events/converted'
FPS = 20
STEP = 50_000          # us between RGB frames (20 Hz)
PRE = STEP
H, W = 480, 640

# image index = row in <seq>_image_timestamps.txt = png number in the zip
CLIPS = [
    # Expression set fixed BEFORE any model output was seen. Three kinds:
    #  - vehicles/cyclist: move with ego-motion, expected to leave weak events
    #  - static structures: fire strongly under ego-motion, clearly visible
    #  - negative control: object absent from the scene
    dict(clip='zurich13a_cars', seq='zurich_city_13_a', frame_start=40, n=80,
         exps=['the black SUV in the left lane',
               'the car driving ahead of us in the middle of the road',
               'the cyclist on the right side of the road',
               'the overhead sign gantry above the road',
               'the white building on the right side of the street',
               'a horse running across the road']),          # negative control
    dict(clip='zurich13a_brake', seq='zurich_city_13_a', frame_start=200, n=80,
         exps=['the car ahead with its brake lights on',
               'the traffic light',
               'the white building on the right side of the street',
               'a horse running across the road']),          # negative control
    dict(clip='thun01a_parked', seq='thun_01_a', frame_start=30, n=80,
         exps=['the parked grey car on the right',
               'the building on the right side of the street',
               'a horse running across the road']),          # negative control
]


def main():
    os.makedirs(OUT, exist_ok=True)
    videos, meta, stats = {}, {}, {}
    rect_cache = {}
    for c in CLIPS:
        seq, fs, n = c['seq'], c['frame_start'], c['n']
        d = f'{SEQ_ROOT}/{seq}'
        ts = np.loadtxt(f'{SEQ_ROOT}/{seq}_image_timestamps.txt', dtype=np.int64)
        assert fs + n <= len(ts), (seq, fs, n, len(ts))
        t_start = int(ts[fs]) - PRE
        t_end = int(ts[fs + n - 1]) + STEP

        if seq not in rect_cache:
            with h5py.File(f'{d}/events/rectify_map.h5', 'r') as f:
                rect_cache[seq] = f['rectify_map'][:]
        rm = rect_cache[seq]
        assert rm.shape == (H, W, 2), rm.shape

        with h5py.File(f'{d}/events/events.h5', 'r') as f:
            print(seq, 'keys:', list(f.keys()), '| events keys:',
                  list(f['events'].keys()), flush=True)
            toff = int(f['t_offset'][()])
            ms2idx = f['ms_to_idx']
            ms_lo = max(0, (t_start - toff) // 1000 - 2)
            ms_hi = min((t_end - toff) // 1000 + 2, len(ms2idx) - 1)
            lo, hi = int(ms2idx[ms_lo]), int(ms2idx[ms_hi])
            t = f['events/t'][lo:hi].astype(np.int64) + toff
            x = f['events/x'][lo:hi]
            y = f['events/y'][lo:hi]
            p = f['events/p'][lo:hi]

        keep = (t >= t_start) & (t < t_end)
        t, x, y, p = t[keep], x[keep], y[keep], p[keep]
        xy = rm[y.astype(np.int64), x.astype(np.int64)]
        xr = np.rint(xy[:, 0]).astype(np.int64)
        yr = np.rint(xy[:, 1]).astype(np.int64)
        ok = (xr >= 0) & (xr < W) & (yr >= 0) & (yr < H)
        t_rel = (t[ok] - t_start).astype(np.int32)
        xr, yr = xr[ok].astype(np.uint16), yr[ok].astype(np.uint16)
        pol = np.where(p[ok] > 0, 1, -1).astype(np.int8)
        assert np.all(np.diff(t_rel) >= 0), 'events not time-sorted'

        os.makedirs(f'{OUT}/{c["clip"]}', exist_ok=True)
        with h5py.File(f'{OUT}/{c["clip"]}/events.h5', 'w') as g:
            kw = dict(compression='gzip', compression_opts=4)
            g.create_dataset('x', data=xr, **kw)
            g.create_dataset('y', data=yr, **kw)
            g.create_dataset('t', data=t_rel, **kw)
            g.create_dataset('p', data=pol, **kw)
            g.attrs['native_height'] = H
            g.attrs['native_width'] = W
            g.attrs['native_fps'] = float(FPS)

        dur = (t_end - t_start) / 1e6
        centres = np.arange(1, n + 1) * STEP
        cnt = (np.searchsorted(t_rel, centres + 5000) -
               np.searchsorted(t_rel, centres - 5000))
        stats[c['clip']] = dict(
            seq=seq, n_events=int(len(t_rel)), duration_s=dur,
            ev_per_px_per_s=len(t_rel) / (H * W * dur),
            mean_events_per_10ms_window=float(cnt.mean()),
            min_events_per_10ms_window=int(cnt.min()),
            frac_on=float((pol > 0).mean()))
        print(c['clip'], stats[c['clip']], flush=True)

        videos[c['clip']] = dict(
            frames=[f'{k:05d}' for k in range(1, n + 1)],
            expressions={str(i): {'exp': e} for i, e in enumerate(c['exps'])})
        meta[c['clip']] = dict(seq=seq, frame_start=fs, n=n)

    json.dump({'videos': videos}, open(f'{OUT}/meta_expressions.json', 'w'), indent=1)
    json.dump(meta, open(f'{OUT}/clips_meta.json', 'w'), indent=1)
    json.dump(stats, open(f'{OUT}/stats.json', 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
