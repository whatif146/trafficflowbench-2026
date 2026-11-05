"""Compute the exact S_FD of the observed field (per regime) from cached arrays.

This is the FD branch of score_task3 as-is; no boundary flux involved.  From it,
invert the official perfect-answer S_physics to see what S_LWR the organizer flux
must yield for an exact answer.
"""
import sys
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\libs')
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\work\lib')
import numpy as np
import pandas as pd
import traffic

REG_ORDER = ('R1', 'R2', 'R3')


def hash_regime(panel, dates):
    import hashlib
    return np.array([REG_ORDER[int.from_bytes(hashlib.blake2b((panel + '|' + d).encode('utf-8'),
                                                             digest_size=8).digest(), 'big') % 3]
                     for d in dates], dtype=object)


def params_from_traffic(panel):
    net = traffic.load_network(panel)['links']
    return net.set_index('link_id')


def s_fd(x, p):
    """x: dict of arrays; p: params frame aligned. Mirrors score_panel FD branch."""
    lanes = np.clip(p['lanes'].to_numpy(), 1.0, None)
    q_lane = x['flow'] / lanes
    k_lane = (x['flow'] / np.clip(x['speed'], 1.0, None)) / lanes
    cap_lane = p['capacity'].to_numpy() / lanes
    kcrit_lane = p['critical_density'].to_numpy() / lanes
    kjam_lane = p['k_jam'].to_numpy() / lanes
    fs = p['free_speed'].to_numpy()
    q_fd = np.where(k_lane <= kcrit_lane, fs * k_lane,
                    (cap_lane / np.maximum(kjam_lane - kcrit_lane, 1e-9)) * np.maximum(kjam_lane - k_lane, 0))
    return float(max(0.0, 1.0 - np.abs(q_lane - q_fd).sum() / (np.abs(q_lane).sum() + 1e-9)))


for panel in ['D7_I10_E', 'D7_I10_W']:
    p = params_from_traffic(panel)
    d = traffic.load_mainline(panel, 'train')
    plain = d['plain']; masked = d['masked']
    links = plain['links'].astype(str)
    # link order == station order (verified one station per link); map link ids to params
    pl = p.reindex(links)
    times = plain['times']
    dates = pd.to_datetime(times.astype('int64'), unit='s', utc=True).strftime('%Y-%m-%d').to_numpy()
    reg = hash_regime(panel, dates)
    # verify hash regime against cached masked regime array
    mreg = masked['regime']
    agree = (np.array([REG_ORDER.index(r) + 1 for r in reg]) == mreg)
    print(f'{panel}: hash-regime vs cached masked regime agreement {agree.mean():.6f}')
    # masked observed consistency with plain
    both = np.isfinite(masked['speed']) & np.isfinite(plain['speed'])
    print('  masked==plain where both finite:', np.allclose(masked['speed'][both], plain['speed'][both], atol=1e-4))
    print('  masked nan frac', np.isnan(masked['speed']).mean(), ' plain nan frac', np.isnan(plain['speed']).mean())
    elig = plain['elig']
    ok = elig & np.isfinite(plain['speed']) & np.isfinite(plain['flow'])
    print(f'  cells eligible & finite: {ok.sum():,}  ({ok.mean():.4f})')
    vals = []
    for r in REG_ORDER:
        dm = reg == r
        okr = ok & dm[:, None]
        # rows where ok; np.where on masked arrays
        sp = plain['speed'][okr]; fl = plain['flow'][okr]; lk = np.broadcast_to(links[None, :], plain['speed'].shape)[okr]
        pp = p.reindex(lk)
        s = s_fd({'speed': sp, 'flow': fl}, pp)
        vals.append(s)
        print(f'  {r}: n={okr.sum():,} S_FD_obs={s:.4f}')
    pooled = ok
    sp = plain['speed'][pooled]; fl = plain['flow'][pooled]
    lk = np.broadcast_to(links[None, :], plain['speed'].shape)[pooled]
    print(f'  pooled S_FD_obs={s_fd({"speed": sp, "flow": fl}, p.reindex(lk)):.4f}')
    macro = float(np.mean(vals))
    print(f'  macro S_FD_obs={macro:.4f}  -> implied S_LWR for S_physics 0.9636: {(3*0.9636-macro)/2:.4f}'
          f' ; for 0.9598: {(3*0.9598-macro)/2:.4f}')
