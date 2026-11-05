import sys
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\libs')
sys.path.insert(0, r'C:\Users\lenovo\Desktop\kaggle\work\lib')
import numpy as np, traffic

for panel in ['D7_I10_E', 'D7_I10_W']:
    d = traffic.load_mainline(panel, 'train')
    p = d['plain']
    links = p['links'].astype(str)
    u, c = np.unique(links, return_counts=True)
    print(panel, 'T', p['speed'].shape[0], 'S', p['speed'].shape[1], 'unique_links', len(u),
          'dup_links', int((c > 1).sum()), 'max stations per link', int(c.max()))
    print('  keys:', sorted(p.keys()))
    print('  regime values (plain):', np.unique(p['regime']))
    m = d['masked']
    print('  masked regime values:', np.unique(m['regime']), 'evens/odds?', m['regime'][:8])
    tgt = traffic.load_targets(panel, 'train')
    print('  targets n=', len(tgt['t_idx']), 'regime counts', np.bincount(tgt['regime']))
    ramps = traffic.load_ramps(panel, 'train')
    print('  ramps on_valid frac', ramps['on_valid'].mean(), 'off_valid frac', ramps['off_valid'].mean())
    # eligibility / finite stats
    sp, fl, el = p['speed'], p['flow'], p['elig']
    print('  elig frac', el.mean(), 'finite speed frac', np.isfinite(sp).mean(), 'speed nan', np.isnan(sp).mean())
    print('  observed speed min/max', np.nanmin(sp), np.nanmax(sp), 'flow min/max', np.nanmin(fl), np.nanmax(fl))
