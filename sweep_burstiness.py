#!/usr/bin/env python3
"""Burstiness sweeps for the periodic-incast experiment (scratch/ECNMC).

Two sweeps, both against a 100 KB (BDP) last-hop buffer with MinTh=0.15 -- DCTCP's own
K > C*RTT/7 rule, giving a 15 KB marking threshold:

  spread   The burstiness axis. periodicPhaseSpread goes 0 -> 1 while everything else is held
           fixed, so sender i starts at phase spread*i/N of the period. At 0 all 12 senders fire
           together and the receiver sees one 12-packet batch; at 1 they are staggered evenly and
           it sees a near-deterministic smooth stream. Offered load, message size, buffer and ECN
           threshold are identical at every point, and relative phase is the one thing TCP pacing
           cannot undo. Values are dense below 0.25 because the transition happens where the
           stagger reaches the burst's drain time (0.98 us at 6 x 2048 B, i.e. spread ~0.077 at a
           12.8 us period), so 0.04/0.08/0.15 bracket it. The burst is 82% of the ECN threshold at
           every point, so nothing marks and pacing is inert -- this axis is measured in the clean
           regime, and `msgsize` is where crossing into marking is measured.
           NOTE the message must stay multi-segment (16384 B = 12 segments). With a one-segment
           message each sender contributes a single packet, the monitored flow's position in the
           batch is fixed by ns-3's deterministic tie-break, and it reads exactly 0 ns queuing
           delay at every point -- measured, against 246-843 ns for its rack-mates.

  msgsize  The validation axis. Phase spread is pinned at 0 and the message grows, with the period
           scaled to hold the offered load fixed, so the burst runs from 82% to 328% of the ECN
           marking threshold -- crossing it between the first and second point. Below it nothing
           marks and paced and unpaced should coincide; above it RED marks, DCTCP cuts cwnd and
           pacing smears the burst, so the two arms separate. That separation is the measurement.
           The larger points are NOMINALLY burstier, but RED marks, DCTCP cuts cwnd, and pacing
           then smears the burst flat -- so their MEASURED burstiness should stop rising. Plotted
           against measured IDC rather than against the config knob they should fall on the same
           curve as the spread sweep, which is what shows that burstiness, not the parameter,
           governs the result.

--pacing runs either or both of pctPacedBack 1.0 and 0.0; the setting is appended to the result
directory name. The unpaced arm is the upper bound for the whole curve and, paired point by point
with the paced arm, isolates what pacing costs at each burstiness level.

Each sweep point is given its own `load` value purely as a directory label (with periodicTraffic
the offered load comes from periodicPeriod and periodicMsgSize, not from `load`), so points land
in separate result directories and PostProcessing.py's existing per-load axis becomes the sweep
axis for free.

  python3 sweep_burstiness.py --list                       # show both sweeps
  python3 sweep_burstiness.py spread --dry-run             # print the commands, change nothing
  python3 sweep_burstiness.py spread --test                # one experiment per point
  python3 sweep_burstiness.py spread --pacing both --threads 30
"""
import argparse, configparser, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'Parameters.config')
NS3 = os.path.abspath(os.path.join(HERE, '..', '..'))
RAW = os.path.join(NS3, 'scratch', 'Results_forward')

LINK_BPS = 100e9          # hostToTorLinkRate
BUFFER_B = 100e3          # switchSrcREDQueueDiscMaxSize
MIN_TH = 0.15             # ECN marking threshold as a fraction of the buffer
# Senders feeding the MONITORED receiver, i.e. (sending racks per receiver) x
# periodicSenders. With 3 sending racks split over 3 receivers that is 1 x 6 = 6, so the
# monitored last hop sees the same 98 KB burst as the single-rack version did.
SENDERS_PER_RECEIVER = 6
DELTA = 0.0768            # offered load on the receiver's last hop, held fixed across both sweeps

def period_for(msg_size):
    """Period that holds the busy fraction at DELTA for this message size."""
    return 8.0 * SENDERS_PER_RECEIVER * msg_size / (DELTA * LINK_BPS)

SWEEPS = {
    'spread': {
        'dir': 'forward_burstiness_spread',
        'fixed': {'periodicMsgSize': 2048, 'periodicPeriod': '12.800us',
                  'periodicSenderRacks': 3, 'periodicSenders': 6,
                  'periodicDstHosts': 3},
        # label -> overrides. The label is also the x-axis value.
        'points': [(s, {'periodicPhaseSpread': s}) for s in
                   (0.0, 0.04, 0.08, 0.15, 0.30, 1.0)],
    },
    'msgsize': {
        'dir': 'forward_burstiness_msgsize',
        'fixed': {'periodicPhaseSpread': 0.0,
                  'periodicSenderRacks': 3, 'periodicSenders': 6,
                  'periodicDstHosts': 3},
        'points': [(round(m / 1000.0, 3),
                    {'periodicMsgSize': m,
                     'periodicPeriod': '%.3fus' % (period_for(m) * 1e6)})
                   for m in (2048, 2560, 3072, 4096, 8192)],
    },
}

PACING = {'on': 1.0, 'off': 0.0}

def read_config():
    c = configparser.ConfigParser()
    c.read(CONFIG)
    return c

def describe(name):
    sw = SWEEPS[name]
    c = read_config()
    print('sweep %-8s -> scratch/Results_%s_{paced,unpaced}' % (name, sw['dir']))
    print('  fixed: %s' % sw['fixed'])
    print('  %-9s %-10s %-10s %-8s %8s %9s %8s %10s' %
          ('label', 'msg', 'period', 'spread', 'burst', 'vs MinTh', 'delta', 'periods'))
    window = c.getfloat('Settings', 'steadyEnd') - c.getfloat('Settings', 'steadyStart')
    for label, over in sw['points']:
        v = dict(sw['fixed']); v.update(over)
        msg = int(v.get('periodicMsgSize', c.getint('DCSim', 'periodicMsgSize')))
        per = v.get('periodicPeriod', c.get('DCSim', 'periodicPeriod'))
        per_s = float(str(per).replace('us', '')) * 1e-6
        spread = float(v.get('periodicPhaseSpread', c.getfloat('DCSim', 'periodicPhaseSpread')))
        burst = SENDERS_PER_RECEIVER * msg
        delta = 8.0 * burst / (per_s * LINK_BPS)
        flag = '' if burst <= MIN_TH * BUFFER_B else '  <-- marks, pacing flattens'
        print('  %-9s %-10s %-10s %-8s %7.1fK %8.0f%% %8.4f %10.0f%s' %
              (label, msg, per, spread, burst / 1000.0,
               100 * burst / (MIN_TH * BUFFER_B), delta, window / per_s, flag))
    print('  window %.2f s, %s experiments per point (--test runs 1)'
          % (window, c.get('Settings', 'experiments')))
    print('  PostProcessing.py loads = %s' % [float(x) for x, _ in sw['points']])

def patch_config(overrides):
    """Rewrite the key = value lines in place; comments and layout are preserved."""
    text = open(CONFIG).read()
    for key, value in overrides.items():
        pat = re.compile(r'^%s=.*$' % re.escape(key), re.MULTILINE)
        if not pat.search(text):
            sys.exit('Parameters.config has no key %r' % key)
        text = pat.sub('%s=%s' % (key, value), text, count=1)
    open(CONFIG, 'w').write(text)

def run(name, pacing, args):
    sw = SWEEPS[name]
    labels = [p[0] for p in sw['points']]
    if len(set(labels)) != len(labels):
        sys.exit('sweep point labels must be unique -- they are the result directory names')
    tag = 'paced' if pacing == 'on' else 'unpaced'
    dest = os.path.join(NS3, 'scratch', 'Results_%s_%s' % (sw['dir'], tag))
    if os.path.isdir(dest) and not args.dry_run:
        sys.exit('%s already exists -- move it aside before rerunning' % dest)

    if os.path.isdir(RAW) and os.listdir(RAW) and not args.dry_run:
        backup = '%s.bak.%s' % (RAW, time.strftime('%Y%m%d-%H%M%S'))
        print('moving existing %s aside -> %s' % (RAW, backup))
        shutil.move(RAW, backup)

    saved = open(CONFIG).read()
    try:
        for label, over in sw['points']:
            overrides = dict(sw['fixed'])
            overrides.update(over)
            overrides['load'] = label                    # directory label / sweep x-axis
            overrides['pctPacedBack'] = PACING[pacing]
            cmd = ['python3', 'exp.py', '--IsForward', '1',
                   '--IsTest', '1' if args.test else '0',
                   '--NumThreads', str(args.threads)]
            print('\n=== %s/%s point %s: %s' % (name, tag, label, overrides))
            print('    %s' % ' '.join(cmd))
            if args.dry_run:
                continue
            patch_config(overrides)
            rc = subprocess.call(cmd, cwd=HERE)
            if rc != 0:
                sys.exit('exp.py failed at point %s (exit %d)' % (label, rc))
    finally:
        open(CONFIG, 'w').write(saved)
        print('\nParameters.config restored')

    if args.dry_run:
        return
    shutil.move(RAW, dest)
    print('raw data -> %s' % dest)
    print('postprocess with:  --dir %s_%s   loads = %s'
          % (sw['dir'], tag, [float(x) for x in labels]))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep', nargs='?', choices=sorted(SWEEPS))
    ap.add_argument('--list', action='store_true', help='describe both sweeps and exit')
    ap.add_argument('--dry-run', action='store_true', help='print commands, change nothing')
    ap.add_argument('--test', action='store_true', help='one experiment per point')
    ap.add_argument('--threads', type=int, default=30)
    ap.add_argument('--pacing', choices=['on', 'off', 'both'], default='on',
                    help='pctPacedBack 1.0, 0.0, or both arms in sequence')
    a = ap.parse_args()
    if a.list or not a.sweep:
        for n in sorted(SWEEPS):
            describe(n)
            print()
        sys.exit(0)
    describe(a.sweep)
    for p in (['on', 'off'] if a.pacing == 'both' else [a.pacing]):
        run(a.sweep, p, a)
