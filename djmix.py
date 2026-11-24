#!/usr/bin/env python
"""djmix - DJ-style beatmatched mixes (built for trance / 4-on-the-floor).

  analyze  TRACK...                 BPM, beat grid, downbeat, phrases, key (Camelot), breakdowns
  mix      TRACK TRACK [TRACK...]   chained set: beatmatch (rubberband, key-lock), phrase-aligned
                                    entry, EQ bass-swap transition, loudness match, limiter

Run with:  ~/dj-env/Scripts/python.exe ~/dj-mix/djmix.py mix a.mp3 b.mp3 -o set.mp3
"""
import argparse, json, os, subprocess, sys, tempfile, shutil
import numpy as np

SR = 44100
AN_SR = 22050
HOP = 256
LOW_HZ = 150          # kick / bass band for analysis
XOVER_HZ = 120        # crossover for the bass swap
PHRASE = 8            # bars per phrase (8 bars = 32 beats)

PITCHES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------------------------------------------------------------- audio io
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        sys.exit('ffmpeg failed:\n' + r.stderr[-2000:])


def decode(path, out_wav, tempo=1.0, semitones=0.0):
    """Decode any format to 44.1k stereo float wav; optional key-locked tempo change / pitch shift."""
    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', path, '-vn', '-ac', '2', '-ar', str(SR)]
    if abs(tempo - 1) > 1e-6 or semitones:
        cmd += ['-af', f'rubberband=tempo={tempo:.7f}:pitch={2 ** (semitones / 12):.7f}'
                        ':transients=crisp:detector=percussive:channels=together:pitchq=quality']
    run(cmd + ['-c:a', 'pcm_f32le', out_wav])
    import soundfile as sf
    y, _ = sf.read(out_wav, dtype='float32', always_2d=True)
    return y


def fmt(t):
    t = max(0.0, t)
    return f'{int(t // 60)}:{t % 60:05.2f}'


# ---------------------------------------------------------------- analysis
def camelot(pc, minor):
    return f"{((pc * 7 + (4 if minor else 7)) % 12) + 1}{'A' if minor else 'B'}"


def key_compatible(c1, c2):
    n1, l1, n2, l2 = int(c1[:-1]), c1[-1], int(c2[:-1]), c2[-1]
    if n1 == n2:
        return True
    return l1 == l2 and (n1 - n2) % 12 in (1, 11)


def detect_key(y, sr):
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=4096).mean(axis=1)
    best = None
    for pc in range(12):
        for minor, prof in ((False, KS_MAJ), (True, KS_MIN)):
            r = np.corrcoef(np.roll(prof, pc), chroma)[0, 1]
            if best is None or r > best[0]:
                best = (r, pc, minor)
    _, pc, minor = best
    return {'key': PITCHES[pc] + ('m' if minor else ''), 'pc': pc, 'minor': minor,
            'camelot': camelot(pc, minor), 'confidence': round(float(best[0]), 2)}


def refine_period(flux, P0):
    """Sharpen a rough beat period with the kick-onset autocorrelation at growing multiples
    of the beat (1, 2, 4 ... 256 beats) - a constant-tempo track gives sub-0.01 BPM accuracy."""
    fr = AN_SR / HOP
    x = flux - flux.mean()
    n = len(x)
    P, rel = P0, 0.04
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        lag0 = k * P * fr
        if lag0 > n * 0.7:
            break
        w = max(2, int(np.ceil(lag0 * rel)) + 1)
        lags = np.arange(max(1, int(lag0) - w), int(lag0) + w + 2)
        r = np.array([np.dot(x[:-L], x[L:]) / (n - L) for L in lags])
        i = int(r.argmax())
        if 0 < i < len(r) - 1:
            den = r[i - 1] - 2 * r[i] + r[i + 1]
            off = 0.5 * (r[i - 1] - r[i + 1]) / den if den else 0.0
        else:
            off = 0.0
        P = (lags[i] + off) / fr / k
        rel = 1.5 / (lags[i] + off)
    return P


def analyze_audio(y_st, bpm_range=(120, 155), period=None, want_key=True):
    import librosa
    mono = y_st.mean(axis=1)
    y = librosa.resample(mono, orig_sr=SR, target_sr=AN_SR)
    dur = len(y) / AN_SR
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=AN_SR, n_fft=2048)
    low_e = S[freqs < LOW_HZ].sum(0)
    low_db = np.log1p(low_e)
    low_c = np.sqrt(low_e)                                             # compressed, not log: a kick
    flux = np.maximum(0, np.diff(low_c, prepend=low_c[0]))             # over offbeat bass still jumps
    ft = librosa.frames_to_time(np.arange(len(flux)), sr=AN_SR, hop_length=HOP)

    # --- tempo: librosa estimate, folded into range, refined by a linear fit of the beats
    if period is None:
        oenv = librosa.onset.onset_strength(y=y, sr=AN_SR, hop_length=HOP)
        tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=AN_SR, hop_length=HOP,
                                               start_bpm=138, tightness=400, units='time')
        P = 60.0 / float(np.atleast_1d(tempo)[0])
        while 60 / P < bpm_range[0]:
            P /= 2
        while 60 / P > bpm_range[1]:
            P *= 2
        P = refine_period(flux, P)
        bpm = 60 / P
        if abs(bpm - round(bpm)) < 0.05:
            bpm = float(round(bpm))
        P = 60 / bpm
    else:
        P = period
        bpm = 60 / P

    # --- grid phase: offset that puts the most kick energy on the grid
    def grid_score(o):
        ts = np.arange(o, dur, P)
        return np.interp(ts, ft, flux).sum()
    offs = np.linspace(0, P, 400, endpoint=False)
    sc = np.array([grid_score(o) for o in offs])
    o0 = offs[sc.argmax()]
    fine = np.linspace(o0 - P / 400, o0 + P / 400, 41)
    t0 = fine[np.argmax([grid_score(o) for o in fine])] % P
    if t0 > P - 0.08:          # onset detection reads ~30 ms early: a beat right at 0:00 is beat 1
        t0 -= P
    nbeats = int((dur - t0) / P)
    beat_t = t0 + P * np.arange(nbeats)

    # --- per-beat features
    def mean_between(arr, a, b):
        i, j = np.searchsorted(ft, [a, b])
        return arr[i:max(j, i + 1)].mean()
    beat_low = np.array([mean_between(low_db, t, t + P) for t in beat_t])
    beat_kick = np.interp(beat_t, ft, flux)

    # downbeat phase: section changes (kick/bass in/out) land on bar 1
    d = np.abs(np.diff(beat_low, prepend=beat_low[0]))
    phase = int(np.argmax([d[p::4].sum() for p in range(4)]))
    nbars = (nbeats - phase) // 4
    bar_t = beat_t[phase] + 4 * P * np.arange(nbars)
    bar_low = np.array([beat_low[phase + 4 * b: phase + 4 * b + 4].mean() for b in range(nbars)])
    bar_kick = np.array([beat_kick[phase + 4 * b: phase + 4 * b + 4].mean() for b in range(nbars)])
    has_kick = bar_kick > 0.35 * np.percentile(bar_kick, 90)
    bar_rms = np.array([np.sqrt(np.mean(mono[max(0, int(t * SR)): int((t + 4 * P) * SR)] ** 2) + 1e-12)
                        for t in bar_t])

    # phrase offset: section changes on multiples of 8 bars
    ch = np.abs(np.diff(bar_low, prepend=bar_low[0])) + np.abs(np.diff(has_kick.astype(float), prepend=has_kick[0]))
    phrase_off = int(np.argmax([ch[o::PHRASE].sum() for o in range(PHRASE)])) if nbars >= PHRASE else 0
    phrases = list(range(phrase_off, nbars, PHRASE))

    # breakdowns = runs of >= 4 bars without kick, between kick sections
    breakdowns, b = [], 0
    while b < nbars:
        if not has_kick[b]:
            e = b
            while e < nbars and not has_kick[e]:
                e += 1
            if e - b >= 4 and b > 0 and e < nbars:
                breakdowns.append((b, e))
            b = e
        else:
            b += 1

    res = {'bpm': round(bpm, 3), 'period': P, 'duration': dur, 'first_beat': t0,
           'downbeat_phase': phase, 'bar_times': bar_t.tolist(), 'nbars': nbars,
           'has_kick': has_kick.tolist(), 'bar_rms': bar_rms.tolist(),
           'phrase_starts': phrases, 'breakdowns': breakdowns,
           'first_kick_bar': int(np.argmax(has_kick)) if has_kick.any() else 0,
           'last_kick_bar': int(nbars - 1 - np.argmax(has_kick[::-1])) if has_kick.any() else nbars - 1}
    if want_key:
        res.update(detect_key(y, AN_SR))
    return res


def summary(name, a):
    lines = [f'== {name}',
             f"   BPM {a['bpm']}   key {a.get('key', '?')} ({a.get('camelot', '?')}, conf {a.get('confidence', '?')})"
             f"   length {fmt(a['duration'])}   bars {a['nbars']}",
             f"   first downbeat {fmt(a['bar_times'][0])}   phrases (8 bars) start at bar "
             f"{a['phrase_starts'][0] + 1}, every 8 bars",
             f"   kick: bars {a['first_kick_bar'] + 1}-{a['last_kick_bar'] + 1}"]
    for s, e in a['breakdowns']:
        lines.append(f"   breakdown: bars {s + 1}-{e} ({fmt(a['bar_times'][s])} - {fmt(a['bar_times'][min(e, a['nbars'] - 1)])})")
    return '\n'.join(lines)


# ---------------------------------------------------------------- mixing
def sin_ramp(n, up=True):
    x = np.linspace(0, np.pi / 2, n) if n > 0 else np.zeros(0)
    return np.sin(x) if up else np.cos(x)


def build_gain(length, points):
    """points: list of (sample, value) with linear-free piecewise sin ramps between them."""
    g = np.zeros(length, np.float32)
    pts = sorted(points)
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        s0c, s1c = max(0, s0), min(length, s1)
        if s1c <= s0c:
            continue
        n = s1 - s0
        x = (np.arange(s0c, s1c) - s0) / max(n, 1)
        w = np.sin(x * np.pi / 2) ** 2          # smooth S-curve
        g[s0c:s1c] = v0 + (v1 - v0) * w
    last_s, last_v = pts[-1]
    if last_s < length:
        g[max(0, last_s):] = last_v
    first_s, first_v = pts[0]
    if first_s > 0:
        g[:min(first_s, length)] = first_v
    return g


def fine_align(ya, sa, yb, sb, length, max_ms=20):
    """Sample-level nudge: cross-correlate onset envelopes of the overlapping parts (like riding
    the jog wheel until the hats lock). Returns the shift in samples to add to B's position."""
    hop = 32
    def env(y, s):
        x = y[max(0, s): s + length].mean(axis=1)
        x = np.diff(x, prepend=0.0)                          # emphasise transients
        e = np.sqrt(np.add.reduceat(x ** 2, np.arange(0, len(x) - hop, hop)) / hop)
        return np.maximum(0, np.diff(e, prepend=e[0]))
    a, b = env(ya, sa), env(yb, sb)
    m = min(len(a), len(b))
    if m < 100:
        return 0
    a, b = a[:m] - a[:m].mean(), b[:m] - b[:m].mean()
    K = int(max_ms / 1000 * SR / hop)
    lags = np.arange(-K, K + 1)
    r = np.array([np.dot(a[max(0, l):m + min(0, l)], b[max(0, -l):m - max(0, l)]) for l in lags])
    i = int(r.argmax())
    off = 0.0
    if 0 < i < len(r) - 1:
        den = r[i - 1] - 2 * r[i] + r[i + 1]
        off = 0.5 * (r[i - 1] - r[i + 1]) / den if den else 0.0
    return int(round((lags[i] + off) * hop))


def split_bands(y):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, XOVER_HZ, 'lowpass', fs=SR, output='sos')
    low = sosfiltfilt(sos, y, axis=0).astype(np.float32)
    return low, (y - low).astype(np.float32)


def cmd_analyze(args):
    work = tempfile.mkdtemp(prefix='djmix_')
    try:
        out = {}
        for i, p in enumerate(args.tracks):
            y = decode(p, os.path.join(work, f'{i}.wav'))
            a = analyze_audio(y, tuple(args.bpm_range))
            print(summary(os.path.basename(p), a))
            out[p] = {k: v for k, v in a.items() if k not in ('bar_times', 'has_kick', 'bar_rms')}
        if args.json:
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def per_track(lst, i, default):
    if not lst:
        return default
    v = lst[i] if i < len(lst) else default
    return default if v in (None, '-', '') else v


def cmd_mix(args):
    work = tempfile.mkdtemp(prefix='djmix_')
    try:
        n = len(args.tracks)
        print('analyzing ...')
        orig = []
        for i, p in enumerate(args.tracks):
            y = decode(p, os.path.join(work, f'o{i}.wav'))
            a = analyze_audio(y, tuple(args.bpm_range))
            orig.append(a)
            print(summary(os.path.basename(p), a))
        target_bpm = args.bpm or orig[0]['bpm']
        P = 60 / target_bpm
        print(f'\nset tempo: {target_bpm} BPM')

        # harmonic check / pitch shifts
        semis = [float(per_track(args.semitones, i, 0)) for i in range(n)]
        for i in range(1, n):
            ca, cb = orig[i - 1]['camelot'], orig[i]['camelot']
            if not key_compatible(ca, cb):
                sugg = None
                for s in (1, -1, 2, -2):
                    pcb = (orig[i]['pc'] + s) % 12
                    if key_compatible(ca, camelot(pcb, orig[i]['minor'])):
                        sugg = s
                        break
                msg = f'  ! keys clash: {orig[i-1]["key"]} ({ca}) -> {orig[i]["key"]} ({cb})'
                if sugg is not None:
                    msg += f'; shifting track {i+1} by {sugg:+d} semitone(s) would fix it'
                    if args.harmonic and not args.semitones:
                        semis[i] = sugg
                        msg += ' -> applied (--harmonic)'
                print(msg)

        # stretch to set tempo and re-analyse the stretched audio (exact grid, no latency guesswork)
        tracks = []
        for i, p in enumerate(args.tracks):
            ratio = target_bpm / orig[i]['bpm']
            if abs(ratio - 1) < 1e-6 and not semis[i]:
                y = decode(p, os.path.join(work, f's{i}.wav'))
            else:
                print(f'  stretching track {i+1}: {orig[i]["bpm"]} -> {target_bpm} BPM ({(ratio-1)*100:+.2f}%)'
                      + (f', pitch {semis[i]:+g} st' if semis[i] else ''))
                y = decode(p, os.path.join(work, f's{i}.wav'), tempo=ratio, semitones=semis[i])
            a = analyze_audio(y, period=P, want_key=False)
            a['phrase_starts'] = orig[i]['phrase_starts']        # structure from the original
            a['downbeat_phase_orig'] = orig[i]['downbeat_phase']
            tracks.append((y, a))

        # loudness match on the bars with kick
        def loud(a):
            r = np.array(a['bar_rms'])[np.array(a['has_kick'], bool)]
            return np.median(r) if len(r) else np.median(a['bar_rms'])
        ref = loud(tracks[0][1])
        gains = [float(np.clip(ref / loud(a), 10 ** (-6 / 20), 10 ** (6 / 20))) for _, a in tracks]

        L = args.bars
        swap = args.swap if args.swap is not None else L // 2
        pos = [0]                    # sample offset of each track in the output
        trans = []                   # (T0, Ts, T1) in output samples, between i-1 and i
        play_from = [0]              # first audible bar of each track (own bars)
        for i in range(1, n):
            ya, aa = tracks[i - 1]
            yb, ab = tracks[i]
            bt_a, bt_b = aa['bar_times'], ab['bar_times']
            # incoming track enters on its first phrase start (intro), or user bar
            in_bar = int(per_track(args.in_bar, i, 0)) - 1 if per_track(args.in_bar, i, 0) else ab['phrase_starts'][0]
            in_bar = min(max(in_bar, 0), ab['nbars'] - 1)
            # outgoing track: latest phrase start so the whole transition fits in A
            ob = per_track(args.out_bar, i - 1, 0)
            if ob:
                s = int(ob) - 1
            else:
                cand = [b for b in aa['phrase_starts'] if b + L <= aa['nbars'] and b >= play_from[-1] + swap]
                s = cand[-1] if cand else max(play_from[-1], aa['nbars'] - L)
            s = min(max(s, 0), aa['nbars'] - 1)
            T0 = pos[-1] + bt_a[s] * SR
            pb = int(round(T0 - bt_b[in_bar] * SR))
            pb += fine_align(ya, int(round(bt_a[s] * SR)), yb, int(round(bt_b[in_bar] * SR)),
                             int(L * 4 * P * SR))
            T0 = int(round(T0))
            Ts = int(round(T0 + swap * 4 * P * SR))
            T1 = int(round(T0 + L * 4 * P * SR))
            pos.append(pb)
            play_from.append(in_bar + L)
            trans.append((T0, Ts, T1, s, in_bar))

        total = max(pos[i] + len(tracks[i][0]) for i in range(n))
        out = np.zeros((total, 2), np.float32)
        ramp = int(P * SR / 4)       # bass swap: 1/16 note ramp
        tracklist = []
        for i, (y, a) in enumerate(tracks):
            low, high = split_bands(y * gains[i])
            ln = len(y)
            lo_pts, hi_pts = [(0, 1.0)], [(0, 1.0)]
            if i > 0:                                         # incoming
                T0, Ts, T1, _, in_bar = trans[i - 1]
                st = int(round(a['bar_times'][in_bar] * SR))   # own sample where it starts
                hi_pts = [(st, 0.0), (st + (Ts - T0), 1.0)]
                lo_pts = [(st + (Ts - T0) - ramp, 0.0), (st + (Ts - T0), 1.0)]
                tracklist.append((T0, Ts, i))
            if i < n - 1:                                     # outgoing
                T0, Ts, T1, _, _ = trans[i]
                o_s, o_1 = Ts - pos[i], T1 - pos[i]
                hi_pts = hi_pts + [(o_s, 1.0), (o_1, 0.0)]
                lo_pts = lo_pts + [(o_s - ramp, 1.0), (o_s, 0.0)]
            g_lo, g_hi = build_gain(ln, lo_pts), build_gain(ln, hi_pts)
            seg = low * g_lo[:, None] + high * g_hi[:, None]
            a0 = pos[i]
            s0 = max(0, -a0)
            out[a0 + s0: a0 + ln] += seg[s0:]
        tracklist.insert(0, (0, 0, 0))

        base, ext = os.path.splitext(args.output)
        raw = os.path.join(work, 'mix.wav')
        import soundfile as sf
        sf.write(raw, out, SR, subtype='FLOAT')
        af = 'alimiter=limit=0.89:attack=5:release=60:level=false'
        if ext.lower() == '.wav':
            run(['ffmpeg', '-y', '-v', 'error', '-i', raw, '-af', af, '-c:a', 'pcm_s24le', args.output])
        else:
            run(['ffmpeg', '-y', '-v', 'error', '-i', raw, '-af', af, '-c:a', 'libmp3lame', '-b:a', '320k', args.output])

        # report + tracklist
        lines = [f'set tempo {target_bpm} BPM, transitions {L} bars, bass swap at bar {swap}']
        for T0, Ts, i in tracklist:
            nm = os.path.basename(args.tracks[i])
            if i == 0:
                lines.append(f'0:00.00  {nm}')
            else:
                T0_, Ts_, T1_, s, in_bar = trans[i - 1]
                lines.append(f'{fmt(T0 / SR)}  {nm}  (enters at its bar {in_bar + 1}; '
                             f'bass swap {fmt(Ts / SR)}; previous track out by {fmt(T1_ / SR)}; '
                             f'previous track mixed from its bar {s + 1})')
        lines.append(f'length {fmt(total / SR)}')
        rep = '\n'.join(lines)
        print('\n' + rep)
        with open(base + '.tracklist.txt', 'w', encoding='utf-8') as f:
            f.write(rep + '\n')
        with open(base + '.report.json', 'w', encoding='utf-8') as f:
            json.dump({'bpm': target_bpm, 'positions_sec': [p / SR for p in pos],
                       'transitions_sec': [[t[0] / SR, t[1] / SR, t[2] / SR] for t in trans],
                       'gains_db': [20 * np.log10(g) for g in gains], 'semitones': semis,
                       'tracks': args.tracks}, f, indent=1, ensure_ascii=False)
        print(f'\nwritten: {args.output}')
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('tracks', nargs='+')
    a.add_argument('--json')
    a.add_argument('--bpm-range', nargs=2, type=float, default=[120, 155])
    m = sub.add_parser('mix')
    m.add_argument('tracks', nargs='+')
    m.add_argument('-o', '--output', required=True, help='.mp3 (320k) or .wav (24-bit)')
    m.add_argument('--bars', type=int, default=32, help='transition length in bars (default 32)')
    m.add_argument('--swap', type=int, help='bar of the bass swap inside the transition (default half)')
    m.add_argument('--bpm', type=float, help='set tempo (default: first track)')
    m.add_argument('--out-bar', nargs='+', help='per outgoing track: bar where its mix-out starts (1-based, - = auto)')
    m.add_argument('--in-bar', nargs='+', help='per track: bar where it enters (1-based, - = auto; first ignored)')
    m.add_argument('--semitones', nargs='+', help='per track pitch shift in semitones')
    m.add_argument('--harmonic', action='store_true', help='auto pitch-shift clashing keys by +-1/2 semitones')
    m.add_argument('--bpm-range', nargs=2, type=float, default=[120, 155])
    args = ap.parse_args()
    if args.cmd == 'mix' and len(args.tracks) < 2:
        sys.exit('mix needs at least 2 tracks')
    {'analyze': cmd_analyze, 'mix': cmd_mix}[args.cmd](args)


if __name__ == '__main__':
    main()
