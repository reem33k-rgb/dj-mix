"""Generate synthetic trance-like test tracks: test_gen.py OUT.mp3 BPM ROOT_PC MINOR(0/1) LEAD_SILENCE_SEC"""
import sys, subprocess, numpy as np, soundfile as sf

SR = 44100
out, bpm, root, minor, lead = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
P = 60 / bpm
# structure (bars): intro kick+hat 32 | +bass 32 | breakdown 16 | full 32 | outro kick+hat 32
sections = [('intro', 32), ('groove', 32), ('break', 16), ('full', 32), ('outro', 32)]
nbars = sum(b for _, b in sections)
N = int((lead + nbars * 4 * P + 1) * SR)
y = np.zeros(N)
rng = np.random.default_rng(int(bpm))

def place(sig, t, gain=1.0):
    i = int(round(t * SR)); j = min(N, i + len(sig))
    y[i:j] += gain * sig[:j - i]

tk = np.arange(int(0.3 * SR)) / SR
kick = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-tk * 30)) / SR) * np.exp(-tk * 9)
th = np.arange(int(0.05 * SR)) / SR
hat = np.diff(rng.standard_normal(len(th) + 1)) * np.exp(-th * 80) * 0.15
f_bass = 55 * 2 ** (((root - 9) % 12) / 12)
tb = np.arange(int(P / 2 * 0.9 * SR)) / SR
bass = (2 * ((tb * f_bass) % 1) - 1) * np.exp(-tb * 6) * 0.35
third = 3 if minor else 4
bar = 0
for name, nb in sections:
    for b in range(nb):
        t_bar = lead + (bar + b) * 4 * P
        for q in range(4):
            tb_ = t_bar + q * P
            if name != 'break':
                place(kick, tb_, 0.9)
                place(hat, tb_ + P / 2)
            if name in ('groove', 'full'):
                place(bass, tb_ + P / 2)
        if name in ('break', 'full'):
            tp = np.arange(int(4 * P * SR)) / SR
            f0 = 220 * 2 ** (((root - 9) % 12) / 12)
            pad = sum(np.sin(2 * np.pi * f0 * 2 ** (s / 12) * tp) for s in (0, third, 7)) * 0.05
            place(pad, t_bar)
    bar += nb
y /= np.abs(y).max() * 1.1
wav = out + '.wav'
sf.write(wav, np.stack([y, y], 1), SR)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', wav, '-b:a', '320k', out], check=True)
import os; os.remove(wav)
print(out, nbars, 'bars', f'{N/SR:.1f}s')
