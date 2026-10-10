"""CD-XA ADPCM encoder: 4-bit, 37800 Hz, stereo (or mono duplicated to both channels).
Sector payload = 18 sound groups x 128 bytes (+20 zero bytes) = 2324 bytes (Mode2 Form2)."""
import numpy as np

K1 = (0, 60, 115, 98)
K2 = (0, 0, -52, -55)
SAMPLES_PER_SECTOR = 2016          # per channel
UNITS = 28

try:
    from numba import njit          # optional speed-up
except Exception:
    njit = None

def _select(x):
    """Open-loop choice of filter + shift for every 28-sample unit (vectorised). x: int32 array, len % 28 == 0."""
    n = len(x)
    p1 = np.r_[0, x[:-1]]; p2 = np.r_[0, 0, x[:-2]]
    best_shift = np.full(n // UNITS, -1); best_filt = np.zeros(n // UNITS, np.int64); best_cost = np.full(n // UNITS, 1 << 60, np.int64)
    for f in range(4):
        res = (x - ((K1[f] * p1 + K2[f] * p2 + 32) >> 6)).reshape(-1, UNITS)
        hi, lo, cost = res.max(1), res.min(1), np.abs(res).sum(1)
        rs = np.zeros(len(hi), np.int64)
        for _ in range(12):
            bad = ((hi >> rs) > 7) | ((lo >> rs) < -8)
            rs = np.where(bad & (rs < 12), rs + 1, rs)
        shift = 12 - rs
        better = (shift > best_shift) | ((shift == best_shift) & (cost < best_cost))
        best_shift = np.where(better, shift, best_shift); best_filt = np.where(better, f, best_filt)
        best_cost = np.where(better, cost, best_cost)
    return best_filt, best_shift

def _attempt(xs, base, k1, k2, s, p1, p2, out):
    err = 0; clipped = False
    for i in range(UNITS):
        x = xs[base + i]
        pred = (k1 * p1 + k2 * p2 + 32) >> 6
        e = (((x - pred) << s) + 2048) >> 12
        if e > 7: e = 7; clipped = True
        elif e < -8: e = -8; clipped = True
        d = ((e << 12) >> s) + pred
        if d > 32767: d = 32767
        elif d < -32768: d = -32768
        err += (d - x) * (d - x)
        out[i] = e & 15
        p2 = p1; p1 = d
    return err, clipped, p1, p2

def _encode_units(xs, filt, shift):
    n = len(filt)
    nib = [0] * (n * UNITS); hdr = [0] * n
    p1 = p2 = 0
    a = [0] * UNITS; b = [0] * UNITS
    for u in range(n):
        f = filt[u]; s = shift[u]; k1 = K1[f]; k2 = K2[f]
        err, clipped, q1, q2 = _attempt(xs, u * UNITS, k1, k2, s, p1, p2, a)
        if clipped and s > 0:                    # try one step coarser, keep whichever has less error
            err2, _, r1, r2 = _attempt(xs, u * UNITS, k1, k2, s - 1, p1, p2, b)
            if err2 < err:
                a, b = b, a; s -= 1; q1, q2 = r1, r2
        nib[u * UNITS:(u + 1) * UNITS] = a
        hdr[u] = (s & 15) | (f << 4)
        p1, p2 = q1, q2
    return nib, hdr

if njit:
    _attempt = njit(cache=True)(_attempt)

def encode_channel(x):
    x = np.asarray(x, np.int64)
    pad = (-len(x)) % UNITS
    x = np.r_[x, np.zeros(pad, np.int64)]
    filt, shift = _select(x)
    nib, hdr = _encode_units(x.tolist(), filt.tolist(), shift.tolist())
    return np.array(nib, np.uint8).reshape(-1, UNITS), np.array(hdr, np.uint8)

def encode(pcm):
    """pcm: int16 array shape (N, 2). Returns list of 2324-byte sector payloads."""
    n_sec = -(-len(pcm) // SAMPLES_PER_SECTOR)
    need = n_sec * SAMPLES_PER_SECTOR
    pcm = np.vstack([pcm, np.zeros((need - len(pcm), 2), pcm.dtype)])
    nl, hl = encode_channel(pcm[:, 0]); nr, hr = encode_channel(pcm[:, 1])
    S = n_sec
    nl = nl.reshape(S, 18, 4, UNITS); nr = nr.reshape(S, 18, 4, UNITS)
    hl = hl.reshape(S, 18, 4); hr = hr.reshape(S, 18, 4)
    sec = np.zeros((S, 18, 128), np.uint8)
    h = np.zeros((S, 18, 16), np.uint8)
    for k in range(4):
        base = 0 if k < 2 else 8
        h[:, :, base + 2 * (k % 2)] = hl[:, :, k]
        h[:, :, base + 2 * (k % 2) + 1] = hr[:, :, k]
    h[:, :, 4:8] = h[:, :, 0:4]; h[:, :, 12:16] = h[:, :, 8:12]
    sec[:, :, :16] = h
    data = (nl.transpose(0, 1, 3, 2) | (nr.transpose(0, 1, 3, 2) << 4)).astype(np.uint8)   # (S,18,28,4): sample i, unit k
    sec[:, :, 16:] = data.reshape(S, 18, 112)
    out = np.zeros((S, 2324), np.uint8)
    out[:, :2304] = sec.reshape(S, 2304)
    return [bytes(r) for r in out]
