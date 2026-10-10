"""CD-XA ADPCM decoder (4/8 bit, mono/stereo) -> int16 PCM. Used to inspect / re-time / convert existing audio."""
import numpy as np
K1 = (0, 60, 115, 98)
K2 = (0, 0, -52, -55)

def _decode_units(nib, hdr, bits):
    """nib: (U,28) signed ints, hdr: (U,) param bytes -> (U,28) int PCM, with per-channel predictor state carried by caller."""
    raise NotImplementedError

def decode(payloads, bits=4, stereo=True):
    """payloads: list of 2324-byte sector payloads. Returns int16 array (N, 2) (mono is duplicated to both channels)."""
    nsec = len(payloads)
    if nsec == 0: return np.zeros((0, 2), np.int16)
    raw = np.frombuffer(b"".join(p[:2304] for p in payloads), np.uint8).reshape(nsec * 18, 128)
    hdr = raw[:, :16]; data = raw[:, 16:].reshape(-1, 28, 4)            # (groups, sample i, byte j)
    G = nsec * 18
    # unit parameter bytes
    par = np.concatenate([hdr[:, 0:4], hdr[:, 8:12]], axis=1)           # (G, 8) for units 0..7
    if bits == 4:
        lo = (data & 15).astype(np.int8); hi = (data >> 4).astype(np.int8)
        lo = np.where(lo > 7, lo - 16, lo); hi = np.where(hi > 7, hi - 16, hi)
        if stereo:   # byte j: low = L_j, high = R_j  (units: L0 R0 L1 R1 | L2 R2 L3 R3 -> param order 0,1,2,3 | 4..7)
            L = lo.transpose(0, 2, 1); R = hi.transpose(0, 2, 1)         # (G, 4, 28)
            pL = par[:, [0, 2, 4, 6]]; pR = par[:, [1, 3, 5, 7]]
            chans = [(L, pL), (R, pR)]
        else:        # mono: unit u = byte u//2, nibble u%2
            u = np.stack([lo, hi], axis=3).transpose(0, 2, 3, 1)         # (G, 4, 2, 28)
            u = u.reshape(G, 8, 28)
            chans = [(u, par)]
        shift_base = 12
    else:
        s8 = data.astype(np.int8).transpose(0, 2, 1).astype(np.int32)    # (G,4,28)
        if stereo:
            chans = [(s8[:, [0, 2], :], par[:, [0, 2]]), (s8[:, [1, 3], :], par[:, [1, 3]])]
        else:
            chans = [(s8, par[:, :4])]
        shift_base = 8
    outs = []
    for units, p in chans:
        U = units.shape[1]
        x = units.astype(np.int64).reshape(G * U, 28)
        pp = p.reshape(G * U)
        shift = (pp & 15).astype(np.int64); shift = np.where(shift > shift_base, 9 if shift_base == 12 else shift_base, shift)
        filt = (pp >> 4) & 3
        v = ((x << shift_base) >> shift[:, None]).tolist()                # arithmetic shift as the hardware does
        fl = filt.tolist()
        out = np.empty((G * U, 28), np.int64)
        p1 = p2 = 0
        res = []
        for ui in range(G * U):
            k1 = K1[fl[ui]]; k2 = K2[fl[ui]]
            row = v[ui]
            o = [0] * 28
            for i in range(28):
                s = row[i] + ((k1 * p1 + k2 * p2 + 32) >> 6)
                if s > 32767: s = 32767
                elif s < -32768: s = -32768
                o[i] = s; p2 = p1; p1 = s
            res.append(o)
        outs.append(np.array(res, np.int16).reshape(-1))
    if len(outs) == 1:
        a = outs[0]; return np.stack([a, a], axis=1)
    return np.stack(outs, axis=1)

def resample_to_37800(pcm, rate):
    """Very simple resampler for 18900 -> 37800 (linear interpolation)."""
    if rate == 37800 or len(pcm) == 0: return pcm
    n = len(pcm); t = np.arange(int(n * 37800 / rate)) * rate / 37800.0
    return np.stack([np.interp(t, np.arange(n), pcm[:, c].astype(np.float64)).round() for c in range(2)], axis=1).astype(np.int16)
