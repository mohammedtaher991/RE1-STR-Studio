"""CD-ROM XA (Mode 2) sector helpers: EDC/ECC generation, build & parse."""
import struct

SYNC = b"\x00" + b"\xff" * 10 + b"\x00"

def _tables():
    edc = []
    for i in range(256):
        x = i
        for _ in range(8):
            x = (x >> 1) ^ (0xD8018001 if x & 1 else 0)
        edc.append(x)
    f, b = [0] * 256, [0] * 256
    for i in range(256):
        j = (i << 1) ^ (0x11D if i & 0x80 else 0)
        f[i] = j & 0xFF
        b[i ^ j & 0xFF] = i
    return edc, f, b
EDC_T, ECC_F, ECC_B = _tables()

def edc(data):
    c = 0
    for x in data:
        c = (c >> 8) ^ EDC_T[(c ^ x) & 0xFF]
    return c

import numpy as np
_F = np.array(ECC_F, np.uint8); _B = np.array(ECC_B, np.uint8)
EDC_NP = np.array(EDC_T, np.uint32)

def _idx(major, minor, msize, moff):
    size = major * minor
    ix = np.zeros((major, minor), np.int64)
    for m in range(major):
        a = (m >> 1) * moff + (m & 1)
        for k in range(minor):
            ix[m, k] = a
            a += msize
            if a >= size: a -= size
    return ix
_P_IX = _idx(86, 24, 86, 2)
_Q_IX = _idx(52, 43, 88, 86)

def _ecc_np(area, ix):
    """area: (N, size) uint8 -> (N, 2*major) parity bytes."""
    major, minor = ix.shape
    a = np.zeros((area.shape[0], major), np.uint8); b = np.zeros_like(a)
    for k in range(minor):
        t = area[:, ix[:, k]]
        a ^= t; b ^= t
        a = _F[a]
    a = _B[_F[a] ^ b]
    return np.concatenate([a, a ^ b], axis=1)

def ecc_batch(sectors):
    """sectors: (N, 2352) uint8 Form1 sectors (header+data+EDC filled). Returns (N, 276) P+Q parity."""
    area = np.zeros((sectors.shape[0], 2340), np.uint8)
    area[:, :2064] = sectors[:, 12:2076]
    area[:, 0:4] = 0
    p = _ecc_np(area[:, :2064], _P_IX)
    area[:, 2064:2236] = p
    q = _ecc_np(area[:, :2236], _Q_IX)
    return np.concatenate([p, q], axis=1)

def edc_batch(data):
    """data: (N, L) uint8 -> (N,) uint32 EDC."""
    c = np.zeros(data.shape[0], np.uint32)
    for k in range(data.shape[1]):
        c = (c >> 8) ^ EDC_NP[(c ^ data[:, k]) & 0xFF]
    return c

def check_sectors(arr):
    """arr: (N, 2352) uint8. Returns dict of bool arrays: sync, edc, ecc (form-aware: Form2 EDC may be zero)."""
    n = arr.shape[0]
    sync = (arr[:, :12] == np.frombuffer(SYNC, np.uint8)).all(1)
    form2 = (arr[:, 18] & 0x20) != 0
    edc_ok = np.ones(n, bool); ecc_ok = np.ones(n, bool)
    f1 = np.nonzero(~form2)[0]
    if len(f1):
        sub = arr[f1]
        want = edc_batch(sub[:, 16:2072])
        got = sub[:, 2072:2076].copy().view("<u4")[:, 0]
        edc_ok[f1] = want == got
        par = ecc_batch(sub)
        ecc_ok[f1] = (par == sub[:, 2076:2352]).all(1)
    f2 = np.nonzero(form2)[0]
    if len(f2):
        sub = arr[f2]
        got = sub[:, 2348:2352].copy().view("<u4")[:, 0]
        zero = got == 0
        want = edc_batch(sub[:, 16:2348])
        edc_ok[f2] = zero | (want == got)
    return dict(sync=sync, edc=edc_ok, ecc=ecc_ok, form2=form2)

def bcd(n): return ((n // 10) << 4) | (n % 10)
def lba_to_msf(lba):
    lba += 150
    return bcd(lba // 4500), bcd(lba // 75 % 60), bcd(lba % 75)
def msf_to_lba(m, s, f):
    d = lambda x: (x >> 4) * 10 + (x & 15)
    return (d(m) * 60 + d(s)) * 75 + d(f) - 150

def build_sector(lba, sub, payload, form2=None):
    """sub = (file, channel, submode, coding). payload: 2048 (form1) / 2324 (form2) bytes.
    Form is taken from submode bit5 unless form2 given."""
    f, ch, sm, co = sub
    if form2 is None:
        form2 = bool(sm & 0x20)
    hdr = bytes(lba_to_msf(lba)) + b"\x02"
    sh = bytes((f, ch, sm, co)) * 2
    if form2:
        payload = payload.ljust(2324, b"\0")[:2324]
        body = sh + payload
        return SYNC + hdr + body + b"\0\0\0\0"   # original XA audio sectors carry no EDC
    payload = payload.ljust(2048, b"\0")[:2048]
    s = bytearray(SYNC + hdr + sh + payload)
    s += struct.pack("<I", edc(bytes(s[16:2072])))
    par = ecc_batch(np.frombuffer(bytes(s) + b"\0" * 276, np.uint8).reshape(1, 2352))[0]
    return bytes(s) + par.tobytes()

def build_empty(lba, file=1, channel=0):
    """Null sector as in the original's tail: Form1, submode 0, zero data, valid EDC/ECC."""
    return build_sector(lba, (file, channel, 0, 0), b"")

def parse_sector(s):
    return dict(msf=s[12:15], sub=tuple(s[16:20]), sub2=tuple(s[20:24]), data=s[24:])
