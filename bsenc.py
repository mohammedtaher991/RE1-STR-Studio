"""PS1 MDEC BS v2 video frame encoder (+ reference decoder used for self-tests).
Frame data layout (what goes into the STR sectors): 8-byte header + 16-bit-LE-word bitstream.
Header: u16 mdec_words, u16 0x3800, u16 qscale, u16 version(2)."""
import numpy as np
from bs_tables import AC_CODES

QUANT = np.array([
     2, 16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83], dtype=np.float64)
ZIGZAG = np.array([
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])

# ---- code tables -------------------------------------------------------
LUT_LEN = np.full((64, 1024), 22, dtype=np.int64)
LUT_COD = np.zeros((64, 1024), dtype=np.int64)
for _r in range(64):
    for _l in range(-512, 512):
        LUT_COD[_r, _l & 0x3FF] = (1 << 16) | (_r << 10) | (_l & 0x3FF)       # escape: 000001 + halfword
DECODE = {}
for bits, code, run, lvl in AC_CODES:
    for sign in (0, 1):
        l = -lvl if sign else lvl
        LUT_LEN[run, l & 0x3FF] = bits + 1
        LUT_COD[run, l & 0x3FF] = (code << 1) | sign
        DECODE[(bits + 1, (code << 1) | sign)] = (run, l)

# ---- DCT ---------------------------------------------------------------
_k = np.arange(8)[:, None]; _n = np.arange(8)[None, :]
DCT = np.sqrt(2 / 8) * np.cos((2 * _n + 1) * _k * np.pi / 16)
DCT[0] /= np.sqrt(2)

def rnd(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def planes_to_blocks(Y, Cb, Cr):
    """Y: HxW, Cb/Cr: H/2xW/2 uint8. Returns (N,8,8) float blocks in MDEC order (MBs column-major; Cr,Cb,Y0..Y3)."""
    H, W = Y.shape
    mw, mh = W // 16, H // 16
    Yb = (Y.astype(np.float64) - 128).reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    Cbb = (Cb.astype(np.float64) - 128).reshape(H // 16, 8, W // 16, 8).transpose(0, 2, 1, 3)
    Crb = (Cr.astype(np.float64) - 128).reshape(H // 16, 8, W // 16, 8).transpose(0, 2, 1, 3)
    out = np.empty((mw, mh, 6, 8, 8))
    for mx in range(mw):
        out[mx, :, 0] = Crb[:, mx]
        out[mx, :, 1] = Cbb[:, mx]
        out[mx, :, 2] = Yb[0::2, 2 * mx]
        out[mx, :, 3] = Yb[0::2, 2 * mx + 1]
        out[mx, :, 4] = Yb[1::2, 2 * mx]
        out[mx, :, 5] = Yb[1::2, 2 * mx + 1]
    return out.reshape(-1, 8, 8)

def fdct(blocks):
    return 8.0 * np.einsum("ij,njk,lk->nil", DCT, blocks, DCT)      # 8x orthonormal (jpeg islow scale)

# ---- encoder -----------------------------------------------------------
class FrameCoeffs:
    def __init__(self, Y, Cb, Cr):
        F = fdct(planes_to_blocks(Y, Cb, Cr)).reshape(-1, 64)
        self.n = F.shape[0]
        self.dc = np.clip(rnd(F[:, 0] / 16.0), -512, 510).astype(np.int64)
        self.F = F[:, ZIGZAG[1:]]                                   # AC in scan order

    def symbols(self, qs):
        q = np.clip(rnd(self.F / (QUANT[ZIGZAG[1:]] * qs)), -512, 510).astype(np.int64)
        b, pos = np.nonzero(q)                                       # row-major => ordered by block then scan pos
        lvl = q[b, pos]
        prev = np.empty_like(pos); prev[:] = -1
        same = np.r_[False, b[1:] == b[:-1]]
        prev[1:] = np.where(same[1:], pos[:-1], -1)
        run = pos - prev - 1
        return b, pos, run, lvl

    def size_bits(self, qs):
        b, pos, run, lvl = self.symbols(qs)
        return int(LUT_LEN[run, lvl & 0x3FF].sum()) + 12 * self.n + 10, len(b)

    def pack(self, qs):
        b, pos, run, lvl = self.symbols(qs)
        n = self.n
        ac_len = LUT_LEN[run, lvl & 0x3FF]; ac_cod = LUT_COD[run, lvl & 0x3FF]
        blk = np.arange(n)
        key = np.concatenate([blk * 70, b * 70 + 1 + pos, blk * 70 + 69])
        lens = np.concatenate([np.full(n, 10), ac_len, np.full(n, 2)])
        cods = np.concatenate([self.dc & 0x3FF, ac_cod, np.full(n, 2)])
        o = np.argsort(key, kind="stable")
        lens, cods = lens[o], cods[o]
        lens = np.r_[lens, 10]; cods = np.r_[cods, 0x1FF]           # end-of-frame marker
        total = int(lens.sum())
        starts = np.cumsum(lens) - lens
        sid = np.repeat(np.arange(len(lens)), lens)
        k = np.arange(total) - starts[sid]
        bits = ((cods[sid] >> (lens[sid] - 1 - k)) & 1).astype(np.uint8)
        pad = (-total) % 16
        bits = np.r_[bits, np.zeros(pad, np.uint8)]
        by = np.packbits(bits)                                       # MSB first
        by = by.reshape(-1, 2)[:, ::-1].reshape(-1)                  # each 16-bit word stored little-endian
        hw = len(b) + 2 * n + 2
        hw = (hw + 63) & ~63
        words = (hw + 1) >> 1
        data = np.array([words & 255, words >> 8, 0x00, 0x38, qs & 255, qs >> 8, 2, 0], np.uint8).tobytes() + by.tobytes()
        data += b"\0" * ((-len(data)) % 4)
        return data, words

def encode_frame(Y, Cb, Cr, max_bytes, min_qs=1, coeffs=None):
    """Smallest quantiser scale whose frame fits max_bytes. Returns dict(data, words, qs)."""
    c = coeffs or FrameCoeffs(Y, Cb, Cr)
    def fits(qs):
        bits, _ = c.size_bits(qs)
        return 8 + ((bits + 15) // 16) * 2 + 3 <= max_bytes
    lo, hi = min_qs, 63
    if not fits(hi):
        raise ValueError("frame cannot fit the sector budget even at the coarsest quantiser")
    while lo < hi:
        mid = (lo + hi) // 2
        if fits(mid): hi = mid
        else: lo = mid + 1
    data, words = c.pack(lo)
    assert len(data) <= max_bytes
    return dict(data=data, words=words, qs=lo)

# ---- reference decoder (tests only) ------------------------------------
def decode_frame(data, w, h):
    """Decode BS v2 data (with 8 byte header) into Y,Cb,Cr planes; also returns halfword count used."""
    words, m, qs, ver = np.frombuffer(data[:8], "<u2")
    assert m == 0x3800 and ver == 2, (hex(m), ver)
    bs = np.frombuffer(data[8:8 + (len(data) - 8) // 2 * 2], "<u2")
    bits = np.unpackbits(bs.astype(">u2").view(np.uint8))
    p = 0
    def rd(n):
        nonlocal p
        v = 0
        for _ in range(n):
            v = (v << 1) | int(bits[p]); p += 1
        return v
    nmb = (w // 16) * (h // 16)
    blocks = np.zeros((nmb * 6, 64)); hw = 0
    zz_inv = ZIGZAG
    for bi in range(nmb * 6):
        dc = rd(10)
        if dc == 0x1FF and False: break
        dc = dc - 1024 if dc >= 512 else dc
        blocks[bi, 0] = dc * QUANT[0] * 1.0 / 1 * 1  # dc * 2  (scaled below)
        hw += 1; i = 1
        while True:
            if bits[p] == 1 and bits[p + 1] == 0:
                p += 2; hw += 1; break
            # prefix decode
            for L in range(2, 18):
                code = 0
                for t in range(L): code = (code << 1) | int(bits[p + t])
                if (L, code) in DECODE: break
            else:
                if rd(6) != 1: raise ValueError("bad vlc at bit %d" % p)
                hv = rd(16); run = hv >> 10; l = hv & 0x3FF; l = l - 1024 if l >= 512 else l
                i += run; blocks[bi, ZIGZAG[i]] = l * QUANT[ZIGZAG[i]] * qs / 8; i += 1; hw += 1
                continue
            p += L
            lvl_run, lvl = DECODE[(L, code)]
            i += lvl_run
            blocks[bi, ZIGZAG[i]] = lvl * QUANT[ZIGZAG[i]] * qs / 8; i += 1; hw += 1
    end = rd(10)
    assert end == 0x1FF, hex(end)
    blocks[:, 0] = blocks[:, 0]                      # dc*2 : orthonormal coeff = dc * Q0(2)
    # inverse DCT (orthonormal)
    B = np.einsum("ji,njk,kl->nil", DCT, blocks.reshape(-1, 8, 8), DCT) if False else np.einsum("ki,nkl,lj->nij", DCT, blocks.reshape(-1, 8, 8), DCT)
    mw, mh = w // 16, h // 16
    B = B.reshape(mw, mh, 6, 8, 8)
    Y = np.zeros((h, w)); Cb = np.zeros((h // 2, w // 2)); Cr = np.zeros((h // 2, w // 2))
    for mx in range(mw):
        for my in range(mh):
            Cr[my*8:my*8+8, mx*8:mx*8+8] = B[mx, my, 0]
            Cb[my*8:my*8+8, mx*8:mx*8+8] = B[mx, my, 1]
            Y[my*16:my*16+8, mx*16:mx*16+8] = B[mx, my, 2]
            Y[my*16:my*16+8, mx*16+8:mx*16+16] = B[mx, my, 3]
            Y[my*16+8:my*16+16, mx*16:mx*16+8] = B[mx, my, 4]
            Y[my*16+8:my*16+16, mx*16+8:mx*16+16] = B[mx, my, 5]
    hw += 1
    return (np.clip(Y + 128, 0, 255), np.clip(Cb + 128, 0, 255), np.clip(Cr + 128, 0, 255)), int(words), hw
