"""Reading STR movie files that are not in the ideal layout.

Real-world STR files come as: RIFF/CDXA wrapped, raw 2352-byte sectors, raw sectors with some junk before the first
sync pattern, or 2336-byte sectors ("Mode 2 without sync and header", written by some tools' "save as")."""
import os, struct, tempfile
import numpy as np

SEC = 2352
SYNC = b"\x00" + b"\xff" * 10 + b"\x00"

def _bcd(v): return ((v // 10) << 4) | (v % 10)

def split(raw):
    """-> (riff_header or None, body, layout) where body is made of 2352-byte sectors and layout is
    'cdxa', 'raw', 'offset' (junk before the first sync, removed) or 's2336' (headers rebuilt)."""
    riff = None; body = raw; tag = "raw"
    if raw[:4] == b"RIFF" and raw[8:12] == b"CDXA":
        riff = raw[:44]; body = raw[44:]; tag = "cdxa"
    if body[:12] == SYNC:
        return riff, body, tag
    if len(body) % 2336 == 0 and len(body) >= 2336 and body[:4] == body[4:8] and body[:4] != b"\0\0\0\0":
        return riff, _from2336(body), "s2336"
    if len(body) % 2048 == 0 and len(body) >= 2048 and body[:4] == VMAGIC:
        return riff, _from2048(body), "s2048"
    off = body.find(SYNC, 0, 1 << 16)
    if off > 0:
        return riff, body[off:], "offset"
    return riff, body, tag

def _from2336(body):
    n = len(body) // 2336
    a = np.frombuffer(body, np.uint8).reshape(n, 2336)
    out = np.zeros((n, SEC), np.uint8)
    out[:, 0:12] = np.frombuffer(SYNC, np.uint8)
    lba = np.arange(n) + 150
    out[:, 12] = [_bcd(int(x)) for x in (lba // 75 // 60)]
    out[:, 13] = [_bcd(int(x)) for x in (lba // 75 % 60)]
    out[:, 14] = [_bcd(int(x)) for x in (lba % 75)]
    out[:, 15] = 2
    out[:, 16:] = a
    return out.tobytes()

VMAGIC = b"\x60\x01\x01\x80"

def _from2048(body):
    """Cooked 2048-byte sectors (data only, as copied out of an ISO). Video sectors become Form 1, any other
    non-empty sector is sound: a CD-XA sound sector holds 2324 bytes, so only the first 16 of its 18 sound groups
    survive in such a file and the missing 2 groups are filled with silence."""
    n = len(body) // 2048
    a = np.frombuffer(body, np.uint8).reshape(n, 2048)
    out = np.zeros((n, SEC), np.uint8)
    out[:, 0:12] = np.frombuffer(SYNC, np.uint8)
    lba = np.arange(n) + 150
    out[:, 12] = [_bcd(int(x)) for x in (lba // 75 // 60)]
    out[:, 13] = [_bcd(int(x)) for x in (lba // 75 % 60)]
    out[:, 14] = [_bcd(int(x)) for x in (lba % 75)]
    out[:, 15] = 2
    vid = (a[:, :4] == np.frombuffer(VMAGIC, np.uint8)).all(axis=1)
    null = ~a.any(axis=1)
    aud = ~vid & ~null
    sub = np.zeros((n, 4), np.uint8)
    sub[:, 0] = 1; sub[:, 1] = 1
    sub[vid, 2] = 0x08; sub[vid, 3] = 0x00
    sub[aud, 2] = 0x64; sub[aud, 3] = 0x01
    sub[null, 0] = 0; sub[null, 1] = 0
    out[:, 16:20] = sub; out[:, 20:24] = sub
    out[:, 24:24 + 2048] = a
    # the very last video sector of the file ends the movie
    vi = np.nonzero(vid)[0]
    if len(vi): out[vi[-1], 18] = out[vi[-1], 22] = 0x89
    return out.tobytes()

LAYOUT_TEXT = {"cdxa": "RIFF / CDXA", "raw": "raw 2352-byte sectors", "offset": "raw sectors (extra bytes before the first sector)",
               "s2336": "2336-byte sectors (no sync / address header)",
               "s2048": "2048-byte sectors (data only; sound is incomplete)"}

def looks_like_str(path):
    try:
        with open(path, "rb") as f:
            h = f.read(1 << 16); size = os.path.getsize(path)
    except OSError:
        return False
    if h[:4] == b"RIFF" and h[8:12] == b"CDXA": return True
    if SYNC in h[:4096] and h[:4] != b"RIFF": return True
    if size % 2336 == 0 and h[:4] == h[4:8] and h[:4] != b"\0\0\0\0" and (h[2] & 0x0E) and h[24 - 16 + 0:24 - 16 + 4] != b"":
        return True
    if size % 2048 == 0 and h[:4] == VMAGIC: return True
    return False

def is_clean(path):
    """True if ffmpeg-style readers can use the file as it is."""
    with open(path, "rb") as f: h = f.read(64)
    if h[:4] == b"RIFF" and h[8:12] == b"CDXA": return h[44:56] == SYNC
    return h[:12] == SYNC

def clean_copy(path, dest_dir=None):
    """Return (path_to_use, is_temp). Writes a normalised RIFF-less copy only when needed."""
    if is_clean(path): return path, False
    raw = open(path, "rb").read()
    riff, body, tag = split(raw)
    fd, tmp = tempfile.mkstemp(suffix=".STR", dir=dest_dir)
    with os.fdopen(fd, "wb") as f: f.write(body[: len(body) // SEC * SEC])
    return tmp, True


def ref_is_2336(path):
    """True when the file is stored as headerless 2336-byte sectors (the format RE1 Rebuilder's BUILDCD wants,
    which is what its XA_extract tool writes)."""
    try:
        with open(path, "rb") as f: h = f.read(16)
        size = os.path.getsize(path)
    except OSError:
        return False
    if h[:4] == b"RIFF" or h[:12] == SYNC: return False
    return size % 2336 == 0 and size >= 2336 and h[:4] == h[4:8] and h[:4] != b"\0\0\0\0"

def to_2336(src, dst):
    """Rewrite a normal STR (2352-byte sectors, RIFF wrapper allowed) as headerless 2336-byte sectors."""
    raw = open(src, "rb").read()
    if raw[:4] == b"RIFF" and raw[8:12] == b"CDXA": raw = raw[44:]
    n = len(raw) // SEC
    a = np.frombuffer(raw[: n * SEC], np.uint8).reshape(n, SEC)
    with open(dst, "wb") as f: f.write(np.ascontiguousarray(a[:, 16:]).tobytes())
    return n
