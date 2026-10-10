"""Reference-driven PS1 STR (CD-XA) demuxer/muxer.
The reference file defines everything the game cares about: resolution, file/channel numbers,
submode/coding bytes, audio slot cadence, per-frame sector budget cycle, end-of-stream flag."""
import struct
from psxcd import build_sector, msf_to_lba

SEC = 2352
CHUNK = 2016          # video payload bytes per sector (2048 - 32 byte header)
HDR = 32

def split_riff(data):
    if data[:4] == b"RIFF" and data[8:12] == b"CDXA":
        return data[:44], data[44:]
    return None, data

DEFAULT_RIFF = bytes.fromhex("52494646000000004344584166 6d7420100000000000000025555841010000000000000064617461 00000000".replace(" ", ""))

class Profile:
    pass

def demux(path):
    """Return (profile, frames, audio). frames: list of dict(num,cnt,words,qs,ver,data); audio: list of 2324-byte payloads."""
    import strio
    raw = open(path, "rb").read()
    riff, body, _layout = strio.split(raw)
    n = len(body) // SEC
    P = Profile()
    P.riff = riff
    P.start_lba = msf_to_lba(*body[12:15])
    P.audio_period, P.audio_slot = 8, 7
    frames, audio, order = {}, [], []
    P.null_sub = None; P.vid_sub = None; P.aud_sub = None; P.last_sub = None
    for i in range(n):
        s = body[i * SEC:(i + 1) * SEC]
        sub = tuple(s[16:20])
        sm = sub[2]
        if s[24:28] == b"\x60\x01\x01\x80" or (sm & 0x02 and not sm & 0x04):   # video
            magic, idx, cnt, fn, size, w, h, words, m38, qs, ver, _ = struct.unpack("<IHHIIHHHHHHI", s[24:56])
            f = frames.setdefault(fn, dict(num=fn, cnt=cnt, size=size, words=words, m38=m38, qs=qs, ver=ver, w=w, h=h, chunks={}, start=i))
            f["chunks"][idx] = s[56:56 + CHUNK]
            P.w, P.h, P.magic = w, h, magic
            if i == n - 1 or sm & 0x80:
                P.last_sub = sub
            else:
                P.vid_sub = P.vid_sub or sub
        elif sm & 0x04:
            audio.append(s[24:24 + 2324]); P.aud_sub = P.aud_sub or sub
        else:
            P.null_sub = P.null_sub or sub
    if P.vid_sub is None and P.last_sub: P.vid_sub = (P.last_sub[0], P.last_sub[1], P.last_sub[2] & ~0x80, P.last_sub[3])
    if P.last_sub is None and P.vid_sub: P.last_sub = P.vid_sub          # the original has no end-of-movie flag: copy that
    if P.aud_sub is None and P.vid_sub: P.aud_sub = (P.vid_sub[0], P.vid_sub[1], 0x64, 0x01)   # original has no sound: a normal stereo 37800 Hz XA sector
    if P.null_sub is None and P.vid_sub: P.null_sub = (P.vid_sub[0], 0, 0, 0)   # the original has no empty sectors: use a plain empty one
    out = []
    for fn in sorted(frames):
        f = frames[fn]
        f["data"] = b"".join(f["chunks"].get(k, b"\0" * CHUNK) for k in range(f["cnt"]))[:f["size"]]
        out.append(f)
    P.budgets = [f["cnt"] for f in out]
    P.nframes = len(out)
    P.m38 = out[0]["m38"]; P.ver = out[0]["ver"]
    P.cycle = find_cycle(P.budgets)
    P.total_sectors = n
    # sectors per second at 2x = 150; frame period = avg (video + audio) sectors per frame
    avg_v = sum(P.cycle) / len(P.cycle)
    P.sectors_per_frame = avg_v * P.audio_period / (P.audio_period - 1)
    P.fps = 150.0 / P.sectors_per_frame
    return P, out, audio

def find_cycle(b):
    """Repeating sectors-per-frame pattern of a movie. Uses the most common value of every phase, so a few odd frames
    (e.g. the first ones of the file) do not hide the pattern. E.g. the PJ intro is really [9, 9, 9, 8] = exactly 15 fps."""
    from collections import Counter
    n = len(b)
    if n == 0: return [1]
    for p in range(1, 65):
        if p > n: break
        pat = [Counter(b[i::p]).most_common(1)[0][0] for i in range(p)]
        bad = sum(b[i] != pat[i % p] for i in range(n))
        if bad <= max(2, n // 40):
            return pat
    return [round(sum(b) / n)]

def budget_list(P, nframes):
    return [P.cycle[i % len(P.cycle)] for i in range(nframes)]

def mux(P, frames, audio, budgets=None, start_lba=None):
    """frames: list of dict(num, words, qs, data). audio: list of 2324-byte payloads. Returns file bytes."""
    budgets = budgets or budget_list(P, len(frames))
    lba = P.start_lba if start_lba is None else start_lba
    # build video sector queue
    vq = []
    for fr, cnt in zip(frames, budgets):
        d = fr["data"]
        assert len(d) <= cnt * CHUNK, f"frame {fr['num']} too big: {len(d)} > {cnt * CHUNK}"
        size = (len(d) + 3) & ~3
        d = d.ljust(cnt * CHUNK, b"\0")
        for k in range(cnt):
            hdr = struct.pack("<IHHIIHHHHHHI", P.magic, k, cnt, fr["num"], size, P.w, P.h,
                              fr["words"], P.m38, fr["qs"], fr.get("ver", P.ver), 0)
            vq.append(hdr + d[k * CHUNK:(k + 1) * CHUNK])
    out = []; vi = ai = 0; i = 0
    while vi < len(vq):
        if i % P.audio_period == P.audio_slot:
            if ai < len(audio):
                out.append(build_sector(lba + i, P.aud_sub, audio[ai])); ai += 1
            else:
                out.append(build_sector(lba + i, P.null_sub, b""))
        else:
            sub = P.vid_sub
            if vi == len(vq) - 1:
                sub = P.last_sub
            out.append(build_sector(lba + i, sub, vq[vi])); vi += 1
        i += 1
    body = b"".join(out)
    if P.riff:
        r = bytearray(P.riff)
        struct.pack_into("<I", r, 4, len(body) + 36)
        struct.pack_into("<I", r, 40, len(body))
        body = bytes(r) + body
    return body
