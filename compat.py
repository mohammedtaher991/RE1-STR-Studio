"""Compatibility check and repair of a modded STR against the ORIGINAL game STR.

analyze(mod, ref)  -> report with one row per property (ok / warn / bad) and the fix that addresses it
fix(mod, ref, out, opts) -> new STR; keeps the video untouched (lossless) whenever the layout allows it,
                            re-encodes only when it has to (size, BS version, frames too big)."""
import os, struct, subprocess
import numpy as np
import tools, engine
from psxcd import check_sectors
from mux import SEC, CHUNK, find_cycle, split_riff, DEFAULT_RIFF, demux, mux as mux_streams

HDR = np.dtype([("magic", "<u4"), ("idx", "<u2"), ("cnt", "<u2"), ("fn", "<u4"), ("size", "<u4"), ("w", "<u2"), ("h", "<u2"),
                ("words", "<u2"), ("m38", "<u2"), ("qs", "<u2"), ("ver", "<u2"), ("pad", "<u4")])
COMMON_FPS = (10, 12, 15, 20, 24, 25, 29.97, 30)

def _bcd(a): a = a.astype(np.int64); return (a >> 4) * 10 + (a & 15)

def _audio_fmt(co):
    stereo = bool(co & 1); rate = 18900 if co & 4 else 37800; bits = 8 if co & 0x10 else 4
    per_sector = {(4, True): 2016, (4, False): 4032, (8, True): 1008, (8, False): 2016}[(bits, stereo)]
    return dict(stereo=stereo, rate=rate, bits=bits, per_sector=per_sector)

def _fmt_audio(co):
    f = _audio_fmt(co); return f"{f['rate']} Hz, {'stereo' if f['stereo'] else 'mono'}, {f['bits']}-bit"

# ------------------------------------------------------------------------------------------------ loose parser
class Loose:
    pass

def parse_loose(path):
    """Read any CD-XA movie file without assuming it is well formed."""
    raw = open(path, "rb").read()
    L = Loose(); L.path = path; L.bytes = len(raw)
    import strio
    riff, body, layout = strio.split(raw)
    L.riff = riff; L.layout = layout
    n = len(body) // SEC
    if n == 0: raise ValueError("file is too small to be an STR")
    L.trailing = len(body) - n * SEC
    arr = np.frombuffer(body[:n * SEC], np.uint8).reshape(n, SEC)
    L.arr = arr; L.n = n
    chk = check_sectors(arr)
    L.bad_sync = int((~chk["sync"]).sum()); L.bad_edc = int((~chk["edc"]).sum()); L.bad_ecc = int((~chk["ecc"]).sum())
    L.bad_sub = int((arr[:, 16:20] != arr[:, 20:24]).any(1).sum())
    L.bad_mode = int((arr[:, 15] != 2).sum())
    lba = (_bcd(arr[:, 12]) * 60 + _bcd(arr[:, 13])) * 75 + _bcd(arr[:, 14]) - 150
    L.first_lba = int(lba[0]); L.lba_continuous = bool((np.diff(lba) == 1).all()) if n > 1 else True
    sm = arr[:, 18]
    is_aud = (sm & 4) != 0
    is_vid = (arr[:, 24:28] == np.array([0x60, 0x01, 0x01, 0x80], np.uint8)).all(1) & ~is_aud
    vid_idx = np.nonzero(is_vid)[0]
    aud_idx = np.nonzero(is_aud)[0]
    L.n_video_sectors = len(vid_idx); L.n_audio_sectors = len(aud_idx); L.n_null = int(n - len(vid_idx) - len(aud_idx))
    # ---- frames and clips
    frames = []
    if len(vid_idx):
        hd = np.ascontiguousarray(arr[vid_idx, 24:56]).view(HDR)[:, 0]
        cur = None; clip = -1; prev = None
        for k, i in enumerate(vid_idx):
            h = hd[k]; fn = int(h["fn"]); ix = int(h["idx"])
            if cur is None or fn != cur["fn"] or ix in cur["chunks"]:
                cur = dict(fn=fn, cnt=int(h["cnt"]), size=int(h["size"]), words=int(h["words"]), qs=int(h["qs"]), ver=int(h["ver"]),
                           m38=int(h["m38"]), w=int(h["w"]), h=int(h["h"]), magic=int(h["magic"]), chunks={}, first=int(i))
                if prev is None or fn != prev + 1: clip += 1
                cur["clip"] = clip; prev = fn; frames.append(cur)
            cur["chunks"][ix] = int(i); cur["last"] = int(i)
    for f in frames:
        f["complete"] = f["cnt"] > 0 and set(f["chunks"]) == set(range(f["cnt"]))
    L.frames = frames
    L.clips = max((f["clip"] for f in frames), default=-1) + 1
    L.vid_subs = {}
    for i in vid_idx[:4000]:
        t = tuple(int(x) for x in arr[i, 16:20]); L.vid_subs[t] = L.vid_subs.get(t, 0) + 1
    L.vid_sub = max(L.vid_subs, key=L.vid_subs.get) if L.vid_subs else None
    L.last_vid_sub = tuple(int(x) for x in arr[vid_idx[-1], 16:20]) if len(vid_idx) else None
    L.first_vid, L.last_vid = (int(vid_idx[0]), int(vid_idx[-1])) if len(vid_idx) else (0, 0)
    # ---- audio streams
    L.audio = {}
    for i in aud_idx:
        key = (int(arr[i, 16]), int(arr[i, 17]), int(arr[i, 19])); L.audio.setdefault(key, []).append(int(i))
    L.main_audio = max(L.audio, key=lambda k: len(L.audio[k])) if L.audio else None
    L.aud_sub = None
    if L.main_audio:
        i0 = L.audio[L.main_audio][0]; L.aud_sub = tuple(int(x) for x in arr[i0, 16:20])
    return L

def choose_audio(L, ref_key=None):
    """Pick the audio stream that belongs to the movie: the one on the reference's channel, else the biggest."""
    if ref_key:
        for k in L.audio:
            if (k[0], k[1]) == (ref_key[0], ref_key[1]): return k
    return L.main_audio

def timing(L, ref_audio_key=None):
    """-> dict(fps, spf, sps, audio_period, video_seconds, audio_seconds)."""
    n = len(L.frames)
    t = dict(fps=None, spf=None, sps=150.0, period=None, video_seconds=None, audio_seconds=None, sps_guess=True)
    key = choose_audio(L, ref_audio_key)
    if key:
        fm = _audio_fmt(key[2]); idx = L.audio[key]
        period = int(round(float(np.median(np.diff(idx))))) if len(idx) > 2 else 8
        t["period"] = period; t["sps"] = period * fm["rate"] / fm["per_sector"]; t["sps_guess"] = False
        t["audio_seconds"] = len(idx) * fm["per_sector"] / fm["rate"]
    if n >= 2:
        t["spf"] = (L.last_vid - L.first_vid + 1) / n
        fps = t["sps"] / t["spf"]
        t["fps_nominal"] = next((float(c) for c in COMMON_FPS if abs(fps - c) / c < 0.04), fps)
        t["fps"] = fps; t["video_seconds"] = n / fps
    return t

def audio_clock(L, akey, positions):
    """Real time (seconds) of sector positions, measured by the SOUND: every sound sector is a fixed slice of time (2016 samples at
    37800 Hz), and a muxer places the picture sectors that belong to that slice next to it. This stays right even when the
    spacing of the sound changes inside the file (every 4 sectors in one part, every 8 in another)."""
    fm = _audio_fmt(akey[2]); dur = fm["per_sector"] / fm["rate"]
    ai = np.asarray(L.audio[akey], float)
    return np.interp(np.asarray(positions, float), ai, np.arange(len(ai)) * dur)

def audio_spacing_mixed(L, akey):
    d = np.diff(np.asarray(L.audio[akey])) if akey and len(L.audio.get(akey, [])) > 2 else []
    vals, cnt = np.unique(d, return_counts=True) if len(d) else ([], [])
    big = [int(v) for v, n in zip(vals, cnt) if n > max(10, 0.02 * len(d))]
    return big

def clip_rates(L):
    """Frame rate of every clip, measured from its sector span (video + sound sectors)."""
    out = []
    for c in range(L.clips):
        fs = [f for f in L.frames if f["clip"] == c and f["complete"]]
        if len(fs) < 3: continue
        span = fs[-1]["first"] - fs[0]["first"]
        out.append(150.0 * (len(fs) - 1) / span if span > 0 else 0.0)
    return [x for x in out if x > 0]

def plan_budgets(sizes, cycle, max_drift=1):
    if callable(cycle): cycfn = cycle
    else: cycfn = lambda k: cycle[k % len(cycle)]
    """Sectors per frame so the layout follows the original's pattern. The original itself is allowed to run
    at most `max_drift` sector ahead of the pattern. Returns a list, or None if some frame cannot fit."""
    out, drift = [], 0
    for k, size in enumerate(sizes):
        b = cycfn(k); need = max(1, -(-size // CHUNK))
        if need > b:
            if drift + (need - b) > max_drift: return None
            drift += need - b; b = need
        elif drift > 0 and need <= b - 1:
            b -= 1; drift -= 1
        out.append(b)
    return out

def frame_data(L, f):
    parts = [L.arr[f["chunks"][c], 56:56 + CHUNK].tobytes() for c in range(f["cnt"])]
    return b"".join(parts)[:min(f["size"], f["cnt"] * CHUNK)]

# ------------------------------------------------------------------------------------------------ analysis
OK, WARN, BAD = "ok", "warn", "bad"

def analyze_video(mod_path, ref_path):
    """The 'modded' file is an ordinary video (mp4, avi...): nothing to check in the STR sense, it just gets converted."""
    import engine
    R = parse_loose(ref_path); rt = timing(R)
    v = engine.probe_video(mod_path); vv, au = v["video"], v["audio"]
    rw, rh = R.frames[0]["w"], R.frames[0]["h"]
    rows = []
    def add(id, label, ref, mod, status, note=""):
        rows.append(dict(id=id, label=label, ref=str(ref), mod=str(mod), status=status, note=note, fix="reencode"))
    add("kind", "File type", "STR movie", f"ordinary video ({v['container'].split(',')[0]})", WARN, "Not an STR yet. Adjusting converts it into a game-ready STR.")
    add("size", "Picture size", f"{rw}x{rh}", f"{vv['width']}x{vv['height']}", OK if (vv["width"], vv["height"]) == (rw, rh) else WARN,
        "" if (vv["width"], vv["height"]) == (rw, rh) else "Fitted to the game's size (see Fit picture below).")
    add("fps", "Frame rate", f"{rt['fps']:g} fps", f"{vv['fps']:g} fps", OK if abs(vv["fps"] - rt["fps"]) < 0.5 else WARN,
        "" if abs(vv["fps"] - rt["fps"]) < 0.5 else "Frames are re-timed to the original's rate.")
    add("aud", "Sound", "yes" if R.main_audio else "none", "yes" if au.get("present") else "none", OK if au.get("present") or not R.main_audio else WARN,
        "" if au.get("present") or not R.main_audio else "No sound track: the movie will be silent.")
    dur = v["duration"] or 0; cap = R.n / 150.0
    add("fit", "Length", f"{cap:.1f} s", f"{dur:.1f} s", OK if dur <= cap + 0.05 else WARN, "" if dur <= cap + 0.05 else "Longer than the original: it is cut to fit.")
    return dict(ref=dict(name=os.path.basename(ref_path), sectors=R.n, fps=rt["fps"], fps_nominal=rt.get("fps_nominal"), frames=len(R.frames), seconds=rt["video_seconds"]),
                mod=dict(name=os.path.basename(mod_path), kind="video", fps=vv["fps"], video_seconds=dur, fps_guess=True),
                rows=rows, bad=0, warn=sum(r["status"] == WARN for r in rows), needs_reencode=True, verdict="convert")

def analyze(mod_path, ref_path):
    import engine
    if not engine.looks_like_str(mod_path): return analyze_video(mod_path, ref_path)
    R = parse_loose(ref_path); M = parse_loose(mod_path)
    rt = timing(R); rkey = choose_audio(R)
    mt = timing(M, rkey); mkey = choose_audio(M, rkey)
    rb = [f["cnt"] for f in R.frames]; rcycle = find_cycle(rb)
    mb = [f["cnt"] for f in M.frames]
    rows = []
    def add(id, label, ref, mod, status, note="", fix=None):
        rows.append(dict(id=id, label=label, ref=str(ref), mod=str(mod), status=status, note=note, fix=fix))
    rf0, mf0 = (R.frames[0] if R.frames else None), (M.frames[0] if M.frames else None)
    if not M.frames:
        raise ValueError("No video frames were found in this file. It does not look like a PlayStation STR movie.")
    # container / integrity
    if M.layout == R.layout and M.layout == "s2336":
        import strio
        add("layout", "File layout", strio.LAYOUT_TEXT[R.layout], strio.LAYOUT_TEXT[M.layout], OK,
            "Same as the original: the layout RE1 Rebuilder's BUILDCD and XA_extract use.", "")
    elif M.layout in ("s2336", "s2048", "offset"):
        import strio
        add("layout", "File layout", strio.LAYOUT_TEXT[R.layout], strio.LAYOUT_TEXT[M.layout], BAD,
            "Not a normal STR layout, so most tools (and the game) cannot read it. It is rewritten as normal sectors.", "repack")
    add("container", "File container", "RIFF / CDXA" if R.riff else "raw sectors", "RIFF / CDXA" if M.riff else "raw sectors",
        OK if bool(R.riff) == bool(M.riff) else WARN, "Some injection tools need a particular one." , "container")
    dmg = M.bad_sync + M.bad_edc + M.bad_ecc + M.bad_sub + M.bad_mode
    add("integrity", "Sector checksums", "all valid", "all valid" if not dmg else f"{max(M.bad_edc, M.bad_ecc, M.bad_sync, M.bad_sub, M.bad_mode)} damaged",
        OK if not dmg else BAD, "" if not dmg else "Broken error-correction data or headers. The disc image may refuse them. They are rebuilt.", "rebuild")
    add("lba_cont", "Sector numbering", "continuous", "continuous" if M.lba_continuous else "gaps / jumps", OK if M.lba_continuous else BAD,
        "" if M.lba_continuous else "Sector timecodes jump around; rebuilt in order.", "rebuild")
    if M.trailing: add("trailing", "Extra bytes at the end", "none", f"{M.trailing} bytes", WARN, "Cut off.", "rebuild")
    add("lba", "First sector (LBA)", R.first_lba, M.first_lba, OK if R.first_lba == M.first_lba else WARN,
        "Only matters if your injection tool does not rewrite it.", "lba")
    # video stream
    rv = (R.vid_sub[0], R.vid_sub[1]); mv = (M.vid_sub[0], M.vid_sub[1])
    add("vid_ch", "Video CD-XA file / channel", f"{rv[0]} / {rv[1]}", f"{mv[0]} / {mv[1]}", OK if rv == mv else BAD,
        "The game only reads this file/channel number." if rv != mv else "", "flags")
    add("vid_flags", "Video sector flags", f"0x{R.vid_sub[2]:02X} / 0x{R.vid_sub[3]:02X}", f"0x{M.vid_sub[2]:02X} / 0x{M.vid_sub[3]:02X}",
        OK if R.vid_sub[2:] == M.vid_sub[2:] else WARN, "", "flags")
    add("size", "Video size", f"{rf0['w']}x{rf0['h']}", f"{mf0['w']}x{mf0['h']}", OK if (rf0["w"], rf0["h"]) == (mf0["w"], mf0["h"]) else BAD,
        "" if (rf0["w"], rf0["h"]) == (mf0["w"], mf0["h"]) else "The game expects this exact picture size. The video must be re-encoded.", "reencode")
    add("bs", "Video format (BS version)", f"v{rf0['ver']}", f"v{mf0['ver']}", OK if rf0["ver"] == mf0["ver"] else BAD,
        "" if rf0["ver"] == mf0["ver"] else "Different video compression; must be re-encoded.", "reencode")
    # frame rate / layout
    rfps, mfps = rt["fps"], mt["fps"]
    fps_ok = rfps and mfps and abs(rfps - mfps) / rfps < 0.01
    def fshow(t):
        if not t["fps"]: return "?"
        nom = t.get("fps_nominal") or t["fps"]
        return f"{nom:g} fps" + (f" ({t['fps']:.2f} exactly)" if abs(nom - t["fps"]) > 0.02 else "")
    add("fps", "Frame rate", fshow(rt), fshow(mt), OK if fps_ok else BAD,
        "" if fps_ok else "Plays too fast or too slow in the game. Frames are re-timed to the original's rate.", "retime")
    if mkey and mt.get("sps") and not mt.get("sps_guess"):
        sps = mt["sps"]
        add("cdspeed", "CD speed the file is laid out for", "2x (150 sectors/s)", f"{'1x' if sps < 110 else '2x'} ({sps:.0f} sectors/s)", OK if sps > 110 else BAD,
            "" if sps > 110 else "Laid out for 1x speed. The game reads at 2x, so the picture would play twice too fast. It is re-timed.", "retime")
    ravg, mavg = sum(rcycle) / len(rcycle), sum(mb) / len(mb)
    import itertools
    same_pat = bool(mb) and max(abs(a - b) for a, b in zip(itertools.accumulate(mb), itertools.accumulate(rcycle[i % len(rcycle)] for i in range(len(mb))))) <= 1
    add("budget", "Sectors per frame", f"{ravg:.2f} (pattern {'/'.join(map(str, rcycle[:8]))}…)", f"{mavg:.2f}", OK if same_pat else WARN,
        "" if same_pat else "Different sector layout than the original. It is rebuilt with the original's pattern.", "repack")
    contiguous = [f["fn"] for f in M.frames] == list(range(1, len(M.frames) + 1))
    if M.clips > 1:
        add("clips", "Clips in the file", "1", f"{M.clips} separate clips (" + ", ".join(f"{x:.1f} fps" for x in clip_rates(M)) + ")", BAD,
            "The frame numbers restart, so the game sees separate pieces. They are joined into one movie, and each clip is brought to the right speed.", "repack")
    else:
        add("numbering", "Frame numbering", "1 … N in order", "1 … N in order" if contiguous else "irregular", OK if contiguous else BAD,
            "" if contiguous else "Renumbered.", "repack")
    broken = sum(1 for f in M.frames if not f["complete"])
    if broken: add("incomplete", "Incomplete frames", "0", str(broken), BAD, "Frames with missing sectors are dropped.", "repack")
    over = 0
    done = [f["size"] for f in M.frames if f["complete"]]
    if not fps_ok:
        pass       # a different frame rate gets its own sector layout, so the original's per-frame space does not apply
    elif plan_budgets(done, rcycle) is None:
        over = sum(1 for k, z in enumerate(done) if z > (rcycle[k % len(rcycle)] + 1) * CHUNK) or 1
    add("oversize", "Frames that fit the original's budget", "all", "all" if not over else f"{over} too big", OK if not over else BAD,
        "" if not over else "Too much data for the original's per-frame space; those need re-compression.", "reencode")
    # audio
    rA, mA = R.main_audio, mkey
    if rA and not mA:
        add("aud_present", "Sound", "yes", "none", WARN, "The movie will play silent.", "audio")
    elif rA:
        add("aud_ch", "Sound CD-XA file / channel", f"{rA[0]} / {rA[1]}", f"{mA[0]} / {mA[1]}", OK if rA[:2] == mA[:2] else BAD,
            "" if rA[:2] == mA[:2] else "The game listens on the original's channel: sound would be MUTED.", "flags")
        add("aud_fmt", "Sound format", _fmt_audio(rA[2]), _fmt_audio(mA[2]), OK if rA[2] == mA[2] else BAD,
            "" if rA[2] == mA[2] else "Different sample rate / channels / bits; the sound is converted.", "audio")
        add("aud_period", "Sound spacing", f"every {rt['period']} sectors", f"every {mt['period']} sectors", OK if rt["period"] == mt["period"] else WARN,
            "", "repack")
        if sum(1 for k, v in M.audio.items() if k != mkey and (len(v) > 20 or k[2] != 0)) > 0:
            add("aud_multi", "Sound channels in the file", "1", str(len(M.audio)), WARN, "Only the one on the original's channel is kept.", "flags")
        if mt["audio_seconds"] and mt["video_seconds"]:
            d = mt["video_seconds"] - mt["audio_seconds"]
            if rt["audio_seconds"] and rt["video_seconds"]:   # the original's own audio/video ratio is "normal"
                d = d - (rt["video_seconds"] - rt["audio_seconds"])   # the original's own gap is "normal"
            rel = abs(d) / max(mt["video_seconds"], 0.1)
            ok = rel < 0.03 or abs(d) < 0.45
            add("sync", "Video vs sound length", "equal", f"video {mt['video_seconds']:.2f} s / sound {mt['audio_seconds']:.2f} s",
                OK if ok else BAD, "" if ok else ("The picture is shorter than the sound (looks sped up)." if d < 0 else "The picture is longer than the sound (looks slowed down)."), "retime")
    # end flag
    ref_end = R.last_vid_sub[2] ^ R.vid_sub[2]; mod_end = M.last_vid_sub[2] ^ M.vid_sub[2]
    add("end", "End-of-movie marker", "present" if ref_end else "none", "present" if mod_end else "missing", OK if bool(ref_end) == bool(mod_end) else WARN,
        "", "flags")
    # size vs original
    if M.n > R.n:
        add("fit", "Sectors vs original", f"{R.n}", f"{M.n} (+{M.n - R.n})", BAD, "Larger than the original file: it cannot replace it in place. The end is cut off, or choose Speed up to fit to keep all of it.", "fit")
    else:
        add("fit", "Sectors vs original", f"{R.n}", f"{M.n}", OK if M.n == R.n else WARN, "" if M.n == R.n else "Shorter. It is padded to the exact size.", "pad")
    bad = sum(r["status"] == BAD for r in rows); warn = sum(r["status"] == WARN for r in rows)
    needs_reenc = any(r["id"] in ("size", "bs", "oversize") and r["status"] == BAD for r in rows)
    return dict(ref=dict(name=os.path.basename(ref_path), sectors=R.n, fps=rfps, fps_nominal=rt.get("fps_nominal"), frames=len(R.frames), seconds=rt["video_seconds"]),
                mod=dict(name=os.path.basename(mod_path), kind="str", sectors=M.n, fps=mfps, frames=len(M.frames), clips=M.clips,
                         video_seconds=mt["video_seconds"], audio_seconds=mt["audio_seconds"], fps_guess=mt["sps_guess"]),
                rows=rows, bad=bad, warn=warn, needs_reencode=needs_reenc,
                verdict="ok" if not bad and not warn else ("fix" if bad else "minor"))

# ------------------------------------------------------------------------------------------------ repair
DEFAULTS = dict(source=None, overflow="trim", audio_limit=False, fps_mode="auto", out_fps=None, flags=True, container="ref", lba=True, retime=True, speed=1.0, src_fps=None, audio="auto", audio_offset_ms=0.0,
                audio_gain_db=0.0, pad=True, allow_larger=False, reencode="auto", fit="pad")

class FixError(Exception): pass

def _decode_audio(path, nsamples, offset_ms, gain_db, limit=False, speed=1.0):
    flt = []
    if abs(speed - 1) > 1e-6: flt.append(f"atempo={speed}")
    if offset_ms > 0: flt.append(f"adelay={int(offset_ms)}|{int(offset_ms)}")
    elif offset_ms < 0: flt.append(f"atrim=start={-offset_ms / 1000:.4f},asetpts=PTS-STARTPTS")
    if gain_db: flt.append(f"volume={gain_db}dB")
    cmd = ["-v", "error", "-nostdin", "-i", path, "-vn"] + (["-af", ",".join(flt)] if flt else []) + ["-ar", "37800", "-ac", "2", "-f", "s16le", "-"]
    r = tools.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pcm = np.frombuffer(r.stdout[:len(r.stdout) // 4 * 4], "<i2").reshape(-1, 2)
    if limit:
        import re1str
        pcm, _g = re1str.limit_peak(pcm)
    out = np.zeros((nsamples, 2), np.int16); m = min(nsamples, len(pcm)); out[:m] = pcm[:m]
    return out

def _fix_inner(mod_path, ref_path, out_path, opts=None, log=print, progress=None, cancel=None):
    import re1str, xaenc
    o = dict(DEFAULTS); o.update(opts or {})
    progress = progress or (lambda f, t="": None); cancel = cancel or (lambda: False)
    if tools.ffmpeg_path() is None: tools.need_ffmpeg()
    if o.get("source"):          # process from the ORIGINAL video (mp4, avi...) instead of the already converted STR
        if not os.path.isfile(o["source"]): raise FixError("The original video was not found: " + str(o["source"]))
        log(f"processing from the original video {os.path.basename(o['source'])} (the modded STR is not used for picture or sound)")
        mod_path = o["source"]; o["src_fps"] = None; o["retime"] = True
    progress(0.02, "reading both files")
    R = parse_loose(ref_path); M = parse_loose(mod_path)
    P, _, _ = demux(ref_path)
    rt = timing(R); mkey_pref = choose_audio(R)
    mt = timing(M, mkey_pref); akey = choose_audio(M, mkey_pref)
    rcyc = P.cycle
    is_str = engine.looks_like_str(mod_path)
    src_fps = float(o["src_fps"]) if o["src_fps"] else (mt["fps"] if is_str else None)
    ref_fps = P.fps              # exact rate of the original's sector pattern (the measured value is rt["fps"])
    _unused = rt["fps"]          # the real playing rate of the original's sector layout (e.g. 14.58 for a "15 fps" movie)
    out_fps = None
    if o["fps_mode"] == "custom" and o["out_fps"]: out_fps = float(o["out_fps"])
    elif o["fps_mode"] == "auto":            # same frame rate as the modded file itself
        crates = clip_rates(M) if is_str else []
        if len(crates) > 1 and max(crates) / max(min(crates), 0.01) > 1.02:
            mf = None
            log("the clips run at different speeds (" + ", ".join(f"{x:.1f}" for x in crates) + " fps): all of them are brought to the original's frame rate")
        elif is_str: mf = src_fps
        else:
            try: mf = engine.probe_video(mod_path)["video"]["fps"]
            except Exception: mf = None
        if mf:
            mf = max(1.0, min(30.0, float(mf)))
            out_fps = next((float(c) for c in COMMON_FPS + (23.976,) if abs(mf - c) / c < 0.004), round(mf, 3))
    custom = bool(out_fps) and abs(out_fps / ref_fps - 1) > 0.002
    if custom: ref_fps = out_fps
    bud = re1str.exact_bud(out_fps, P.audio_period) if custom else (lambda i: rcyc[i % len(rcyc)])
    frames = [f for f in M.frames if f["complete"]]
    if is_str: log(f"modded file: {len(M.frames)} frames in {M.clips} clip(s), about {src_fps:.2f} fps; the original plays at {rt['fps']:.2f} fps")
    else: log(f"modded file is an ordinary video; the original plays at {rt['fps']:.2f} fps")
    log(f"output frame rate: {ref_fps:.3f} fps" + (" (same as the modded file; own sector layout)" if custom else " (the original's own timing)"))
    # ---- decide lossless / re-encode
    reasons = []
    w0, h0 = R.frames[0]["w"], R.frames[0]["h"]
    if any((f["w"], f["h"]) != (w0, h0) for f in frames): reasons.append("picture size differs")
    if any(f["ver"] != P.ver for f in frames): reasons.append("video format (BS version) differs")
    if not is_str: frames = []; reasons.append("ordinary video file")
    elif not frames: reasons.append("no complete frames")
    # too long for the original? either cut the end (default) or speed everything up so all of it fits
    if o["overflow"] == "speedup" and not o["allow_larger"]:
        try:
            if is_str and frames:
                tsrc = (M.last_vid + 1 - M.first_vid) / 150.0
            elif not is_str:
                tsrc = engine.probe_video(mod_path)["duration"] or 0
            else: tsrc = 0
        except Exception: tsrc = 0
        cap_t = R.n / 150.0 * 0.99
        if tsrc > cap_t * 1.003:
            o["speed"] = float(o["speed"]) * tsrc / cap_t
            log(f"the modded movie is {tsrc:.1f} s but the original only has room for {cap_t:.1f} s: playing it {o['speed']:.3f}x faster so all of it fits")
    # frame selection (re-timing): by REAL TIME (sector position), so clips that run at different speeds all play at the right speed
    fidx = [i for i, f in enumerate(M.frames) if f["complete"]]
    need_retime = is_str and frames and (o["retime"] or abs(o["speed"] - 1) > 1e-6) and (M.clips > 1 or abs(o["speed"] - 1) > 1e-6 or
                                                         (src_fps and ref_fps and abs(src_fps - ref_fps) / ref_fps > 0.01))
    if o["src_fps"] and is_str and frames:      # the user told us the real rate: trust it instead of the sector timing
        need_retime = need_retime and True
    if is_str and frames and akey and (len(audio_spacing_mixed(M, akey)) > 1 or mt.get("period") not in (None, 8)): need_retime = bool(o["retime"])
    if need_retime:
        mixed = audio_spacing_mixed(M, akey) if akey else []
        if o["src_fps"]:
            tt = np.arange(len(frames)) / float(o["src_fps"])
        elif akey and (len(mixed) > 1 or mt.get("period") not in (None, 8)):
            tt = audio_clock(M, akey, [f["first"] for f in frames])
            tt = tt - tt[0]
            log("the sound is spaced differently in different parts of the file (every " + " and every ".join(str(x) for x in mixed) +
                " sectors): the sound is used as the clock so every part gets the right speed")
        else:
            tt = np.array([(f["first"] - M.first_vid) / 150.0 for f in frames])
            cl = np.array([f["clip"] for f in frames])
            for ci in (np.unique(cl) if not (akey and (len(mixed) > 1 or mt.get("period") not in (None, 8))) else []):          # smooth out the +-7 ms jitter caused by the sound sectors in between
                ix = np.nonzero(cl == ci)[0]
                if len(ix) >= 3:
                    sl, ic = np.polyfit(np.arange(len(ix)), tt[ix], 1); tt[ix] = ic + sl * np.arange(len(ix))
        step = float(np.median(np.diff(tt))) if len(tt) > 1 else 1.0 / ref_fps
        total_t = (tt[-1] + (tt[-1] - tt[-2] if len(tt) > 1 else step)) / o["speed"]
        nout = max(1, int(total_t * ref_fps + 1e-6))
        sel = [max(0, int(np.searchsorted(tt, (k / ref_fps) * o["speed"] + 2e-3, "right")) - 1) for k in range(nout)]
        log(f"re-timing by real time: {len(frames)} frames in {M.clips} clip(s) -> {len(sel)} frames at {ref_fps:g} fps" +
            (" (clips with different speeds are each brought to the same speed)" if M.clips > 1 and len(set(round(x) for x in clip_rates(M))) > 1 else ""))
    else:
        sel = list(range(len(frames)))
    # fit to the original's size
    cap = len(sel)
    if not o["allow_larger"]:
        v = 0; c = 0
        while c < len(sel):
            v2 = v + bud(c)
            if re1str.total_sectors(v2) > R.n: break
            v = v2; c += 1
        if c < len(sel): log(f"WARNING: the last {(len(sel) - c) / max(ref_fps, 1):.1f} s did not fit and were cut off (the original has {R.n} sectors). Use 'Speed up to fit' to keep everything.")
        cap = c
    sel = sel[:cap]
    if custom:
        budgets_plan = plan_budgets([frames[j]["size"] for j in sel], bud) if frames else []
    else:
        budgets_plan = plan_budgets([frames[j]["size"] for j in sel], rcyc) if frames else []
    over = 0 if budgets_plan is not None else 1
    if over: reasons.append(f"{over} frame(s) are too big for the original's per-frame space")
    if o["reencode"] == "always": reasons = reasons or ["requested"]
    need = o["reencode"] == "always" or (o["reencode"] == "auto" and reasons)
    if o["reencode"] == "never" and reasons:
        raise FixError("Cannot keep the video untouched: " + "; ".join(reasons) + ". Allow re-encoding to fix this.")
    cont = {"ref": None, "keep": "keep", "riff": True, "raw": False}.get(o["container"], None)
    if need:
        log("method: RE-ENCODE (" + "; ".join(reasons) + ")")
        keep_aud = None
        if is_str and akey and R.main_audio and akey[2] == R.main_audio[2] and o["audio"] == "auto" and not o["audio_offset_ms"] and not o["audio_gain_db"] and not o["audio_limit"] \
                and abs(o["speed"] - 1) < 1e-6 and not (o["retime"] and src_fps and abs(src_fps - ref_fps) / ref_fps > 0.01 and not custom):
            keep_aud = [M.arr[i, 24:24 + 2324].tobytes() for i in M.audio[akey]]
        st = re1str.convert(mod_path, ref_path, out_path, fit=o["fit"], audio_sectors=keep_aud, no_audio=False, allow_larger=o["allow_larger"], pad=o["pad"],
                            riff={"ref": None, "keep": None, "riff": True, "raw": False}[o["container"]],
                            start_lba=None if o["lba"] else M.first_lba, log=log, progress=progress, cancel=cancel,
                            source_fps=src_fps, fps=out_fps if custom else None, auto_fps=False,
                            frame_sel=[fidx[j] for j in sel] if (need_retime and is_str) else None, sel_total=len(M.frames), speed=o["speed"], audio_offset_ms=o["audio_offset_ms"], audio_gain_db=o["audio_gain_db"], audio_limit=o["audio_limit"])
        st["method"] = "re-encode"; st["reasons"] = reasons
        return st
    # ---- lossless repack
    log("method: LOSSLESS REPACK (the video data is copied unchanged)")
    out_frames = []
    for k, j in enumerate(sel):
        f = frames[j]
        out_frames.append(dict(num=k + 1, words=f["words"], qs=f["qs"], data=frame_data(M, f), ver=f["ver"]))
        if cancel(): raise re1str.Cancelled()
    if not out_frames: raise FixError("nothing to write: no frames left")
    budgets = budgets_plan
    vsum = sum(budgets)
    nslots = re1str.audio_slots(vsum)
    progress(0.4, "sound")
    # audio
    ref_fmt = R.main_audio[2] if R.main_audio else None
    if not R.main_audio:
        asec = []
    else:
        recode = o["audio"] == "always" or abs(o["audio_offset_ms"]) > 0 or abs(o["audio_gain_db"]) > 0 or o["audio_limit"] or abs(o["speed"] - 1) > 1e-6 or not akey or \
                 (o["audio"] == "auto" and akey[2] != ref_fmt)
        if recode:
            log("sound: converting to the original's format" if akey else "sound: none in the modded file -> silence")
            pcm = _decode_audio(mod_path, nslots * xaenc.SAMPLES_PER_SECTOR, o["audio_offset_ms"], o["audio_gain_db"], o["audio_limit"], o["speed"]) if akey else \
                  np.zeros((nslots * xaenc.SAMPLES_PER_SECTOR, 2), np.int16)
            asec = xaenc.encode(pcm)
        else:
            log("sound: kept as is (same format)")
            asec = [M.arr[i, 24:24 + 2324].tobytes() for i in M.audio[akey]][:nslots]
    # profile (what the output should look like)
    if not o["flags"]:
        P.vid_sub = M.vid_sub; P.last_sub = M.last_vid_sub
        if M.aud_sub: P.aud_sub = M.aud_sub
    if o["container"] == "keep": P.riff = M.riff
    elif o["container"] in ("riff", "raw"): P.riff = DEFAULT_RIFF if o["container"] == "riff" else None
    start_lba = P.start_lba if o["lba"] else M.first_lba
    progress(0.5, "writing sectors")
    data = mux_streams(P, out_frames, asec, budgets=budgets, start_lba=start_lba)
    body_secs = re1str.total_sectors(vsum)
    if o["pad"] and not o["allow_larger"] and body_secs < R.n:
        from psxcd import build_empty
        extra = b"".join(build_empty(start_lba + k, P.null_sub[0] if P.null_sub else P.vid_sub[0], P.null_sub[1] if P.null_sub else 0) for k in range(body_secs, R.n))
        if P.riff:
            data = bytearray(data) + extra
            struct.pack_into("<I", data, 4, len(data) - 44 + 36); struct.pack_into("<I", data, 40, len(data) - 44); data = bytes(data)
        else:
            data += extra
        log(f"padded with {R.n - body_secs} empty sectors (same size as the original)")
    with open(out_path, "wb") as fh: fh.write(data)
    problems = re1str.verify(out_path, ref_path, check_container=o["container"] == "ref", check_layout=not custom)
    if problems and o["flags"]:
        raise FixError("result did not pass the check: " + "; ".join(problems))
    progress(1.0, "done")
    return dict(method="lossless", reasons=[], frames=len(out_frames), seconds=len(out_frames) / ref_fps, bytes=len(data), sectors=body_secs)


def fix(mod_path, ref_path, out_path, opts=None, log=print, progress=None, cancel=None):
    """Adjust a movie. With container "s2336" (or "ref" when the original is itself stored that way) the result is
    written as headerless 2336-byte sectors, the layout RE1 Rebuilder's BUILDCD / XA_extract use."""
    import strio
    o = dict(DEFAULTS); o.update(opts or {})
    want = o["container"] == "s2336" or (o["container"] == "ref" and strio.ref_is_2336(ref_path))
    if not want:
        return _fix_inner(mod_path, ref_path, out_path, opts, log, progress, cancel)
    o["container"] = "raw"
    tmp = out_path + ".tmp2352"
    try:
        r = _fix_inner(mod_path, ref_path, tmp, o, log, progress, cancel)
        n = strio.to_2336(tmp, out_path)
    finally:
        try: os.remove(tmp)
        except OSError: pass
    log(f"saved as 2336-byte sectors ({n} sectors), the layout RE1 Rebuilder's BUILDCD expects")
    P2, fr2, au2 = demux(out_path)           # read it back: the converted file must give the same frames
    if len(fr2) != r["frames"]:
        raise FixError("result did not pass the check: the 2336-byte file has %d frames, expected %d" % (len(fr2), r["frames"]))
    r["bytes"] = os.path.getsize(out_path)
    return r
