#!/usr/bin/env python3
"""
RE1 STR tool - turn any video into a PlayStation STR movie that matches an ORIGINAL game STR.

    python re1str.py encode  my_video.mp4  --ref ORIGINAL.STR  -o NEW.STR
    python re1str.py info    ORIGINAL.STR
    python re1str.py verify  NEW.STR  --ref ORIGINAL.STR

The reference STR (the file you are replacing) decides: resolution, CD-XA file/channel numbers,
sector flags, audio interleave, per-frame sector budget, frame rate, end-of-stream flag.
Needs: Python 3.8+, numpy, ffmpeg (in PATH).
"""
import argparse, os, re, shutil, struct, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psxcd import build_sector, build_empty, msf_to_lba
from mux import demux, mux, budget_list, split_riff, DEFAULT_RIFF, SEC, CHUNK
from bsenc import FrameCoeffs, encode_frame
import xaenc
import tools

def die(msg):
    print("ERROR: " + msg, file=sys.stderr); sys.exit(1)

class ConvertError(Exception):
    pass

class Cancelled(Exception):
    pass

# --------------------------------------------------------------------------- layout maths
def last_index(vsum):                  # index of the last video sector in the stream
    return vsum - 1 + (vsum - 1) // 7

def total_sectors(vsum):
    return last_index(vsum) + 1

def audio_slots(vsum):
    return (last_index(vsum) + 1) // 8

def info(path):
    P, fr, au = demux(path)
    secs = P.total_sectors
    print(f"file            : {path}")
    print(f"container       : {'RIFF/CDXA (44 byte header)' if P.riff else 'raw 2352-byte sectors'}")
    print(f"sectors         : {secs}  ({secs/150:.2f} s at 2x speed)   first LBA {P.start_lba}")
    print(f"video           : {P.w}x{P.h}, {P.nframes} frames, {P.fps:.2f} fps, BS v{P.ver}")
    print(f"sectors/frame   : cycle {P.cycle}  (avg {sum(P.cycle)/len(P.cycle):.4f})")
    print(f"video sectors   : file={P.vid_sub[0]} channel={P.vid_sub[1]} submode=0x{P.vid_sub[2]:02X} coding=0x{P.vid_sub[3]:02X}")
    print(f"last sector     : submode=0x{P.last_sub[2]:02X}")
    print(f"audio sectors   : {len(au)}  file={P.aud_sub[0]} channel={P.aud_sub[1]} submode=0x{P.aud_sub[2]:02X} coding=0x{P.aud_sub[3]:02X}"
          f"  (one every {P.audio_period} sectors)")
    return P

# --------------------------------------------------------------------------- ffmpeg input
def vfilter(P, fit, fps, src_fps=None, speed=1.0):
    W, H = P.w, P.h
    if fit == "stretch":
        geo = f"scale={W}:{H}:flags=lanczos"
    elif fit == "crop":
        geo = f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=increase,crop={W}:{H}"
    else:
        geo = (f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black")
    if src_fps: pre = f"setpts=N/({src_fps}*{speed}*TB),fps={fps}"
    elif abs(speed - 1) > 1e-6: pre = f"setpts=PTS/{speed},fps={fps}"
    else: pre = f"fps={fps}"
    return f"{pre},{geo},setsar=1,scale=out_range=full:flags=lanczos,format=yuv420p"

def read_video(path, P, fit, fps, max_frames, start, dur, src_fps=None, speed=1.0):
    cmd = ["-v", "error", "-nostdin"]
    if start: cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if dur: cmd += ["-t", str(dur)]
    cmd += ["-an", "-vf", vfilter(P, fit, fps, src_fps, speed), "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    pr = tools.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fs = P.w * P.h * 3 // 2
    try:
        while True:
            buf = pr.stdout.read(fs)
            if len(buf) < fs: break
            a = np.frombuffer(buf, np.uint8)
            yield (a[:P.w * P.h].reshape(P.h, P.w),
                   a[P.w * P.h:P.w * P.h * 5 // 4].reshape(P.h // 2, P.w // 2),
                   a[P.w * P.h * 5 // 4:].reshape(P.h // 2, P.w // 2))
    finally:
        try: pr.kill()
        except Exception: pass
        pr.stdout.close(); pr.wait()

def limit_peak(pcm, target=0.84):
    """Scale the sound down (never up) so its loudest peak stays at `target` of full scale. Returns (pcm, gain_db)."""
    pk = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if pk <= target * 32767: return pcm, 0.0
    g = target * 32767 / pk
    return np.clip(pcm.astype(np.float64) * g, -32768, 32767).round().astype(np.int16), 20 * np.log10(g)

def read_video_sel(path, P, fit, sel, expect_total=None, log=print):
    """Decode EVERY frame of an STR (no re-timing) and yield, for each entry j of `sel`, source frame j (held if repeated)."""
    W, H = P.w, P.h
    if fit == "stretch": geo = f"scale={W}:{H}:flags=lanczos"
    elif fit == "crop": geo = f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=increase,crop={W}:{H}"
    else: geo = f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black"
    vf = f"{geo},setsar=1,scale=out_range=full:flags=lanczos,format=yuv420p"
    cmd = ["-v", "error", "-nostdin", "-i", path, "-an", "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    pr = tools.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fs = W * H * 3 // 2
    def planes(buf):
        a = np.frombuffer(buf, np.uint8)
        return (a[:W * H].reshape(H, W), a[W * H:W * H * 5 // 4].reshape(H // 2, W // 2), a[W * H * 5 // 4:].reshape(H // 2, W // 2))
    cur, cur_idx, total = None, -1, 0
    try:
        for j in sel:
            while cur_idx < j:
                buf = pr.stdout.read(fs)
                if len(buf) < fs: break
                cur = planes(buf); cur_idx += 1; total += 1
            if cur is None: return
            yield cur
        if expect_total and total < expect_total - 1 and sel and sel[-1] >= total:
            log(f"warning: only {total} of {expect_total} frames could be decoded")
    finally:
        try: pr.kill()
        except Exception: pass
        pr.stdout.close(); pr.wait()

def read_audio(path, nsamples, start, dur, offset_ms=0, gain_db=0.0, speed=1.0, limit=False):
    cmd = ["-v", "error", "-nostdin"]
    if start: cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if dur: cmd += ["-t", str(dur)]
    cmd += ["-vn"]
    if abs(speed - 1) > 1e-6: cmd += ["-af", f"atempo={speed}"]
    cmd += ["-ar", "37800", "-ac", "2", "-f", "s16le", "-"]
    r = tools.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pcm = np.frombuffer(r.stdout[:len(r.stdout) // 4 * 4], "<i2").reshape(-1, 2)
    if gain_db:
        pcm = np.clip(pcm.astype(np.float64) * (10 ** (gain_db / 20.0)), -32768, 32767).round().astype(np.int16)
    if limit: pcm, _g = limit_peak(pcm)
    if offset_ms:
        k = int(round(abs(offset_ms) * 37.8))
        pcm = np.vstack([np.zeros((k, 2), np.int16), pcm]) if offset_ms > 0 else pcm[k:]
    out = np.zeros((nsamples, 2), np.int16)
    m = min(nsamples, len(pcm)); out[:m] = pcm[:m]
    return out

# --------------------------------------------------------------------------- verify
def verify(path, ref=None, quiet=False, check_container=True, check_layout=True):
    """Structural checks. Returns list of problems (empty = OK)."""
    bad = []
    raw = open(path, "rb").read()
    riff, body = split_riff(raw)
    if len(body) % SEC: bad.append(f"size is not a multiple of {SEC}")
    n = len(body) // SEC
    P, fr, au = demux(path)
    # every sector regenerates byte-exactly (EDC/ECC/sync/MSF) and LBAs are continuous
    for i in range(n):
        s = body[i * SEC:(i + 1) * SEC]
        lba = msf_to_lba(*s[12:15])
        if lba != P.start_lba + i: bad.append(f"sector {i}: LBA not continuous"); break
        sub = tuple(s[16:20]); form2 = bool(sub[2] & 0x20) or bool(sub[2] & 0x04)
        pl = s[24:24 + (2324 if form2 else 2048)]
        if build_sector(lba, sub, pl, form2=form2) != s: bad.append(f"sector {i}: EDC/ECC/header mismatch"); break
    nums = [f["num"] for f in fr]
    if nums != list(range(1, len(fr) + 1)): bad.append("frame numbers are not 1..N")
    for f in fr:
        if len(f["chunks"]) != f["cnt"]: bad.append(f"frame {f['num']}: missing chunks"); break
        if f["words"] <= 0: bad.append(f"frame {f['num']}: bad MDEC word count"); break
    if ref:
        R, _, _ = demux(ref)
        for name in ("w", "h", "magic", "m38", "ver", "vid_sub", "aud_sub", "null_sub", "last_sub", "audio_period", "audio_slot") + (("cycle",) if check_layout else ()):
            if name == "null_sub" and (P.null_sub is None or R.null_sub is None): continue
            if getattr(P, name) != getattr(R, name): bad.append(f"{name} differs from reference: {getattr(P, name)} vs {getattr(R, name)}")
        if check_container and (P.riff is None) != (R.riff is None): bad.append("container (RIFF/raw) differs from reference")
        if P.start_lba != R.start_lba: print("note: first LBA differs from reference (fine if your injector rewrites it)") if not quiet else None
    return bad

# --------------------------------------------------------------------------- encode
def probe_fps(path):
    r = tools.run(["-hide_banner", "-i", path], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    t = r.stderr.decode("utf8", "ignore")
    m = re.search(r"([\d.]+) fps", t) or re.search(r"([\d.]+) tbr", t)
    return float(m.group(1)) if m else None

def exact_bud(nominal, period=8):
    """Sectors per frame (i -> int) so that the movie plays at exactly `nominal` fps at 2x speed (150 sectors/s)."""
    vper = 150.0 / nominal * (period - 1) / period
    return lambda i: max(1, int(round((i + 1) * vper)) - int(round(i * vper)))

def convert(input, ref, output, *a, **k):
    """See _convert. STR inputs in an unusual layout (no sync headers, junk bytes) are normalised first."""
    import strio
    inp, tmp = strio.clean_copy(input) if strio.looks_like_str(input) else (input, False)
    try:
        return _convert(inp, ref, output, *a, **k)
    finally:
        if tmp:
            try: os.remove(inp)
            except OSError: pass

def _convert(input, ref, output, fit="pad", fps=None, start=None, duration=None, max_frames=None,
            no_audio=False, allow_larger=False, pad=True, riff=None, start_lba=None,
            log=print, progress=None, cancel=None, source_fps=None, audio_offset_ms=0, audio_gain_db=0.0, speed=1.0, auto_fps=True, audio_sectors=None, audio_limit=False, frame_sel=None, sel_total=None):
    """Video -> STR matching `ref`. progress(fraction, text); cancel() -> True to abort.
    Raises ConvertError / Cancelled. Returns a stats dict."""
    tools.need_ffmpeg()
    cancel = cancel or (lambda: False)
    progress = progress or (lambda f, t="": None)
    if not os.path.exists(input): raise ConvertError(f"input not found: {input}")
    P, rframes, raudio = demux(ref)
    ref_secs = P.total_sectors
    if fps is None and auto_fps:     # default: the video's own frame rate
        try:
            vf = probe_fps(input)
            if vf: fps = max(1.0, min(30.0, vf)); fps = next((float(c) for c in (10, 12, 15, 20, 24, 25, 30, 23.976, 29.97) if abs(fps - c) / c < 0.004), round(fps, 3))
        except Exception: pass
    nominal = fps
    fps = fps or P.fps
    custom = bool(nominal) and abs(nominal / P.fps - 1) > 0.002        # the user asked for a different frame rate than the original's layout gives
    if custom:
        bud = exact_bud(nominal, P.audio_period)
        vper = 150.0 / nominal * (P.audio_period - 1) / P.audio_period
        log(f"custom frame rate {nominal:g} fps: {vper:.2f} video sectors per frame (the original's layout gives {P.fps:.2f} fps)")
    else:
        bud = lambda i: P.cycle[i % len(P.cycle)]
    if riff is not None: P.riff = DEFAULT_RIFF if riff else None
    if start_lba is not None: P.start_lba = start_lba
    limit = None if allow_larger else ref_secs
    cyc = len(P.cycle)
    cap = 10 ** 9
    if limit:
        cap = 0; v = 0
        while True:
            v2 = v + bud(cap)
            if total_sectors(v2) > limit: break
            v = v2; cap += 1
    if max_frames: cap = min(cap, max_frames)
    # expected frames (for the progress bar)
    expect = cap
    try:
        r = tools.run(["-hide_banner", "-i", input], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
        m = re.search(rb"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
        if m:
            d = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) - (start or 0)
            if duration: d = min(d, duration)
            expect = min(cap, max(1, int(d * fps)))
    except Exception:
        pass
    log(f"output: {P.w}x{P.h} @ {fps:.2f} fps, {ref_secs} sectors ({ref_secs/150:.1f} s)"
        + (f" -> room for {cap} frames ({cap/fps:.1f} s)" if cap < 10 ** 9 else ""))
    frames, t0, qs_all = [], time.time(), []
    if frame_sel is not None:
        cap = min(cap, len(frame_sel)); expect = cap
        src_iter = read_video_sel(input, P, fit, frame_sel[:cap], sel_total, log)
    else:
        src_iter = read_video(input, P, fit, fps, cap, start, duration, source_fps, speed)
    for i, (Y, Cb, Cr) in enumerate(src_iter):
        if cancel(): raise Cancelled()
        cnt = bud(i)
        try:
            e = encode_frame(Y, Cb, Cr, cnt * CHUNK)
        except ValueError:
            raise ConvertError(f"frame {i+1} does not fit its {cnt}-sector budget even at the coarsest quantiser")
        frames.append(dict(num=i + 1, words=e["words"], qs=e["qs"], data=e["data"])); qs_all.append(e["qs"])
        if (i + 1) % 5 == 0:
            progress(0.9 * min(1.0, (i + 1) / max(1, expect)), f"encoding video  {i+1}/{expect} frames  ({(i+1)/(time.time()-t0):.1f} fps)")
    if not frames: raise ConvertError("no video frames could be read from the input")
    log(f"video: {len(frames)} frames, quantiser scale avg {np.mean(qs_all):.1f} (lower = sharper), max {max(qs_all)}")
    budgets = [bud(i) for i in range(len(frames))]
    vsum = sum(budgets)
    nslots = audio_slots(vsum)
    progress(0.9, "encoding audio")
    if audio_sectors is not None:          # the modded file's own sound, already in the right format: reuse it untouched
        pcm = None
    elif no_audio:
        pcm = np.zeros((nslots * xaenc.SAMPLES_PER_SECTOR, 2), np.int16)
    else:
        pcm = read_audio(input, nslots * xaenc.SAMPLES_PER_SECTOR, start, duration, audio_offset_ms, audio_gain_db, speed, audio_limit)
    if cancel(): raise Cancelled()
    if audio_sectors is not None:
        asec = list(audio_sectors)[:nslots]
        log(f"audio: {len(asec)} XA sectors (the modded file's sound is kept as it is: no second compression)")
    else:
        asec = xaenc.encode(pcm)
        log(f"audio: {len(asec)} XA sectors")
    progress(0.96, "writing STR")
    out = mux(P, frames, asec, budgets=budgets)
    body_secs = total_sectors(vsum)
    if pad and limit and body_secs < limit:
        extra = b"".join(build_empty(P.start_lba + k, (P.null_sub or P.vid_sub)[0], P.null_sub[1] if P.null_sub else 0) for k in range(body_secs, limit))
        if P.riff:
            out = bytearray(out); out += extra
            struct.pack_into("<I", out, 4, len(out) - 44 + 36); struct.pack_into("<I", out, 40, len(out) - 44)
            out = bytes(out)
        else:
            out += extra
        log(f"padded with {limit - body_secs} empty sectors (file is exactly as large as the reference)")
    with open(output, "wb") as fh: fh.write(out)
    log(f"wrote {output}: {len(out)} bytes, {body_secs} sectors, {len(frames)} frames, {len(frames)/fps:.1f} s")
    problems = verify(output, ref, check_container=riff is None, check_layout=not custom)
    if problems:
        raise ConvertError("verify failed: " + "; ".join(problems))
    log("verify: OK (all sectors valid, layout matches the reference)")
    progress(1.0, "done")
    return dict(frames=len(frames), seconds=len(frames) / fps, bytes=len(out), sectors=body_secs, avg_qscale=float(np.mean(qs_all)))

def cmd_encode(a):
    try:
        convert(a.input, a.ref, a.output, fit=a.fit, fps=a.fps, start=a.start, duration=a.duration, max_frames=a.max_frames,
                no_audio=a.no_audio, allow_larger=a.allow_larger, pad=a.pad, riff=a.riff, start_lba=a.start_lba,
                progress=lambda f, t="": print(f"\r  {t}" + " " * 10, end="", flush=True) if t and f < 1 else None)
        print()
    except (ConvertError, Cancelled) as e:
        print(); die(str(e))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode", help="convert a video into a game-ready STR")
    e.add_argument("input"); e.add_argument("--ref", required=True, help="the ORIGINAL .STR you are replacing")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--fit", choices=["pad", "crop", "stretch"], default="pad", help="how to fit the video to the game's resolution (default pad = black bars)")
    e.add_argument("--fps", type=float, help="override frame rate (default: the reference's)")
    e.add_argument("--start", type=float, help="start offset in seconds"); e.add_argument("--duration", type=float, help="seconds to encode")
    e.add_argument("--max-frames", type=int)
    e.add_argument("--no-audio", action="store_true")
    e.add_argument("--allow-larger", action="store_true", help="allow output bigger than the reference (needs an ISO rebuild tool)")
    e.add_argument("--no-pad", dest="pad", action="store_false", help="don't pad the file to the reference size")
    e.add_argument("--riff", dest="riff", action="store_true", default=None, help="force RIFF/CDXA container")
    e.add_argument("--raw", dest="riff", action="store_false", help="force raw 2352-byte sectors (no RIFF header)")
    e.add_argument("--start-lba", type=int, help="LBA of the first sector (default: the reference's)")
    i = sub.add_parser("info", help="describe an STR"); i.add_argument("file")
    v = sub.add_parser("verify", help="check an STR"); v.add_argument("file"); v.add_argument("--ref")
    a = ap.parse_args()
    if a.cmd == "info": info(a.file)
    elif a.cmd == "verify":
        pr = verify(a.file, a.ref); print("OK" if not pr else "\n".join(pr)); sys.exit(1 if pr else 0)
    else: cmd_encode(a)

if __name__ == "__main__":
    main()
