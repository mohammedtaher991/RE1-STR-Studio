"""Backend for the RE1 STR Studio: STR inspection and STR -> MP4/AVI rendering (no external tools besides ffmpeg)."""
import os, re, struct, subprocess, tempfile, threading, time
import numpy as np
import tools, strio
from psxcd import msf_to_lba

SEC = 2352
class Cancelled(Exception): pass

# ----------------------------------------------------------------------------- inspection
def probe_str(path):
    """Parse a PlayStation STR (RIFF/CDXA or raw 2352-byte sectors) and describe it."""
    raw = open(path, "rb").read()
    riff_h, body, layout = strio.split(raw)
    riff = riff_h is not None
    n = len(body) // SEC
    if n == 0: raise ValueError("file is too small to be an STR")
    vid = {}; aud = {}; nulls = 0; other = 0
    qs = []; frames = {}
    first_lba = msf_to_lba(*body[12:15])
    if body[:12] != strio.SYNC: raise ValueError("not a CD-XA sector file (no sync pattern)")
    vsecs = asecs = 0; vbytes = 0
    for i in range(n):
        s = body[i * SEC:(i + 1) * SEC]
        f, ch, sm, co = s[16:20]
        if sm & 0x04:                                    # audio
            aud.setdefault((f, ch, co), 0); aud[(f, ch, co)] += 1; asecs += 1
        elif s[24:28] == b"\x60\x01\x01\x80":          # video (recognised by its header, whatever the flags say)
            magic, idx, cnt, fn, size, w, h, words, m38, q, ver, _ = struct.unpack("<IHHIIHHHHHHI", s[24:56])
            vsecs += 1
            fr = frames.setdefault(fn, dict(cnt=cnt, size=size, q=q, ver=ver, w=w, h=h, first=i))
            vid[(f, ch)] = vid.get((f, ch), 0) + 1
        else:
            nulls += 1
    if not frames: raise ValueError("no video frames found")
    fnums = sorted(frames)
    first = frames[fnums[0]]
    nframes = len(fnums)
    avg_v = vsecs / nframes
    period = None
    info = dict(path=path, name=os.path.basename(path), bytes=len(raw), container=strio.LAYOUT_TEXT[layout],
                sectors=n, first_lba=first_lba, null_sectors=nulls)
    # audio
    if aud:
        (af, ac, co), cnt_a = max(aud.items(), key=lambda kv: kv[1])
        stereo = bool(co & 1); rate = 18900 if co & 4 else 37800; bits = 8 if co & 0x10 else 4
        spf = {(4, True): 2016, (4, False): 4032, (8, True): 1008, (8, False): 2016}[(bits, stereo)]   # samples/channel per sector
        sec_per_s_audio = rate / spf
        # sector spacing of audio inside the stream
        pos = [i for i in range(min(n, 400)) if (body[i * SEC + 18] & 0x04)]
        period = int(round(np.median(np.diff(pos)))) if len(pos) > 2 else 8
        sectors_per_second = period * sec_per_s_audio
        speed = sectors_per_second / 75.0
        audio = dict(present=True, rate=rate, bits=bits, channels=2 if stereo else 1, sectors=asecs, file=af, channel=ac,
                     coding=co, codec="XA-ADPCM", duration=cnt_a * spf / rate, channels_in_stream=len(aud), spacing=period)
    else:
        sectors_per_second = 150.0; speed = 2.0; audio = dict(present=False)
    sectors_per_frame = (n - nulls) / nframes if not aud else (vsecs + asecs) / nframes
    fps = sectors_per_second / sectors_per_frame
    # snap to common rates (the last frame/padding can skew the average a little)
    for c in (10, 12, 15, 20, 24, 25, 29.97, 30):
        if abs(fps - c) / c < 0.02: fps = float(c); break
    dur = nframes / fps
    qavg = float(np.mean([frames[k]["q"] for k in fnums]))
    vf, vc = max(vid.items(), key=lambda kv: kv[1])[0]
    sub = body[16:24]
    info.update(
        video=dict(width=first["w"], height=first["h"], frames=nframes, fps=fps, duration=dur, codec="PSX MDEC",
                   bs_version=first["ver"], avg_qscale=qavg, sectors=vsecs, sectors_per_frame=avg_v, file=vf, channel=vc,
                   bitrate_kbps=(sum(frames[k]["size"] for k in fnums) * 8 / dur / 1000)),
        audio=audio,
        speed=speed, sectors_per_second=sectors_per_second, duration=dur,
        type=("CD-XA interleaved movie (Mode 2: Form 1 video + Form 2 XA-ADPCM audio), %.0fx speed" % speed) if aud
             else ("CD-XA video-only movie, %.0fx speed" % speed),
        frame_gaps=len(fnums) != fnums[-1] - fnums[0] + 1)
    return info

def looks_like_str(path):
    return strio.looks_like_str(path)

VIDEO_EXT = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp", ".ogv", ".gif"}

def probe_video(path):
    """Describe an ordinary video file by reading ffmpeg's banner (no ffprobe needed)."""
    r = tools.run(["-hide_banner", "-i", path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t = r.stderr.decode("utf8", "ignore")
    if "Invalid data" in t and "Stream #" not in t: raise ValueError("ffmpeg cannot read this file")
    cont = re.search(r"Input #0, ([^,]+(?:,[^,]+)*?), from", t)
    dm = re.search(r"Duration: (\d+):(\d+):([\d.]+)", t)
    dur = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3)) if dm else None
    vl = next((l for l in t.splitlines() if "Video:" in l and "Stream #" in l), None)
    al = next((l for l in t.splitlines() if "Audio:" in l and "Stream #" in l), None)
    if not vl: raise ValueError("no video stream found in this file")
    size = re.search(r"\b(\d{2,5})x(\d{2,5})\b", vl.split("Video:", 1)[1])
    fpsm = re.search(r"([\d.]+) fps", vl) or re.search(r"([\d.]+) tbr", vl)
    fps = float(fpsm.group(1)) if fpsm else 30.0
    pix = re.search(r"Video: \w+[^,]*, (\w+)", vl)
    vcodec = re.search(r"Video: (\w+)(?: \(([^)]*)\))?", vl)
    vbr = re.search(r"(\d+) kb/s", vl)
    info = dict(kind="video", path=path, name=os.path.basename(path), bytes=os.path.getsize(path),
                container=(cont.group(1) if cont else "?"), duration=dur,
                video=dict(width=int(size.group(1)), height=int(size.group(2)), fps=fps, codec=vcodec.group(1) if vcodec else "?",
                           profile=vcodec.group(2) if vcodec and vcodec.group(2) else "", pix_fmt=pix.group(1) if pix else "?",
                           frames=int(dur * fps) if dur else None, bitrate_kbps=int(vbr.group(1)) if vbr else None))
    if al:
        ac = re.search(r"Audio: (\w+)[^,]*, (\d+) Hz, ([^,]+)", al)
        abr = re.search(r"(\d+) kb/s", al)
        info["audio"] = dict(present=True, codec=ac.group(1) if ac else "?", rate=int(ac.group(2)) if ac else None,
                             channels_text=ac.group(3).strip().replace("2 channels", "stereo").replace("1 channels", "mono") if ac else "?", bitrate_kbps=int(abr.group(1)) if abr else None)
    else:
        info["audio"] = dict(present=False)
    return info

def probe_any(path):
    if looks_like_str(path):
        i = probe_str(path); i["kind"] = "str"; return i
    return probe_video(path)

def fmt_info(i):
    """Flat (label, value) rows for display."""
    if i.get("kind") == "video": return fmt_info_video(i)
    v, a = i["video"], i["audio"]
    def mb(b): return f"{b/1048576:.2f} MB ({b:,} bytes)"
    rows = [("File", i["name"]), ("STR type", i["type"]), ("Container", i["container"]), ("File size", mb(i["bytes"])),
            ("Sectors", f"{i['sectors']:,}  (first LBA {i['first_lba']}, {i['null_sectors']} empty)"),
            ("Duration", f"{i['duration']:.2f} s"), ("", ""),
            ("Video size", f"{v['width']} x {v['height']}"), ("Frame rate", f"{v['fps']:g} fps"), ("Frames", f"{v['frames']:,}"),
            ("Video codec", f"{v['codec']} BS v{v['bs_version']}  (avg quantiser {v['avg_qscale']:.1f})"),
            ("Video bitrate", f"{v['bitrate_kbps']:.0f} kbit/s"), ("Sectors / frame", f"{v['sectors_per_frame']:.2f}"),
            ("CD-XA video file / channel", f"{v['file']} / {v['channel']}"), ("", "")]
    if a["present"]:
        rows += [("Audio", f"{a['codec']}  {a['bits']}-bit"), ("Sample rate", f"{a['rate']:,} Hz"),
                 ("Audio channels", f"{a['channels']} ({'stereo' if a['channels']==2 else 'mono'})"),
                 ("Audio sectors", f"{a['sectors']:,}  (one every {a['spacing']} sectors)"),
                 ("CD-XA audio file / channel", f"{a['file']} / {a['channel']}"), ("Audio duration", f"{a['duration']:.2f} s")]
    else:
        rows += [("Audio", "none")]
    return rows

def fmt_info_video(i):
    v, a = i["video"], i["audio"]
    def mb(b): return f"{b/1048576:.2f} MB ({b:,} bytes)"
    ext = os.path.splitext(i["name"])[1].upper().lstrip(".")
    rows = [("File", i["name"]), ("Type", f"Video file ({ext})" if ext else "Video file"), ("Container", i["container"]), ("File size", mb(i["bytes"])),
            ("Duration", f"{i['duration']:.2f} s" if i["duration"] else "unknown"), ("", ""),
            ("Video size", f"{v['width']} x {v['height']}"), ("Frame rate", f"{v['fps']:g} fps"),
            ("Frames", f"about {v['frames']:,}" if v["frames"] else "unknown"),
            ("Video codec", v["codec"] + (f" ({v['profile']})" if v["profile"] else "")), ("Pixel format", v["pix_fmt"])]
    if v["bitrate_kbps"]: rows.append(("Video bitrate", f"{v['bitrate_kbps']} kbit/s"))
    rows.append(("", ""))
    if a["present"]:
        rows += [("Audio", a["codec"]), ("Sample rate", f"{a['rate']:,} Hz" if a["rate"] else "?"), ("Audio channels", a["channels_text"])]
        if a["bitrate_kbps"]: rows.append(("Audio bitrate", f"{a['bitrate_kbps']} kbit/s"))
    else:
        rows.append(("Audio", "none"))
    return rows

def browser_direct_ok(i, fmt):
    """Can the browser play this video file as-is (no conversion)?"""
    if i.get("kind") != "video": return False
    v, a = i["video"], i["audio"]
    ext = os.path.splitext(i["path"])[1].lower()
    if v["pix_fmt"] not in ("yuv420p", "yuvj420p"): return False
    if fmt == "mp4":
        return ext in (".mp4", ".m4v", ".mov") and v["codec"] == "h264" and (not a["present"] or a["codec"] in ("aac", "mp3"))
    return ext == ".webm" and v["codec"] in ("vp8", "vp9", "av1") and (not a["present"] or a["codec"] in ("opus", "vorbis"))

def render_preview_video(src, out, fmt, info, progress=None, cancel=None, max_w=1280):
    """Convert any video to something every browser can play (H.264/AAC mp4 or VP9/Opus webm)."""
    progress = progress or (lambda f, t="": None)
    dur = info.get("duration") or 0
    has_a = info["audio"]["present"]
    a = ["-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0"] + (["-map", "0:a:0"] if has_a else [])
    a += ["-vf", f"scale='min({max_w},iw)':-2,format=yuv420p"]
    if fmt == "webm":
        a += (["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-crf", "28", "-b:v", "0"]
              if tools.has_encoder("libvpx-vp9") else ["-c:v", "libvpx", "-crf", "12", "-b:v", "3M"])
        if has_a: a += ["-c:a", "libopus", "-b:a", "128k", "-ac", "2"]
    else:
        a += (["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"] if tools.has_encoder("libx264") else ["-c:v", "mpeg4", "-q:v", "3"])
        if has_a: a += ["-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000"]
        a += ["-movflags", "+faststart"]
    a += ["-progress", "pipe:1", "-nostats", out]
    errf = tempfile.TemporaryFile()
    p = tools.popen(a, stdout=subprocess.PIPE, stderr=errf)
    try:
        for line in iter(p.stdout.readline, b""):
            _check(cancel)
            m = re.match(rb"out_time_(?:us|ms)=(\d+)", line)
            if m and dur:
                progress(min(0.99, int(m.group(1)) / 1e6 / dur), "converting for preview")
        p.wait()
    except BaseException:
        try: p.kill()
        except Exception: pass
        try: os.remove(out)
        except OSError: pass
        raise
    if p.returncode != 0:
        errf.seek(0); msg = errf.read().decode("utf8", "ignore")[-400:]
        try: os.remove(out)
        except OSError: pass
        raise RuntimeError("ffmpeg could not convert this video: " + msg)
    progress(1.0, "done")

# ----------------------------------------------------------------------------- rendering
def _check(cancel):
    if cancel and cancel(): raise Cancelled()

def render(src, out, *a, **k):
    src2, tmp = strio.clean_copy(src) if strio.looks_like_str(src) else (src, False)
    try: return _render(src2, out, *a, **k)
    finally:
        if tmp:
            try: os.remove(src2)
            except OSError: pass

def _render(src, out, fmt="mp4", scale=1, crf=18, audio=True, progress=None, cancel=None, info=None, faststart=True):
    """STR -> .mp4 (H.264+AAC) or .avi (MPEG-4+PCM) using the STR's own frame rate. Returns out path."""
    info = info or probe_str(src)
    v = info["video"]; W, H, fps, nfr = v["width"], v["height"], v["fps"], v["frames"]
    progress = progress or (lambda f, t="": None)
    tools.need_ffmpeg()
    tmp = out + ".audio.wav"
    has_audio = audio and info["audio"]["present"]
    try:
        if has_audio:
            progress(0.0, "extracting audio")
            tools.run(["-v", "error", "-nostdin", "-y", "-i", src, "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", "-f", "wav", tmp],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(tmp) or os.path.getsize(tmp) < 100: has_audio = False
        _check(cancel)
        dec = tools.popen(["-v", "error", "-nostdin", "-i", src, "-an"] + tools.passthrough_flags() +
                          ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        a = ["-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", f"{fps:.6f}", "-i", "pipe:0"]
        if has_audio: a += ["-i", tmp]
        vf = []
        if scale and scale != 1: vf.append(f"scale={W*scale}:{H*scale}:flags=lanczos")
        a += ["-map", "0:v"] + (["-map", "1:a"] if has_audio else [])
        if vf: a += ["-vf", ",".join(vf)]
        if fmt == "mp4":
            enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)] if tools.has_encoder("libx264") else ["-c:v", "mpeg4", "-q:v", "2"]
            a += enc + ["-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-b:a", "192k"] if has_audio else [])
            if faststart: a += ["-movflags", "+faststart"]
        elif fmt == "webm":      # in-app preview: plays in any browser engine without H.264
            if tools.has_encoder("libvpx-vp9"):
                a += ["-c:v", "libvpx-vp9", "-crf", "20", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "6", "-row-mt", "1"]
            elif tools.has_encoder("libvpx"):
                a += ["-c:v", "libvpx", "-crf", "6", "-b:v", "8M", "-deadline", "realtime", "-cpu-used", "5"]
            else:
                raise RuntimeError("this ffmpeg build has no VP8/VP9 encoder (needed for the preview)")
            a += ["-pix_fmt", "yuv420p"] + (["-c:a", "libopus", "-b:a", "160k"] if has_audio else [])
        elif fmt == "webm":      # preview fallback for browsers without H.264
            vc = ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-crf", "26", "-b:v", "0"] \
                 if tools.has_encoder("libvpx-vp9") else ["-c:v", "libvpx", "-crf", "10", "-b:v", "4M"]
            a += vc + ["-pix_fmt", "yuv420p"] + (["-c:a", "libopus", "-b:a", "128k"] if has_audio else [])
        elif fmt == "avi":
            a += ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", "-vtag", "xvid"] + (["-c:a", "pcm_s16le"] if has_audio else [])
        else:
            raise ValueError("unknown format " + fmt)
        if has_audio: a += ["-af", "apad", "-t", f"{nfr / fps:.6f}"]
        a += [out]
        enc_p = tools.popen(a, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        fs = W * H * 3; n = 0
        try:
            while True:
                _check(cancel)
                buf = dec.stdout.read(fs)
                if len(buf) < fs: break
                enc_p.stdin.write(buf); n += 1
                if n % 4 == 0: progress(min(0.99, n / max(1, nfr)), f"rendering {fmt.upper()}  {n}/{nfr} frames")
            enc_p.stdin.close()
            enc_p.wait()
        except BaseException:
            for p in (dec, enc_p):
                try: p.kill()
                except Exception: pass
            try: os.remove(out)
            except OSError: pass
            raise
        finally:
            try: dec.stdout.close(); dec.wait()
            except Exception: pass
        if enc_p.returncode != 0:
            err = enc_p.stderr.read().decode("utf8", "ignore")[-400:]
            raise RuntimeError("ffmpeg failed: " + err)
        progress(1.0, "done")
        return out
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
