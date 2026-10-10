"""Locate ffmpeg and helpers for running it quietly (no console window on Windows)."""
import os, shutil, subprocess, sys

_cache = {}

def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def ffmpeg_path():
    if "ff" in _cache: return _cache["ff"]
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    cands = [os.environ.get("RE1STR_FFMPEG"), os.path.join(app_dir(), exe), os.path.join(app_dir(), "ffmpeg", exe),
             getattr(sys, "_MEIPASS", None) and os.path.join(sys._MEIPASS, exe), shutil.which("ffmpeg")]
    p = next((c for c in cands if c and os.path.isfile(c)), None)
    if not p:
        try:
            import imageio_ffmpeg                       # optional: pip install imageio-ffmpeg (bundles a static ffmpeg)
            p = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            p = None
    _cache["ff"] = p
    return p

def need_ffmpeg():
    p = ffmpeg_path()
    if not p:
        raise RuntimeError("ffmpeg not found. Put ffmpeg.exe next to the program, add it to PATH, "
                           "or run:  pip install imageio-ffmpeg")
    return p

def popen_kwargs():
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000                # CREATE_NO_WINDOW
    return kw

def _std(kw):
    # windowed (no console) programs have no valid stdin/stdout/stderr handles: never inherit them
    for k in ("stdin", "stdout", "stderr"):
        kw.setdefault(k, subprocess.DEVNULL)
    return kw

def run(args, **kw):
    return subprocess.run([need_ffmpeg()] + args, **popen_kwargs(), **_std(kw))

def popen(args, **kw):
    return subprocess.Popen([need_ffmpeg()] + args, **popen_kwargs(), **_std(kw))

def passthrough_flags():
    """ffmpeg flag that emits every decoded frame once, whatever the timestamps say."""
    if "pt" not in _cache:
        r = run(["-hide_banner", "-h", "full"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _cache["pt"] = ["-fps_mode", "passthrough"] if b"fps_mode" in r.stdout else ["-vsync", "0"]
    return _cache["pt"]

def has_encoder(name):
    key = "enc_" + name
    if key not in _cache:
        r = run(["-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _cache[key] = (" " + name + " ") in r.stdout.decode("utf8", "ignore")
    return _cache[key]
