"""Local web server behind RE1 STR Studio. Binds to 127.0.0.1 only; every request needs the session token."""
import hashlib, json, os, re, secrets, shutil, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
import tools, engine

def data_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "RE1STRStudio"); os.makedirs(d, exist_ok=True); return d

DEFAULTS = dict(jump=5, jump_options=[5, 10, 15, 20, 25, 30], smooth=True, preview_scale=1, loop=False, volume=0.8,
                autohide=2.5, export_scale=2, export_crf=18,
                theme="dark", c_bg="", c_text="", c_accent="")

class App:
    def __init__(self, data=None):
        self.data = data or data_dir()
        self.cache = os.path.join(self.data, "cache"); os.makedirs(self.cache, exist_ok=True)
        shutil.rmtree(os.path.join(self.cache, "dropped"), ignore_errors=True)  # copies of earlier dropped files
        self.token = secrets.token_hex(16)
        self.jobs = {}; self.lock = threading.Lock(); self.direct = {}
        self.last_ping = None; self.window = ""; self.window_note = ""
        self.sfile = os.path.join(self.data, "settings.json")
        self.settings = dict(DEFAULTS)
        try: self.settings.update(json.load(open(self.sfile)))
        except Exception: pass
        self.port = None

    def save_settings(self, upd):
        for k, v in upd.items():
            if k in DEFAULTS: self.settings[k] = v
        if self.settings["jump"] not in DEFAULTS["jump_options"]: self.settings["jump"] = 5
        try: json.dump(self.settings, open(self.sfile, "w"), indent=1)
        except OSError: pass
        return self.settings

    # ---- jobs
    def new_job(self, fn, kind):
        jid = secrets.token_hex(6)
        job = dict(id=jid, kind=kind, status="running", progress=0.0, text="starting", error=None, result=None, cancel=False, t=time.time(), log=[])
        self.jobs[jid] = job
        def prog(f, t=""):
            job["progress"] = f; job["text"] = t or job["text"]
        def run():
            try:
                job["result"] = fn(prog, lambda: job["cancel"], lambda m: job["log"].append(str(m)))
                job["status"] = "done"; job["progress"] = 1.0
            except (engine.Cancelled,) as e:
                job["status"] = "cancelled"
            except Exception as e:
                if type(e).__name__ == "Cancelled": job["status"] = "cancelled"
                else:
                    job["status"] = "error"; job["error"] = str(e) or type(e).__name__
                    job["log"].append("details: " + " | ".join(traceback.format_exc().strip().splitlines()[-4:]))
                    if os.environ.get("RE1STR_DEBUG"): traceback.print_exc()
        threading.Thread(target=run, daemon=True).start()
        return job

    # ---- preview
    def open_preview(self, path, fmt, scale):
        if not os.path.isfile(path): raise FileNotFoundError(path)
        st = os.stat(path)
        key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{fmt}|{scale}".encode()).hexdigest()[:20]
        out = os.path.join(self.cache, f"{key}.{fmt}")
        def work(prog, cancel, log):
            prog(0.0, "reading file")
            info = engine.probe_any(path)
            if info["kind"] == "video":
                v = info["video"]
                if engine.browser_direct_ok(info, fmt):
                    self.direct[key] = path
                    media = f"/direct/{key}{os.path.splitext(path)[1].lower()}"
                else:
                    if not (os.path.exists(out) and os.path.getsize(out) > 1000):
                        tmp = out + ".part." + fmt
                        engine.render_preview_video(path, tmp, fmt, info, progress=prog, cancel=cancel)
                        os.replace(tmp, out)
                    media = f"/media/{key}.{fmt}"
                fps = v["fps"]; dur = info["duration"]
            else:
                if not (os.path.exists(out) and os.path.getsize(out) > 1000):
                    tmp = out + ".part." + fmt
                    engine.render(path, tmp, fmt, scale=scale, crf=20, progress=prog, cancel=cancel, info=info, faststart=True)
                    os.replace(tmp, out)
                media = f"/media/{key}.{fmt}"; fps = info["video"]["fps"]; dur = info["duration"]
            return dict(info=info, kind=info["kind"], rows=engine.fmt_info(info), media=media, fps=fps,
                        duration=dur, name=os.path.basename(path), has_audio=info["audio"]["present"])
        return self.new_job(work, "preview")

    # ---- exports
    def check_paths(self, inputs, output):
        out = os.path.normcase(os.path.realpath(output))
        for p in inputs:
            if p and os.path.normcase(os.path.realpath(p)) == out:
                raise ValueError("The output file must be different from the input / original file (the original is never overwritten).")
        d = os.path.dirname(os.path.abspath(output))
        if not os.path.isdir(d): raise ValueError(f"Folder does not exist: {d}")
        if not os.access(d, os.W_OK): raise ValueError(f"Cannot write to folder: {d}")

    def export(self, b):
        import re1str
        kind = b.get("kind")
        def num(k, typ=float):
            v = b.get(k)
            if v in (None, "", False): return None
            try: return typ(v)
            except Exception: raise ValueError(f"'{k}' must be a number")
        out = (b.get("output") or "").strip()
        if not out: raise ValueError("Choose where to save the output file.")
        if kind == "str":
            src, ref = (b.get("input") or "").strip(), (b.get("ref") or "").strip()
            if not os.path.isfile(src): raise ValueError("Choose the input video.")
            if not os.path.isfile(ref): raise ValueError("Choose the original STR to match (the file you are replacing).")
            self.check_paths([src, ref], out)
            container = b.get("container", "same")
            kw = dict(fit=b.get("fit", "pad"), fps=num("fps"), start=num("start"), duration=num("duration"), no_audio=not b.get("audio", True), audio_limit=bool(b.get("audio_limit")),
                      allow_larger=bool(b.get("allow_larger")), pad=bool(b.get("pad", True)),
                      riff={"same": None, "riff": True, "raw": False}.get(container), start_lba=num("lba", int))
            def work(prog, cancel, log):
                import re1str as r
                tmp = out + ".part"
                try:
                    st = r.convert(src, ref, tmp, log=lambda m: log(str(m).replace(tmp, out)), progress=prog, cancel=cancel, **kw)
                    os.replace(tmp, out)
                finally:
                    if os.path.exists(tmp):
                        try: os.remove(tmp)
                        except OSError: pass
                st["output"] = out
                return st
            return self.new_job(work, "export-str")
        if kind in ("mp4", "avi"):
            src = (b.get("input") or "").strip()
            if not os.path.isfile(src): raise ValueError("Choose the STR file to convert.")
            self.check_paths([src], out)
            scale = int(num("scale") or 1); crf = int(num("crf") or 18); audio = bool(b.get("audio", True))
            def work(prog, cancel, log):
                tmp = out + ".part." + kind
                try:
                    log(f"reading {os.path.basename(src)}")
                    info = engine.probe_str(src)
                    v = info["video"]
                    log(f"{v['width']}x{v['height']} @ {v['fps']:g} fps, {v['frames']} frames -> {v['width']*scale}x{v['height']*scale} {kind.upper()}")
                    engine.render(src, tmp, kind, scale=scale, crf=crf, audio=audio, progress=prog, cancel=cancel, info=info, faststart=False)
                    os.replace(tmp, out)
                finally:
                    if os.path.exists(tmp):
                        try: os.remove(tmp)
                        except OSError: pass
                log(f"wrote {out} ({os.path.getsize(out)/1048576:.1f} MB)")
                return dict(output=out, bytes=os.path.getsize(out))
            return self.new_job(work, "export-" + kind)
        raise ValueError("unknown export type")

    # ---- compatibility
    def analyze(self, b):
        import compat
        mod, ref = (b.get("mod") or "").strip(), (b.get("ref") or "").strip()
        if not os.path.isfile(mod): raise ValueError("Choose the modded STR to check.")
        if not os.path.isfile(ref): raise ValueError("Choose the original STR.")
        return compat.analyze(mod, ref)

    def fix(self, b):
        import compat
        mod, ref, out = (b.get("mod") or "").strip(), (b.get("ref") or "").strip(), (b.get("output") or "").strip()
        if not os.path.isfile(mod) and not os.path.isfile((b.get("source") or "").strip()): raise ValueError("Choose the modded file to adjust.")
        if not os.path.isfile(ref): raise ValueError("Choose the original STR.")
        if not out: raise ValueError("Choose where to save the adjusted file.")
        self.check_paths([mod, ref] if os.path.isfile(mod) else [ref], out)
        o = {}
        for k, d in compat.DEFAULTS.items():
            if k in b and b[k] not in (None, ""):
                try: o[k] = type(d)(b[k]) if d is not None and not isinstance(d, bool) else (bool(b[k]) if isinstance(d, bool) else b[k])
                except Exception: raise ValueError(f"Invalid value for '{k}'")
        if (b.get("source") or "").strip():
            if not os.path.isfile(b["source"].strip()): raise ValueError("The original video was not found.")
            o["source"] = b["source"].strip()
            self.check_paths([o["source"]], out)
        for k in ("src_fps", "out_fps"):
            o[k] = None
            if b.get(k) not in (None, ""):
                try: o[k] = float(b[k])
                except Exception: raise ValueError(f"'{k}' must be a number")
                if not 1 <= o[k] <= 60: raise ValueError("Frame rate must be between 1 and 60.")
        def work(prog, cancel, log):
            tmp = out + ".part"
            try:
                st = compat.fix(mod, ref, tmp, o, log=lambda m: log(str(m).replace(tmp, out)), progress=prog, cancel=cancel)
                os.replace(tmp, out)
            finally:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except OSError: pass
            st["output"] = out
            return st
        return self.new_job(work, "fix")

    def reveal(self, path):
        import subprocess
        try:
            if os.name == "nt": subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif sys.platform == "darwin": subprocess.Popen(["open", "-R", path])
            else: subprocess.Popen(["xdg-open", os.path.dirname(path)])
        except Exception: pass

    # ---- file browser
    def listdir(self, path, mode):
        exts = {"str": {".str", ".xa", ".bin", ".sti", ".mov"}, "mp4": {".mp4"}, "avi": {".avi"}, "media": {".str", ".xa", ".bin", ".sti", ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp", ".ogv", ".gif"}, "video": {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg", ".ts", ".gif", ".str"}}.get(mode)
        if not path:
            if os.name == "nt":
                import string
                roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
                home = os.path.expanduser("~")
                shortcuts = [dict(name=n, path=os.path.join(home, n), dir=True) for n in ("Desktop", "Documents", "Downloads", "Videos") if os.path.isdir(os.path.join(home, n))]
                return dict(path="", parent=None, entries=shortcuts + [dict(name=r, path=r, dir=True) for r in roots])
            path = os.path.expanduser("~")
        path = os.path.abspath(path)
        ents = []
        try:
            with os.scandir(path) as it:
                for e in it:
                    try:
                        isd = e.is_dir()
                        if e.name.startswith(".") : continue
                        if isd: ents.append(dict(name=e.name, path=e.path, dir=True))
                        elif mode != "dir" and (exts is None or os.path.splitext(e.name)[1].lower() in exts):
                            ents.append(dict(name=e.name, path=e.path, dir=False, size=e.stat().st_size))
                    except OSError: pass
        except OSError as ex:
            return dict(path=path, parent=os.path.dirname(path), entries=[], error=str(ex))
        ents.sort(key=lambda x: (not x["dir"], x["name"].lower()))
        parent = os.path.dirname(path)
        if parent == path: parent = "" if os.name == "nt" else None
        return dict(path=path, parent=parent, entries=ents)


def make_handler(app):
    WEB = os.path.join(tools.app_dir(), "web")
    if not os.path.isdir(WEB): WEB = os.path.join(getattr(sys, "_MEIPASS", ""), "web")
    MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/mp4", ".webm": "video/webm", ".avi": "video/x-msvideo", ".html": "text/html; charset=utf-8",
            ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png"}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass

        def _ok(self):
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in ("127.0.0.1", "localhost"): return False
            q = parse_qs(urlparse(self.path).query)
            tok = (q.get("t") or [self.headers.get("X-Token")])[0]
            return tok == app.token

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

        def _err(self, code, msg):
            self._json(dict(error=msg), code)

        def _file(self, path, ctype, download=None):
            try: size = os.path.getsize(path)
            except OSError: return self._err(404, "not found")
            rng = self.headers.get("Range")
            start, end, code = 0, size - 1, 200
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                if m:
                    if m.group(1): start = int(m.group(1))
                    if m.group(2): end = int(m.group(2))
                    if not m.group(1) and m.group(2): start = max(0, size - int(m.group(2))); end = size - 1
                    end = min(end, size - 1); code = 206
                    if start > end: return self._err(416, "bad range")
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download: self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start); left = end - start + 1
                    while left > 0:
                        b = f.read(min(1 << 20, left))
                        if not b: break
                        self.wfile.write(b); left -= len(b)
            except (ConnectionError, OSError): pass

        def do_GET(self):
            u = urlparse(self.path); p = u.path; q = parse_qs(u.query)
            if not self._ok(): return self._err(403, "forbidden")
            try:
                if p == "/":
                    html = open(os.path.join(WEB, "index.html"), "rb").read().replace(b"__TOKEN__", app.token.encode())
                    self.send_response(200); self.send_header("Content-Type", MIME[".html"]); self.send_header("Content-Length", str(len(html)))
                    self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(html)
                elif p.startswith("/media/"):
                    name = os.path.basename(unquote(p))
                    self._file(os.path.join(app.cache, name), MIME.get(os.path.splitext(name)[1], "application/octet-stream"))
                elif p.startswith("/direct/"):
                    key = os.path.splitext(os.path.basename(unquote(p)))[0]
                    src = app.direct.get(key)
                    if not src or not os.path.isfile(src): return self._err(404, "not found")
                    self._file(src, MIME.get(os.path.splitext(src)[1].lower(), "video/mp4"))
                elif p == "/api/state":
                    self._json(dict(settings=app.settings, defaults=DEFAULTS, ffmpeg=tools.ffmpeg_path(), cwd=os.getcwd(), home=os.path.expanduser("~"),
                                    sep=os.sep, start=getattr(app, "start_files", []), window=app.window, window_note=app.window_note))
                elif p == "/api/ping":
                    app.last_ping = time.time(); self._json(dict(ok=1))
                elif p == "/api/job":
                    j = app.jobs.get((q.get("id") or [""])[0])
                    if not j: return self._err(404, "no such job")
                    self._json({**{k: j[k] for k in ("id", "kind", "status", "progress", "text", "error", "result")}, "log": j["log"][-200:]})
                elif p == "/api/fs":
                    self._json(app.listdir((q.get("path") or [""])[0], (q.get("mode") or ["any"])[0]))
                else:
                    self._err(404, "not found")
            except Exception as e:
                self._err(500, str(e))

        def do_POST(self):
            if not self._ok(): return self._err(403, "forbidden")
            n = int(self.headers.get("Content-Length") or 0)
            if urlparse(self.path).path == "/api/upload":
                # a file dropped onto the page: the browser cannot tell us where it lives, so it is copied here
                try:
                    name = os.path.basename(unquote((parse_qs(urlparse(self.path).query).get("name") or ["dropped"])[0])) or "dropped"
                    d = os.path.join(app.cache, "dropped", secrets.token_hex(4)); os.makedirs(d, exist_ok=True)
                    dest = os.path.join(d, name); left = n
                    with open(dest, "wb") as f:
                        while left > 0:
                            b = self.rfile.read(min(1 << 20, left))
                            if not b: break
                            f.write(b); left -= len(b)
                    return self._json(dict(path=dest))
                except Exception as e:
                    return self._err(500, str(e))
            try: body = json.loads(self.rfile.read(n) or b"{}")
            except Exception: return self._err(400, "bad json")
            p = urlparse(self.path).path
            try:
                if p == "/api/settings":
                    self._json(app.save_settings(body))
                elif p == "/api/open":
                    fmt = body.get("fmt", "mp4"); fmt = fmt if fmt in ("mp4", "webm") else "mp4"
                    j = app.open_preview(body["path"], fmt, int(body.get("scale") or app.settings["preview_scale"]))
                    self._json(dict(job=j["id"]))
                elif p == "/api/probe":
                    info = engine.probe_str(body["path"])
                    cap = info["sectors"] / info["sectors_per_second"]
                    self._json(dict(info=info, rows=engine.fmt_info(info), capacity=cap))
                elif p == "/api/export":
                    try: j = app.export(body)
                    except ValueError as e: return self._err(400, str(e))
                    self._json(dict(job=j["id"]))
                elif p == "/api/analyze":
                    try: self._json(app.analyze(body))
                    except ValueError as e: return self._err(400, str(e))
                elif p == "/api/fix":
                    try: j = app.fix(body)
                    except ValueError as e: return self._err(400, str(e))
                    self._json(dict(job=j["id"]))
                elif p == "/api/exists":
                    self._json(dict(exists=os.path.exists(body["path"])))
                elif p == "/api/reveal":
                    app.reveal(body["path"]); self._json(dict(ok=1))
                elif p == "/api/cancel":
                    j = app.jobs.get(body.get("job"))
                    if j: j["cancel"] = True
                    self._json(dict(ok=1))
                else:
                    self._err(404, "not found")
            except FileNotFoundError as e:
                self._err(404, f"file not found: {e}")
            except Exception as e:
                self._err(500, str(e))
    return H

def start(app, port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    srv.daemon_threads = True
    app.port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
