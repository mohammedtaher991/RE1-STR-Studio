#!/usr/bin/env python3
"""RE1 STR Studio - preview, compare, convert and export PlayStation STR movies.
    python studio.py [ORIGINAL.STR] [MODDED.STR]
"""
import os, shutil, subprocess, sys, threading, time, webbrowser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server, tools

def find_app_browser():
    cands = []
    if os.name == "nt":
        for env in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData"):
            b = os.environ.get(env)
            if b:
                cands += [os.path.join(b, r"Microsoft\Edge\Application\msedge.exe"), os.path.join(b, r"Google\Chrome\Application\chrome.exe"),
                          os.path.join(b, r"BraveSoftware\Brave-Browser\Application\brave.exe")]
    else:
        cands += [shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave-browser")]
        cands += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    return next((c for c in cands if c and os.path.isfile(c)), None)

def selftest():
    """Checks every runtime piece the program needs (used by build_exe.bat on the finished .exe)."""
    import io, tempfile
    out = []
    ok = True
    def check(name, fn):
        nonlocal ok
        try:
            out.append(f"OK    {name}: {fn()}")
        except Exception as e:
            ok = False; out.append(f"FAIL  {name}: {e}")
    def ff():
        r = tools.run(["-hide_banner", "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return r.stdout.decode("utf8", "ignore").splitlines()[0]
    def enc():
        miss = [e for e in ("libx264", "aac", "mpeg4", "pcm_s16le") if not tools.has_encoder(e)]
        if miss: raise RuntimeError("ffmpeg lacks encoders: " + ", ".join(miss))
        return "libx264, aac, mpeg4, pcm_s16le present"
    def npy():
        import numpy; return numpy.__version__
    def codec():
        import numpy as np, bsenc, xaenc
        Y = (np.random.RandomState(1).rand(240, 320) * 255).astype(np.uint8)
        C = np.full((120, 160), 128, np.uint8)
        e = bsenc.encode_frame(Y, C, C, 7 * 2016)
        a = xaenc.encode((np.random.RandomState(2).randn(4032, 2) * 3000).astype(np.int16))
        return f"video frame {len(e['data'])} bytes (q{e['qs']}), {len(a)} audio sectors"
    def clip():
        tmp = os.path.join(tempfile.gettempdir(), "re1str_selftest.mp4")
        tools.run(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp],
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n = os.path.getsize(tmp); os.remove(tmp)
        if n < 500: raise RuntimeError("test clip not written")
        return f"made a test clip ({n} bytes)"
    def web():
        d = os.path.join(tools.app_dir(), "web")
        if not os.path.isdir(d): d = os.path.join(getattr(sys, "_MEIPASS", ""), "web")
        return os.path.join(d, "index.html") if os.path.isfile(os.path.join(d, "index.html")) else (_ for _ in ()).throw(RuntimeError("web/index.html missing"))
    check("ffmpeg", ff); check("ffmpeg encoders", enc); check("numpy", npy); check("encoders", codec); check("ffmpeg run", clip); check("interface files", web)
    out.append("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    text = "\n".join(out)
    try: open(os.path.join(tools.app_dir() if not getattr(sys, "frozen", False) else os.path.dirname(sys.executable), "selftest_result.txt"), "w").write(text + "\n")
    except OSError: pass
    try: print(text)
    except Exception: pass
    return 0 if ok else 1

def run_native_window(app, url):
    """Show the interface in the program's own window (pywebview: Edge WebView2 on Windows) instead of a browser window.
    Returns True when that window was shown and has been closed, False when it is not available (the caller then
    falls back to an app-style browser window)."""
    try:
        import webview
    except Exception as e:
        app.window_note = "pywebview is not installed (%s). Run: python -m pip install pywebview" % e
        return False
    try:
        win = webview.create_window("RE1 STR Studio", url, width=1320, height=820, min_size=(900, 600), background_color="#0e1014")
        try:       # lets the page see the real path of a file dropped onto it (pywebview 5+)
            from webview.dom import DOMEventHandler
            win.dom.document.events.drop += DOMEventHandler(lambda e: None, True, True)
        except Exception:
            pass
        kw = dict(storage_path=os.path.join(app.data, "webview"), private_mode=False)
        if os.name == "nt": kw["gui"] = "edgechromium"      # never the old Internet Explorer engine
        webview.start(**kw)
        return True
    except Exception as e:
        import traceback
        app.window_note = "The program's own window could not start: %s" % e
        try:
            open(os.path.join(app.data, "native_window_error.txt"), "w").write(traceback.format_exc())
            print("native window not available:", e)
        except Exception: pass
        return False

def main(argv):
    if "--selftest" in argv:
        sys.exit(selftest())
    if "-h" in argv or "--help" in argv:
        print(__doc__); return
    if sys.stdout is None: sys.stdout = open(os.devnull, "w")       # windowed (no console) builds
    if sys.stderr is None: sys.stderr = open(os.devnull, "w")
    files = [os.path.abspath(a) for a in argv if os.path.isfile(a)]
    app = server.App()
    app.start_files = files[:2]
    srv = server.start(app)
    url = f"http://127.0.0.1:{app.port}/?t={app.token}"
    print("RE1 STR Studio running at", url)
    if not tools.ffmpeg_path():
        print("WARNING: ffmpeg not found - install ffmpeg or `pip install imageio-ffmpeg`")
    app.window = "native"
    if not os.environ.get("RE1STR_NOWINDOW") and not os.environ.get("RE1STR_BROWSER"):
        if run_native_window(app, url):             # the program's own window; returns when it is closed
            srv.shutdown(); return
    app.window = "browser"
    exe = None if os.environ.get("RE1STR_NOWINDOW") else find_app_browser()
    proc = None
    if exe:
        prof = os.path.join(app.data, "browser-profile")
        proc = subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={prof}", "--window-size=1320,820", "--no-first-run",
                                 "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required"])
    elif not os.environ.get("RE1STR_NOWINDOW"):
        webbrowser.open(url)
    # Exit only when the window is really gone. A page that is minimised, hidden or asleep sends its heartbeat rarely
    # (browsers slow background timers to about once a minute, and a sleeping PC sends none), so a missed heartbeat
    # alone must never stop the program.
    idle_limit = float(os.environ.get("RE1STR_IDLE", 1800))
    t0 = time.time(); prev = t0
    try:
        while True:
            time.sleep(1)
            now = time.time()
            if now - prev > 8:                       # the computer was asleep: start counting again
                app.last_ping = now if app.last_ping else None; t0 = now
            prev = now
            if app.last_ping is None:
                if now - t0 > 120: break             # the window never opened
                continue
            stale = now - app.last_ping
            if proc and proc.poll() is not None and stale > 180: break      # window process ended and nothing talks to us
            if stale > idle_limit: break
    except KeyboardInterrupt:
        pass
    srv.shutdown()

if __name__ == "__main__":
    main(sys.argv[1:])
