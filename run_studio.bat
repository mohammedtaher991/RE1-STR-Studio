@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.9 or newer from python.org ^(tick "Add python.exe to PATH"^), then run this again.
  pause & exit /b 1
)
if not exist ".deps_ok" (
  echo Installing needed packages ^(first run only^)...
  python -m pip install --user numpy imageio-ffmpeg
  if errorlevel 1 ( echo Package install failed. & pause & exit /b 1 )
  echo ok> .deps_ok
)
if not exist ".webview_tried" (
  echo Installing the program's own window ^(optional^)...
  python -m pip install --user pywebview
  if errorlevel 1 ( echo  Could not install it. The program will open in a browser-style window instead. ) else ( echo  Done. )
  echo tried> .webview_tried
)
start "" pythonw studio.py %*
