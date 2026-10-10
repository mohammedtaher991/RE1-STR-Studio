@echo off
setlocal enableextensions
cd /d "%~dp0"
echo.
echo  ===== Building RE1_STR_Studio.exe  (one file, nothing else needed to run it) =====
echo  This needs Python 3.9+ and internet ONLY on this computer, ONLY to build.
echo.
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo Python was not found. Install it from python.org ^(tick "Add python.exe to PATH"^) and run this again.
  goto :fail
)
%PY% -m venv .buildenv || goto :fail
call .buildenv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip >nul 2>nul
python -m pip install numpy imageio-ffmpeg pyinstaller || goto :fail
rem the program's own window (optional: without it the program opens an app-style browser window instead)
set "WV="
python -m pip install pywebview >nul 2>nul && set "WV=--hidden-import webview --hidden-import clr --collect-all webview --collect-all pythonnet --collect-all clr_loader"
if defined WV ( echo  Native window support: yes ) else ( echo  Native window support: no ^(pywebview could not be installed; the browser-style window is used^) )
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
set "COMMON=--noconfirm --clean --onefile --noconsole --name RE1_STR_Studio --add-data "web;web" --collect-all imageio_ffmpeg --hidden-import re1str --hidden-import engine --hidden-import server --hidden-import tools --hidden-import psxcd --hidden-import mux --hidden-import bsenc --hidden-import bs_tables --hidden-import xaenc"
pyinstaller %COMMON% %WV% studio.py
if errorlevel 1 (
  if defined WV (
    echo  Building with the native window failed. Trying again without it...
    if exist dist rmdir /s /q dist
    if exist build rmdir /s /q build
    pyinstaller %COMMON% studio.py || goto :fail
  ) else goto :fail
)
echo.
echo  Testing the finished program...
start /wait "" "dist\RE1_STR_Studio.exe" --selftest
if not exist "dist\selftest_result.txt" ( echo Self-test did not run. & goto :fail )
type "dist\selftest_result.txt"
findstr /c:"SELFTEST PASSED" "dist\selftest_result.txt" >nul || goto :fail
del "dist\selftest_result.txt" >nul 2>nul
echo.
echo  DONE:  dist\RE1_STR_Studio.exe
echo  Send that single file to anyone. They do not need Python, ffmpeg or anything else.
echo.
pause & exit /b 0
:fail
echo.
echo  The build failed. Copy the messages above and send them to whoever helps you.
pause & exit /b 1
