# RE1 STR Studio

Preview, compare, convert and export PlayStation `.STR` movies (made for Resident Evil 1 Director's Cut, works with
any standard 2x CD-XA STR).

## Give it to other people (no installs for them)
Run **`build_exe.bat`** once on your Windows PC. It makes **`dist\RE1_STR_Studio.exe`**: a single file that already contains
Python, the encoder libraries and ffmpeg. Send just that file; people double-click it and nothing else needs installing.
(The build itself needs Python + internet, once, on your PC only. No Python PC? Use the GitHub route in `.github/workflows`.)

## Start it from the folder (for you)
* **Windows:** double-click `run_studio.bat` (first run installs `numpy` and `imageio-ffmpeg`; needs Python 3.9+).
* **Other:** `./run_studio.sh`  (or `python3 studio.py`)
* Optional: `python studio.py ORIGINAL.STR MODDED.STR` opens both straight away.

It opens in its own window (Edge/Chrome app window; your default browser if none is found) and quits when you close it.
Everything stays on your computer.

## View tab
* Open the **original** STR and your **modded** STR. Switch between them (`1` / `2`), or turn on
  **Compare side by side** (off by default, needs both files). In compare mode choose which one the sound comes from.
* Buttons: ◀ jump back, play/pause, jump forward ▶, speed, volume, loop, full screen icon.
* Keys: `←` `→` jump (length set in Settings: 5, 10, 15, 20, 25 or 30 s) · `Space` play/pause · `Enter` or `F` full screen
  (`Esc` exits) · `,` `.` one frame · `↑` `↓` volume · `M` mute · `L` loop · `C` compare · `[` `]` speed · `Home` `End`.
* The right-hand panel shows the **STR information**: type, container, size, duration, video size, fps, frames,
  codec/BS version, bitrate, audio rate (Hz), bits, channels, audio sectors and the CD-XA file/channel numbers.

## Export tab
* **Video → STR**: any video + the original STR you are replacing → a game-ready STR that copies the original's layout
  (resolution, fps, sector budget per frame, audio file/channel, flags). By default the result is padded to exactly the
  original's size, so it can replace the file in place. If your video is longer than fits, it is cut and the log says so.
* **STR → MP4 / AVI**: converts any STR (original or modded) to a normal video, 1×–4× size, sound included.
* Originals are never overwritten. Progress bar + Cancel for everything.

## Adjust tab (make a modded STR compatible)
Pick the **modded file** (an STR, or an ordinary mp4/avi) and the **original STR** it replaces, then press *Check compatibility*.
The table shows every difference with a green / yellow / red dot. *Adjust* then fixes what is wrong and saves a new file (the originals are never changed):

- wrong CD-XA file/channel numbers (the usual cause of **muted sound**), missing end-of-movie marker
- broken sector checksums, wrong first sector, RIFF vs raw container
- clips that were split into several pieces (joined into one movie; clips that run at different speeds are each re-timed by real time so every part plays at the right speed)
- picture playing too fast/slow (frames re-timed to the original's frame rate), plus a manual speed factor
- sound delay (ms) and volume (dB)
- file larger than the original (the end is cut, or "Speed up to fit" plays it slightly faster so all of it stays) or smaller (padded to the exact size)
- STR files written by other tools: no sector headers (2336-byte sectors), damaged checksums, wrong channel, no end marker, two clips

**Process from the original video:** if you still have your source mp4/avi, put it in that box. The picture and sound are then taken from it (not from the already converted STR), which fixes speed problems that cannot be repaired from the STR alone.

If the video data itself is fine, it is copied **unchanged** (lossless). It is only re-compressed when it has to be
(wrong picture size/format, frames too big for the original's space) or when you choose *Always*.
An ordinary mp4/avi (or mkv, mov, webm…) is simply converted into a game-ready STR. The output frame rate defaults to the modded file's own rate, built exactly (15 fps = 15.00 in the sector layout). "Lower the volume if it is too loud" prevents sound clipping.

## Settings tab
Jump length, picture scaling (smooth / sharp), preview size, loop, auto-hide of the controls in full screen.

## Notes
* Previews and settings are cached in `%LOCALAPPDATA%\RE1STRStudio`.
* If a preview does not play, your browser lacks H.264; the program then makes a WebM preview automatically.
* The command-line tool still works: `python re1str.py encode video.mp4 --ref ORIGINAL.STR -o NEW.STR`.
* Tested by decoding with ffmpeg, a browser test of the whole interface, and byte-exact rebuild of a real game STR.
  Not tested on real hardware or an emulator.

## The program window
If `pywebview` is installed (it is part of `requirements.txt` and `build_exe.bat` tries to include it), RE1 STR Studio opens in its own
window (Microsoft Edge WebView2, which is already part of Windows 10/11) and the program ends when you close that window.
If it is missing or cannot start, the program falls back to an app-style Edge/Chrome window. Set `RE1STR_BROWSER=1` to force that fallback.
