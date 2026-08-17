# Meowmeow cat cam meme detector

**Live demo:** https://jhuang165.github.io/meowmeowcatcam/ (once GitHub Pages is enabled — see below)

Point your webcam at yourself, make a face/hand gesture, get a cat meme back in real time. Runs either as a desktop app (OpenCV windows) or entirely in the browser (MediaPipe WASM, no install).

Two windows/panes side by side: 
- **Camera** — your webcam feed with hand landmarks drawn on top, plus a live debug readout in the corner
- **Meme** — the meme matching whatever gesture you're currently making

## Gestures

Checked in this order — when a pose could match more than one, the earlier one wins.

| # | Gesture | How to trigger |
|---|---|---|
| 1 | Muehehe | Both hands up, index fingers only, tips touching |
| 2 | Devo cat | Both hands up, above the top of your head |
| 3 | Crash out cord chewing kitty | Both hands up beside your face to hold yummy electrical cable |
| 4 | I will punch you | One hand, all four fingers curled |
| 5 | EHHEHEEEHEEEE | Thumb + pinky out, rockstar cat |
| 6 | Shhh silenced cat | Index finger only, tip resting on your mouth |
| 7 | Erm ackshuALLY! cat | Index finger only, held away from your face |
| 8 | Shocked/kidnapped cat | Hand cover mouth |
| 9 | gGIMME MONIE!! | One open palm, all fingers extended, away from your face |
| 10 | Side eye cat | Turn your head 15°+ either way (real head-pose yaw) |
| 11 | Pokercat | Default |
| 12 | Spinny OIIAI cat | You spin!!!! |


Meme images live in `memes/`. A couple of gestures pick randomly between multiple images.

## Running it — desktop (Python)

Requires Python 3 and a webcam.

Easiest way: just double-click **`Launch Gesture Meme.command`**. First run takes a minute to set itself up (installs everything automatically), then launches straight away. Every run after that is instant.

**First time opening it:** macOS will warn "cannot be opened because it is from an unidentified developer" — this is normal for any downloaded script, not specific to this one. Right-click the file → **Open** → click **Open** in the dialog that appears. You only need to do this once.

Or manually, if you prefer Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 gesture_meme.py
```

Press `q` or `Esc` in the Camera window to quit.

## Running it — browser

No install needed, but the webcam API requires serving over HTTP (opening `index.html` directly as a `file://` URL will not get camera permission). From this folder:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` and allow camera access. Models load from Google's hosted MediaPipe CDN at runtime, so nothing local is needed for the browser version.

## Hosting it on GitHub Pages

The browser version is entirely static (HTML/JS/images, no backend), so GitHub Pages can serve it directly with no build step:

1. On GitHub, go to **Settings → Pages**
2. Under **Build and deployment**, set **Source** to "Deploy from a branch"
3. Pick branch **main**, folder **/ (root)**, then **Save**
4. Wait a minute for the first deploy, then visit `https://<username>.github.io/meowmeowcatcam/`

Camera access requires HTTPS, which Pages provides automatically. A `.nojekyll` file is included so GitHub doesn't run the site through Jekyll (which can mangle filenames with spaces, like the ones in `memes/`).

## Live debug HUD

The Camera window always shows a small readout in the top-left corner:

```
gesture: sideEyeCat
yaw: +18.4 deg  (side-eye thr +/-15.0)
```

Useful for tuning the detection thresholds at the top of `gesture_meme.py` / `app.js` if a gesture is triggering too easily or not easily enough for your setup/lighting.

## Project layout

```
gesture_meme.py   desktop version (OpenCV + MediaPipe Python tasks API)
app.js            browser version (MediaPipe tasks-vision WASM)
index.html        browser UI shell
memes/            meme images (+ one video, unused for now)
models/           MediaPipe .task model files used by the desktop version
requirements.txt  Python dependencies
```
