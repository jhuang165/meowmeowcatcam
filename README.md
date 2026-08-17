# Meowmeow cat cam meme detector

**Live demo:** https://jhuang165.github.io/meowmeowcatcam/ (once GitHub Pages is enabled — see below)

Point your webcam at yourself, make a face/hand gesture, get a cat meme back in real time. Runs either as a desktop app (OpenCV windows) or entirely in the browser (MediaPipe WASM, no install).

Two windows/panes side by side: 
- **Camera** — your webcam feed with hand landmarks drawn on top, plus a live debug readout in the corner
- **Meme** — the meme matching whatever gesture you're currently making

## Gestures

Detection first buckets your hand by finger shape (fist vs. pointing vs. open palm, etc.) — those buckets are a genuine precedence, since they're mutually exclusive. *Within* a bucket, close calls (fist vs. thumbsUp, shhh vs. "professor cat", crash-out vs. two-hands-on-head) are scored continuously and decided by whichever reading has the clearest margin, rather than a single hard cutoff — so a pose sitting right on a boundary holds its last confident reading instead of flickering between the two. A pose that's a genuine toss-up between two gestures shows neither until it resolves. Spinning beats everything else, hands included.

| Gesture | How to trigger |
|---|---|
| Muehehe | Both hands up, index fingers only, tips touching |
| Devo cat | Both hands up, above the top of your head |
| Crash out cord chewing kitty | Both hands up beside your face to hold yummy electrical cable |
| Thumbs up cat | One hand, thumb stuck out from an otherwise-curled fist, pointing up |
| I will punch you | One hand, all four fingers curled, thumb tucked in |
| EHHEHEEEHEEEE | Thumb + pinky out, rockstar cat |
| Shhh silenced cat | Index finger only, tip resting on your mouth |
| Erm ackshuALLY! cat | Index finger only, pointing up, held away from your face |
| Shocked/kidnapped cat | Hand cover mouth |
| gGIMME MONIE!! | One open palm, all fingers extended, away from your face |
| Tongue out cat | No hands, mouth open wide |
| Side eye cat | Turn your head 15°+ either way (real head-pose yaw) |
| Pokercat | Default |
| Spinny OIIAI cat | You spin!!!! |

Meme images live in `memes/`. A couple of gestures pick randomly between multiple images. All the detection thresholds live in `gesture_config.json`, shared by both the desktop and browser versions — see [Live debug HUD](#live-debug-hud) below for tuning them.

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

## Using it as a Zoom camera

The desktop (Python) version can also publish the meme feed as a virtual camera, so Zoom, Meet, Teams, etc. see it as a normal webcam and show the reacting cat instead of your face. The virtual camera always shows a full-screen meme — `pokercat.jpg` by default, switching to match whatever gesture you're making. Your actual camera feed is never sent out; it only appears in the local "Camera" preview window used for tuning gestures.

macOS has no supported way to publish a camera device without a signed system extension, so this works by handing frames to [OBS Studio](https://obsproject.com/)'s virtual camera, which Zoom already knows how to see.

**One-time setup:**

1. Install or upgrade OBS Studio to **version 30 or newer** (`Applications → OBS → About` shows the current version — older versions aren't supported here).
2. Open OBS, click **Start Virtual Camera** (bottom right), approve the macOS system-extension permission prompt if one appears, then click **Stop Virtual Camera** and quit OBS. This installs the camera device — OBS doesn't need to be running afterward.

**Every time:**

Double-click **`Launch Meowmeow Virtual Cam.command`** (or run `python3 gesture_meme.py --virtual-cam`). In Zoom, open Settings → Video and select **OBS Virtual Camera**.

## Live debug HUD

The Camera window (desktop) / camera pane (browser) always shows a small readout in the top-left corner. Desktop also adds the optical-flow numbers behind spin detection:

```
gesture: sideEyeCat
yaw: +18.4 deg  (side-eye thr +/-15.0)
mouthOpen: 0.04  (tongue-out thr 0.12)
flow mag: 0.12  (thr 0.80)
spin fraction (2.2s window): 0.03  (thr 0.55)
peak score (last 2s): 0.09  <- read this AFTER you stop spinning
top: shhh=0.71  2nd: oneFingerUp=0.52  margin=0.19 (thr 0.12, floor 0.35)
```

That last line is the one to watch when a gesture is triggering too easily, not easily enough, or flickering between two readings: it shows the two closest-scoring hand gestures and the margin between them, straight from `gesture_config.json`. Widen the margin/floor there if a gesture flickers; narrow the specific distance/angle threshold behind it (also in `gesture_config.json`, with a comment on what it controls) if a gesture isn't triggering, or triggers too easily, for your hand/lighting/camera angle. Both the desktop and browser versions read the same file, so a change there applies to both.

## Project layout

```
gesture_meme.py       desktop version (OpenCV + MediaPipe Python tasks API)
app.js                browser version (MediaPipe tasks-vision WASM)
gesture_config.json   every tunable threshold, shared by both versions above
index.html            browser UI shell
memes/                meme images (+ one video, unused for now)
models/               MediaPipe .task model files used by the desktop version
requirements.txt      Python dependencies
```
