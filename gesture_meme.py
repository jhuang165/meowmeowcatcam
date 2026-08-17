"""
Webcam gesture -> meme detector (desktop version).

Opens two windows, side by side like the OBS/streamer setups:
  - "Camera": your webcam feed with hand landmarks drawn on top
  - "Meme": the cat meme matching whatever gesture you're making

Gestures:
  rockstar / shaka  -> memes/cat.jpg
  default (no hand) -> memes/pokercat.jpg
  one finger up     -> memes/profcat.jpg, memes/professorcat.jpg
  fist / punch      -> memes/punchcat.jpg
  thumbs up (fist with thumb stuck out) -> memes/cat_thumbs_up.png
  shhh              -> memes/shhcat.jpg
  two fingers together (both hands, tips touching) -> memes/uwucat.jpg, memes/uwucatt.jpg,
                                                        memes/fingers together muehehe .jpg
  hand covering face -> memes/hand cover face .jpg
  crash-out cat (two hands up beside the face)            -> memes/crashout cat .jpg
  two hands on head                                        -> memes/two hands on head .jpg
  hand stretched out, palm facing camera (open hand)       -> memes/hand stretched out, palm facing up .jpg
  tongue out (mouth open wide, no hands)                   -> memes/cat_tongue.jpeg
  side eye (head turned to the side)                       -> memes/side eye cat.jpg
  spin cat (spinning fast in your chair)                   -> memes/spin cat.mov (plays as a video)

The Camera window shows a live debug readout (head yaw, optical-flow
magnitude/coherence, and the top-2 scored hand gestures with their margin)
vs. their trigger thresholds in the top-left corner, so every gesture can be
tuned by eye - see gesture_config.json for the actual numbers.

Press q or ESC to quit.

Pass --virtual-cam to also publish the meme feed as a virtual camera (via
OBS Studio 30+) that Zoom/Meet/Teams can select as a webcam - see README.md.

The webcam is auto-picked to skip the iPhone Continuity Camera and the OBS
Virtual Camera output; pass --camera-index to override that.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
except ImportError:
    pyvirtualcam = None

try:
    import AVFoundation
except ImportError:
    AVFoundation = None

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "memes"

# fixed canvas the virtual camera publishes at - pyvirtualcam locks
# resolution/fps for the session, so this has to be picked up front rather
# than following whatever size the current meme happens to be.
VIRTUAL_CAM_WIDTH = 1280
VIRTUAL_CAM_HEIGHT = 720
VIRTUAL_CAM_FPS = 30

GESTURE_MEMES = {
    "rockstar": ["cat.jpg"],
    "default": ["pokercat.jpg"],
    "oneFingerUp": ["profcat.jpg", "professorcat.jpg"],
    "fist": ["punchcat.jpg"],
    "thumbsUp": ["cat_thumbs_up.png"],
    "shhh": ["shhcat.jpg"],
    "twoFingersTogether": ["uwucat.jpg", "uwucatt.jpg", "fingers together muehehe .jpg"],
    "handCoverFace": ["hand cover face .jpg"],
    "crashOutCat": ["crashout cat .jpg"],
    "twoHandsOnHead": ["two hands on head .jpg"],
    "handStretchedOut": ["hand stretched out, palm facing up .jpg"],
    "tongueOut": ["cat_tongue.jpeg"],
    "sideEyeCat": ["side eye cat.jpg"],
    "spinCat": ["spin cat.mov"],
}

# gestures whose meme is a video, not a still image
VIDEO_GESTURES = {"spinCat"}

# every numeric tunable (finger-extension curves, hysteresis levels,
# per-gesture distance/angle thresholds, vote timing, spin detection) lives
# in gesture_config.json instead of here, so the desktop (this file) and
# browser (app.js) versions read the same numbers and can't drift apart the
# way they already had (see git history: app.js never gained the spin
# feature this file has). Watch the live HUD in the Camera window to tune
# any of these - see draw_debug_hud below.
CFG = json.loads((ROOT / "gesture_config.json").read_text())

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- geometry helpers (ported from the JS version) -----------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def p2(lm):
    return np.array([lm.x, lm.y])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def dist2(a, b):
    """2D (x, y only) distance. Used for every hand<->face and hand<->hand
    comparison: hand-landmarker z is relative to that hand's own wrist and
    face-landmarker z is relative to the head, so a 3D dist() across two
    different detectors' landmarks mixes two unrelated depths and adds
    noise, not signal. Full 3D dist() is still correct for measurements
    within a single hand's own landmarks (see finger_extension below)."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def ramp(x, lo, hi):
    """0 at/below lo, 1 at/above hi, linear in between. The building block
    for every soft threshold below - replaces a hard boolean cliff with a
    score that degrades gracefully near the boundary instead of flipping."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def latch(prev, score, hi, lo):
    """Schmitt-trigger hysteresis: flip on above hi, flip off below lo,
    otherwise hold the previous state. Turns a continuous score into a
    stable boolean that can't flap every frame just because the score is
    hovering near a single cutoff."""
    if score >= hi:
        return True
    if score <= lo:
        return False
    return prev


FINGER_CHAINS = {
    # (mcp, pip, dip, tip) landmark indices for each finger's chain. The
    # thumb only has three true segments (cmc-mcp-ip-tip); using the same
    # 4-point shape as the other fingers keeps finger_extension generic.
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def finger_extension(world_pts, chain, cfg):
    """Continuous [0,1] "how extended is this finger" score, computed from
    hand WORLD landmarks (MediaPipe's metric, perspective-corrected 3D
    estimate) rather than image landmarks. angle_deg/dist ratios are
    rotation-invariant, so this stays correct however the hand is turned -
    including pointed straight at the camera, where the old image-space
    angle test would foreshorten a straight finger past its 45-degree
    cutoff and misread it as curled.

    Blends two cues:
      - curl: total bend across mcp->pip->dip->tip (0 = dead straight)
      - reach: chord (mcp->tip) over arc (sum of the three segments) -
        ~1.0 straight, ~0.5 fully curled. This ratio is what actually
        survives foreshortening, since numerator and denominator shrink
        together under projection - it's weighted higher than curl for
        that robustness.
    """
    mcp, pip, dip, tip = chain
    seg1 = world_pts[pip] - world_pts[mcp]
    seg2 = world_pts[dip] - world_pts[pip]
    seg3 = world_pts[tip] - world_pts[dip]
    curl = angle_deg(seg1, seg2) + angle_deg(seg2, seg3)

    arc = dist(world_pts[mcp], world_pts[pip]) + dist(world_pts[pip], world_pts[dip]) + dist(world_pts[dip], world_pts[tip])
    chord = dist(world_pts[mcp], world_pts[tip])
    reach = chord / arc if arc > 1e-9 else 0.0

    angle_score = 1.0 - ramp(curl, cfg["curlAngleLoDeg"], cfg["curlAngleHiDeg"])
    reach_score = ramp(reach, cfg["reachLo"], cfg["reachHi"])
    return cfg["angleWeight"] * angle_score + cfg["reachWeight"] * reach_score


def angle_from_vertical_deg(v2d):
    """Angle (degrees) between a 2D image-space vector and "straight up"
    (image y grows downward, so up is (0, -1)). Only used for direction
    tests (which way is the thumb/finger pointing) - unlike finger shape,
    "up" is inherently a screen-relative question, so this deliberately
    uses image landmarks rather than hand-world landmarks."""
    v = np.array([v2d[0], v2d[1], 0.0])
    up = np.array([0.0, -1.0, 0.0])
    return angle_deg(v, up)


def yaw_from_transform_matrix(matrix):
    """Extract the head's left/right turn angle (yaw, degrees) from
    MediaPipe's facial transformation matrix - its own estimate of head
    pose, far more robust than trying to infer turn from landmark
    distances."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)


def classify_hand(image_landmarks, world_landmarks, key, finger_state, now, cfg):
    """Build the per-hand feature set decide() scores gestures from: a
    continuous extension score per finger (world landmarks - see
    finger_extension), hysteresis-latched up/down booleans per finger
    (keyed by handedness label so each hand's state persists across
    frames independently), and the image-space points needed for any
    hand<->face or hand<->hand comparison."""
    img_pts = [p2(lm) for lm in image_landmarks]
    world_pts = [p3(lm) for lm in world_landmarks]

    hand_scale_world = dist(world_pts[0], world_pts[9]) or 1e-6
    hand_scale_img = dist2(img_pts[0], img_pts[9]) or 1e-6

    fcfg = cfg["finger"]
    ext = {name: finger_extension(world_pts, chain, fcfg) for name, chain in FINGER_CHAINS.items()}

    tcfg = cfg["thumb"]
    thumb_spread = dist(world_pts[4], world_pts[17]) / hand_scale_world
    thumb_spread_score = ramp(thumb_spread, tcfg["spreadLo"], tcfg["spreadHi"])

    # thumb direction (up vs down): image-space only, see angle_from_vertical_deg.
    thumb_dir = img_pts[4] - img_pts[2]
    thumb_up_angle = angle_from_vertical_deg(thumb_dir)
    thumb_up_score = 1.0 - ramp(thumb_up_angle, tcfg["upAngleLoDeg"], tcfg["upAngleHiDeg"])

    state = finger_state.setdefault(
        key, {"index": False, "middle": False, "ring": False, "pinky": False, "thumb": False}
    )
    state["last_seen"] = now
    up = {}
    for name in ("index", "middle", "ring", "pinky"):
        up[name] = latch(state[name], ext[name], fcfg["hysteresisHigh"], fcfg["hysteresisLow"])
        state[name] = up[name]
    state["thumb"] = latch(state["thumb"], thumb_spread_score, tcfg["hysteresisHigh"], tcfg["hysteresisLow"])

    curled_count = sum(1 for name in ("index", "middle", "ring", "pinky") if not up[name])

    return {
        "ext": ext,
        "indexUp": up["index"],
        "middleUp": up["middle"],
        "ringUp": up["ring"],
        "pinkyUp": up["pinky"],
        "thumbOut": state["thumb"],
        "thumbSpreadScore": thumb_spread_score,
        "thumbUpScore": thumb_up_score,
        "curledCount": curled_count,
        "handScaleImg": hand_scale_img,
        "indexTipImg": img_pts[8],
        "indexPipImg": img_pts[6],
        "wristImg": img_pts[0],
        "palmCenterImg": img_pts[9],
    }


def is_pointing(h):
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


def frame_flow_signal(frame, prev_small_gray, cfg):
    """Downsize + compute dense optical flow against the previous frame,
    then reduce it to (magnitude, coherence): how much of the frame moved
    on the horizontal axis, and what fraction of that motion agreed on one
    direction. Returns (magnitude, coherence, small_gray_for_next_call)."""
    small = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (cfg["flowWidth"], cfg["flowHeight"])
    )
    if prev_small_gray is None:
        return 0.0, 0.0, small

    flow = cv2.calcOpticalFlowFarneback(
        prev_small_gray, small, None, 0.5, 2, 15, 2, 5, 1.2, 0
    )
    flow_x = flow[..., 0]

    magnitude = float(np.abs(flow_x).mean())

    moving_mask = np.abs(flow_x) > cfg["noiseFloorPx"]
    moving_count = int(moving_mask.sum())
    total = flow_x.size
    if moving_count / total < cfg["minMovingFraction"]:
        coherence = 0.0
    else:
        mean_sign = np.sign(flow_x[moving_mask].mean())
        if mean_sign == 0:
            coherence = 0.0
        else:
            agree = int((np.sign(flow_x[moving_mask]) == mean_sign).sum())
            coherence = agree / moving_count

    return magnitude, coherence, small


def hand_state_keys(hand_result):
    """A stable per-hand key for this frame, from MediaPipe's own handedness
    label, so finger hysteresis state (see classify_hand) persists across
    frames for "the same hand" rather than by list position (which can swap
    if hands cross or one drops out). Suffixed on the rare collision where
    both hands get the same label."""
    seen = {}
    keys = []
    for cats in hand_result.handedness:
        label = cats[0].category_name if cats else "hand"
        n = seen.get(label, 0)
        seen[label] = n + 1
        keys.append(label if n == 0 else f"{label}{n}")
    return keys


def compute_hand_scores(hands, face_is_fresh, last_face, face_seen_this_frame, cfg):
    """Build a {gesture: score in [0,1]} dict for every hand-shape gesture
    the current hand(s) could plausibly be. The coarse routing below (which
    finger-count bucket a hand falls into) mirrors the original cascade's
    genuine precedence - those are mutually exclusive finger configurations,
    not real ties. What's new is that the decision *within* a bucket (fist
    vs thumbsUp; shhh vs oneFingerUp; twoHandsOnHead vs crashOutCat) is a
    continuous, scored comparison instead of a single hard-cliff threshold,
    so decide()'s caller can pick the winner by margin and treat a close
    call as "ambiguous" rather than snapping to whichever side of the cliff
    a noisy frame landed on."""
    hcfg = cfg["hand"]
    scores = {}

    if len(hands) == 2:
        h0, h1 = hands
        if is_pointing(h0) and is_pointing(h1):
            avg_scale = (h0["handScaleImg"] + h1["handScaleImg"]) / 2
            tip_gap = dist2(h0["indexTipImg"], h1["indexTipImg"]) / avg_scale
            # only a candidate at all once the tips are at least roughly
            # close (matching the original's tip_gap < tipGapHi cutoff for
            # even considering this gesture) - otherwise two hands each
            # independently pointing somewhere would always add a
            # near-zero entry here and (see below) wrongly suppress the
            # single-hand reading of hands[0].
            if tip_gap < hcfg["tipGapHi"]:
                scores["twoFingersTogether"] = 1.0 - ramp(tip_gap, hcfg["tipGapLo"], hcfg["tipGapHi"])

        if face_is_fresh:
            mouth_center, face_width = last_face[0], last_face[1]
            near0 = 1.0 - ramp(dist2(h0["palmCenterImg"], mouth_center) / face_width, hcfg["nearFaceLo"], hcfg["nearFaceHi"])
            near1 = 1.0 - ramp(dist2(h1["palmCenterImg"], mouth_center) / face_width, hcfg["nearFaceLo"], hcfg["nearFaceHi"])
            near_both = min(near0, near1)
            if near_both > 0:
                head_top_y = mouth_center[1] - face_width * hcfg["headTopMargin"]
                soft = hcfg["headTopSoftness"]
                above0 = ramp((head_top_y - h0["palmCenterImg"][1]) / face_width, -soft, soft)
                above1 = ramp((head_top_y - h1["palmCenterImg"][1]) / face_width, -soft, soft)
                above_both = min(above0, above1)
                scores["twoHandsOnHead"] = min(near_both, above_both)
                scores["crashOutCat"] = min(near_both, 1.0 - above_both)

        if scores:
            # a two-hand-specific gesture is genuinely plausible here - this
            # is a decision about the PAIR and shouldn't be diluted by also
            # scoring hands[0] alone, which will often independently look
            # like a plausible single-hand gesture too (both hands doing
            # "uwu" each individually look exactly like oneFingerUp). The
            # original cascade gave the two-hand checks unconditional
            # priority over the single-hand cascade whenever they applied
            # at all; this preserves that rather than letting the two
            # readings tie into "ambiguous".
            return scores

    # single-hand shapes are evaluated off hands[0] - reached with two hands
    # present only when neither two-hand check above found anything
    # plausible, matching the original cascade's fallthrough.
    h = hands[0]

    if h["curledCount"] == 4:
        # thumb stuck out from an otherwise-curled fist = thumbs up, rather
        # than a plain fist/punch. Which one wins is now a real contest
        # between "how tucked is the thumb" and "how tucked-and-pointing-up
        # is the thumb" - so a thumb spread out but pointing DOWN scores low
        # on both and doesn't get misread as thumbsUp.
        scores["fist"] = 1.0 - h["thumbSpreadScore"]
        scores["thumbsUp"] = min(h["thumbSpreadScore"], h["thumbUpScore"])
    elif h["thumbOut"] and h["pinkyUp"] and not h["indexUp"] and not h["middleUp"] and not h["ringUp"]:
        scores["rockstar"] = 1.0
    elif is_pointing(h):
        # shhh / one-finger-up ("professor cat"): the same hand shape (only
        # the index finger extended), split by three continuous features
        # instead of one hard-cliff distance check - see gesture_config.json.
        tip, pip = h["indexTipImg"], h["indexPipImg"]
        pointing_up_score = 1.0 - ramp(angle_from_vertical_deg(tip - pip), hcfg["pointUpAngleLoDeg"], hcfg["pointUpAngleHiDeg"])

        if face_is_fresh:
            mouth_center, face_width = last_face[0], last_face[1]
            d = dist2(tip, mouth_center) / face_width
            vert_offset = (mouth_center[1] - tip[1]) / face_width  # >0 = tip above the mouth

            near_mouth = 1.0 - ramp(d, hcfg["shhhDistLo"], hcfg["shhhDistHi"])
            at_mouth_level = 1.0 - ramp(abs(vert_offset), hcfg["shhhVertOffsetLo"], hcfg["shhhVertOffsetHi"])
            scores["shhh"] = min(near_mouth, at_mouth_level, pointing_up_score)

            above_mouth = ramp(vert_offset, hcfg["shhhVertOffsetLo"], hcfg["shhhVertOffsetHi"])
            away_from_face = ramp(d, hcfg["shhhDistLo"], hcfg["shhhDistHi"])
            scores["oneFingerUp"] = min(max(above_mouth, away_from_face), pointing_up_score)
        else:
            # no face to compare against at all - can't possibly be shhh
            # (which is defined by proximity to the mouth), so this is
            # professor cat as long as the finger is actually pointing up
            # rather than sideways or down (the orientation test the
            # original had no version of at all).
            scores["oneFingerUp"] = pointing_up_score
    else:
        # hand covering face: the one hand we see sits roughly where the
        # face last was. Wider tolerance if the face detector has fully
        # lost the face (strong evidence of a real occlusion); tighter if
        # it's still partially tracking through the fingers.
        close_to_face = 0.0
        if face_is_fresh:
            mouth_center, face_width = last_face[0], last_face[1]
            d = dist2(h["palmCenterImg"], mouth_center) / face_width
            threshold = hcfg["handCoverFaceDistFaceLost"] if not face_seen_this_frame else hcfg["handCoverFaceDistFaceSeen"]
            close_to_face = 1.0 - ramp(d, threshold - hcfg["handCoverFaceSoftness"], threshold)
            scores["handCoverFace"] = close_to_face

        if h["curledCount"] == 0:
            # open palm - full credit when there's no face to be near at
            # all, tapering off the closer it gets to the face, so an open
            # palm held directly over the face reads as handCoverFace
            # rather than tying with it (the original checked
            # handCoverFace strictly before handStretchedOut, i.e. "held
            # OUT" specifically meant "not near the face").
            scores["handStretchedOut"] = 1.0 - close_to_face

    return scores


def pick_gesture(scores, face_is_fresh, yaw_deg, cfg):
    """Argmax the hand-gesture scores, requiring the winner to clear both a
    minimum confidence floor and a margin over the runner-up. A real tie
    (margin not cleared) comes back as "ambiguous" - a non-vote in the
    caller's smoother, so a single boundary-straddling frame holds the
    current gesture instead of snapping to whichever side it landed on. A
    genuinely shapeless hand (nothing clears the floor) falls back to
    side-eye or default exactly as before."""
    scfg = cfg["smoothing"]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0] if ranked else (None, 0.0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_name is not None and best_score >= scfg["minScoreFloor"]:
        if best_score - second_score >= scfg["scoreMargin"]:
            return best_name
        return "ambiguous"

    if face_is_fresh and abs(yaw_deg) > cfg["face"]["sideEyeYawDeg"]:
        return "sideEyeCat"
    return "default"


class GestureState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_face = None  # (mouth_center, face_width, mouth_open, yaw_deg, t)
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0
        self.last_mouth_open_debug = 0.0
        self.flow_history = []  # [(t, magnitude), ...] trailing samples, for the fraction-above trigger
        self.flow_peak_history = []  # [(t, score), ...] longer trailing window, for the readable peak display
        self.last_flow_magnitude_debug = 0.0
        self.last_flow_coherence_debug = 0.0
        self.last_flow_score_debug = 0.0
        self.last_flow_peak_debug = 0.0
        self.last_flow_fraction_debug = 0.0
        self.hand_finger_state = {}  # handedness label -> per-finger hysteresis state
        self.last_hand_scores_debug = {}  # gesture -> score, from the most recent frame with hands

    def update_flow(self, magnitude, coherence):
        scfg = self.cfg["spin"]
        now = time.time() * 1000
        score = magnitude * coherence  # kept for the debug HUD/log only, not the trigger

        self.flow_history.append((now, magnitude))
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < scfg["fractionWindowMs"]]

        self.flow_peak_history.append((now, score))
        self.flow_peak_history = [
            (t, s) for t, s in self.flow_peak_history if now - t < scfg["peakHoldMs"]
        ]

        self.last_flow_magnitude_debug = magnitude
        self.last_flow_coherence_debug = coherence
        self.last_flow_score_debug = score
        self.last_flow_peak_debug = max((s for _, s in self.flow_peak_history), default=0.0)
        elevated = sum(1 for _, m in self.flow_history if m > scfg["magThreshold"])
        self.last_flow_fraction_debug = elevated / len(self.flow_history) if self.flow_history else 0.0

    def is_spinning(self, now):
        scfg = self.cfg["spin"]
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < scfg["fractionWindowMs"]]
        if not self.flow_history:
            return False
        elevated = sum(1 for _, m in self.flow_history if m > scfg["magThreshold"])
        fraction = elevated / len(self.flow_history)
        return fraction > scfg["fractionRequired"]

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / face_width

            yaw_deg = 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg = yaw_from_transform_matrix(face_result.facial_transformation_matrixes[0])

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, now)
            self.last_yaw_debug = yaw_deg
            self.last_mouth_open_debug = mouth_open
        self.face_seen_this_frame = saw_face

    def _evict_stale_hand_state(self, now):
        stale = [
            key for key, state in self.hand_finger_state.items()
            if now - state.get("last_seen", 0) > self.cfg["smoothing"]["handStaleMs"]
        ]
        for key in stale:
            del self.hand_finger_state[key]

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face[4] < self.cfg["smoothing"]["faceStaleMs"]

        # spinning in the chair beats everything else, hands included.
        if self.is_spinning(now):
            return "spinCat"

        if not hand_result.hand_landmarks:
            # no hands: tongue-out and side-eye are both face-only poses.
            # Tongue-out is the more deliberate shape, so it's checked first.
            self.last_hand_scores_debug = {}
            if face_is_fresh and self.last_face[2] > self.cfg["face"]["tongueOutMouthOpenRatio"]:
                return "tongueOut"
            if face_is_fresh and abs(self.last_face[3]) > self.cfg["face"]["sideEyeYawDeg"]:
                return "sideEyeCat"
            return "default"

        self._evict_stale_hand_state(now)
        keys = hand_state_keys(hand_result)
        hands = [
            classify_hand(img_lm, world_lm, key, self.hand_finger_state, now, self.cfg)
            for img_lm, world_lm, key in zip(hand_result.hand_landmarks, hand_result.hand_world_landmarks, keys)
        ]

        scores = compute_hand_scores(hands, face_is_fresh, self.last_face, self.face_seen_this_frame, self.cfg)
        self.last_hand_scores_debug = scores
        yaw_deg = self.last_face[3] if face_is_fresh else 0.0
        return pick_gesture(scores, face_is_fresh, yaw_deg, self.cfg)


def pick_camera_index():
    """Find the AVFoundation device index of a real webcam, skipping the
    iPhone Continuity Camera and our own OBS Virtual Camera output - both of
    which show up as ordinary video capture devices on macOS and can outrank
    the actual webcam depending on what's nearby/running. Falls back to
    index 0 (OpenCV's default) if pyobjc isn't installed or nothing else
    looks like a real camera.

    The index returned lines up with cv2.VideoCapture(index,
    cv2.CAP_AVFOUNDATION) because both pyobjc and OpenCV's AVFoundation
    backend enumerate the same underlying AVCaptureDevice list, in the same
    order."""
    if AVFoundation is None:
        return 0

    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    for i, d in enumerate(devices):
        if d.isContinuityCamera():
            continue
        if d.manufacturer() == "OBS Project":
            continue
        print(f"Using camera [{i}]: {d.localizedName()}")
        return i

    print("No non-Continuity-Camera webcam found via AVFoundation, defaulting to index 0")
    return 0


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        if gesture in VIDEO_GESTURES:
            # videos are streamed frame-by-frame in the main loop instead
            continue
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {MEMES / name}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    cfg = state.cfg
    lines = [
        f"gesture: {gesture}",
        f"yaw: {state.last_yaw_debug:+.1f} deg  (side-eye thr +/-{cfg['face']['sideEyeYawDeg']:.1f})",
        f"mouthOpen: {state.last_mouth_open_debug:.2f}  (tongue-out thr {cfg['face']['tongueOutMouthOpenRatio']:.2f})",
        f"flow mag: {state.last_flow_magnitude_debug:.2f}  (thr {cfg['spin']['magThreshold']:.2f})",
        f"spin fraction (2.2s window): {state.last_flow_fraction_debug:.2f}  (thr {cfg['spin']['fractionRequired']:.2f})",
        f"peak score (last 2s): {state.last_flow_peak_debug:.2f}  <- read this AFTER you stop spinning",
    ]

    # scored-gesture readout: the top two candidates and the margin between
    # them, so a chattery boundary (e.g. shhh vs oneFingerUp) can be tuned
    # by watching these numbers instead of guessing from the meme flicker.
    ranked = sorted(state.last_hand_scores_debug.items(), key=lambda kv: kv[1], reverse=True)
    if ranked:
        top_name, top_score = ranked[0]
        second_name, second_score = ranked[1] if len(ranked) > 1 else ("-", 0.0)
        margin = top_score - second_score
        lines.append(
            f"top: {top_name}={top_score:.2f}  2nd: {second_name}={second_score:.2f}  "
            f"margin={margin:.2f} (thr {cfg['smoothing']['scoreMargin']:.2f}, floor {cfg['smoothing']['minScoreFloor']:.2f})"
        )

    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def render_to_canvas(img, width, height):
    """Compose `img` onto a width x height canvas: a blurred backdrop
    cropped to fill the frame (so there are no black bars), with the whole
    image layered on top scaled to fit without cropping. This is what the
    virtual camera actually publishes, since meme images/frames come in a
    mix of aspect ratios but the camera's own resolution is fixed."""
    h, w = img.shape[:2]

    cover_scale = max(width / w, height / h)
    cover = cv2.resize(img, (round(w * cover_scale), round(h * cover_scale)))
    cx = (cover.shape[1] - width) // 2
    cy = (cover.shape[0] - height) // 2
    backdrop = cover[cy:cy + height, cx:cx + width]
    # cheap blur: shrink way down then scale back up, instead of a
    # large-sigma GaussianBlur at full resolution every frame.
    backdrop = cv2.resize(backdrop, (32, 18), interpolation=cv2.INTER_LINEAR)
    backdrop = cv2.resize(backdrop, (width, height), interpolation=cv2.INTER_LINEAR)

    contain_scale = min(width / w, height / h)
    fw, fh = round(w * contain_scale), round(h * contain_scale)
    fg = cv2.resize(img, (fw, fh))
    ox, oy = (width - fw) // 2, (height - fh) // 2

    canvas = backdrop
    canvas[oy:oy + fh, ox:ox + fw] = fg
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--virtual-cam",
        action="store_true",
        help="also publish the meme feed as a virtual camera (via OBS) for use in Zoom/Meet/Teams",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="force a specific AVFoundation camera index instead of auto-picking one "
        "(auto-pick skips the iPhone Continuity Camera and the OBS Virtual Camera output)",
    )
    args = parser.parse_args()

    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()

    # every frame's flow numbers get logged here, timestamped - so we can
    # look at exactly what a real, full-effort spin looked like afterward
    # instead of trying to read a jittery number while dizzy.
    flow_log_path = ROOT / "flow_debug_log.csv"
    flow_log = open(flow_log_path, "w", buffering=1)  # line-buffered so data survives a hard kill
    flow_log.write(
        "t_ms,magnitude,coherence,score,fraction,peak_2s,gesture,"
        "top_gesture,top_score,second_gesture,second_score,margin\n"
    )

    spin_video_cap = cv2.VideoCapture(str(MEMES / GESTURE_MEMES["spinCat"][0]))
    if not spin_video_cap.isOpened():
        raise FileNotFoundError(f"missing meme file: {MEMES / GESTURE_MEMES['spinCat'][0]}")

    def next_spin_frame():
        ok, vframe = spin_video_cap.read()
        if not ok:
            spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, vframe = spin_video_cap.read()
        return vframe

    camera_index = args.camera_index if args.camera_index is not None else pick_camera_index()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {camera_index})")

    cam = None
    if args.virtual_cam:
        if pyvirtualcam is None:
            raise RuntimeError(
                "pyvirtualcam is not installed - run: pip install -r requirements.txt"
            )
        try:
            cam = pyvirtualcam.Camera(
                VIRTUAL_CAM_WIDTH, VIRTUAL_CAM_HEIGHT, VIRTUAL_CAM_FPS, fmt=PixelFormat.BGR
            )
        except RuntimeError as e:
            raise RuntimeError(
                "Could not start the virtual camera. Make sure OBS Studio 30+ is "
                "installed and you've clicked Start Virtual Camera, then Stop "
                f"Virtual Camera, in OBS at least once. (underlying error: {e})"
            ) from e
        print(f"Virtual camera running: {cam.device}")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState(CFG)
    current_gesture = "default"
    votes = {}  # gesture -> decaying vote weight, replaces the old N-consecutive-frames streak
    last_vote_update_at = time.time() * 1000
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])
    prev_flow_gray = None
    canvas_cache_key = None
    canvas_cache = None

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, like a selfie cam

            magnitude, coherence, prev_flow_gray = frame_flow_signal(frame, prev_flow_gray, CFG["spin"])
            state.update_flow(magnitude, coherence)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            top_gesture, top_score, second_gesture, second_score = "-", 0.0, "-", 0.0
            ranked_debug = sorted(state.last_hand_scores_debug.items(), key=lambda kv: kv[1], reverse=True)
            if ranked_debug:
                top_gesture, top_score = ranked_debug[0]
                if len(ranked_debug) > 1:
                    second_gesture, second_score = ranked_debug[1]
            flow_log.write(
                f"{time.time() * 1000:.0f},{magnitude:.4f},{coherence:.4f},"
                f"{state.last_flow_score_debug:.4f},{state.last_flow_fraction_debug:.4f},"
                f"{state.last_flow_peak_debug:.4f},{gesture},"
                f"{top_gesture},{top_score:.4f},{second_gesture},{second_score:.4f},{top_score - second_score:.4f}\n"
            )

            # time-decayed vote, replacing the old "N consecutive identical
            # frames" debounce: each gesture's vote decays continuously with
            # elapsed wall-clock time (so latency doesn't drift with fps)
            # and a single dissenting frame no longer resets progress the
            # way a broken streak used to. "ambiguous" - a real tie between
            # two candidate gestures, see pick_gesture - casts no vote at
            # all, so the display holds instead of flickering toward
            # whichever side of the tie a noisy frame landed on.
            now = time.time() * 1000
            scfg = CFG["smoothing"]
            decay = math.exp(-(now - last_vote_update_at) / scfg["voteTauMs"])
            last_vote_update_at = now
            for g in list(votes.keys()):
                votes[g] *= decay
                if votes[g] < 1e-3:
                    del votes[g]
            if gesture != "ambiguous":
                votes[gesture] = votes.get(gesture, 0.0) + 1.0

            if votes:
                best_gesture, best_votes = max(votes.items(), key=lambda kv: kv[1])
            else:
                best_gesture, best_votes = current_gesture, 0.0

            if best_votes >= scfg["switchVoteThreshold"] and best_gesture != current_gesture:
                current_gesture = best_gesture
                if best_gesture not in VIDEO_GESTURES:
                    current_meme = random.choice(memes[best_gesture])
                elif best_gesture == "spinCat":
                    spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > scfg["defaultFallbackMs"] and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])
                votes.clear()

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            if current_gesture == "spinCat":
                vframe = next_spin_frame()
                source_img = vframe if vframe is not None else current_meme
                canvas = render_to_canvas(source_img, VIRTUAL_CAM_WIDTH, VIRTUAL_CAM_HEIGHT)
            else:
                cache_key = id(current_meme)
                if cache_key != canvas_cache_key:
                    canvas_cache = render_to_canvas(current_meme, VIRTUAL_CAM_WIDTH, VIRTUAL_CAM_HEIGHT)
                    canvas_cache_key = cache_key
                canvas = canvas_cache

            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", canvas)

            if cam is not None:
                cam.send(canvas)
                cam.sleep_until_next_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        spin_video_cap.release()
        flow_log.close()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()
        if cam is not None:
            cam.close()


if __name__ == "__main__":
    main()
