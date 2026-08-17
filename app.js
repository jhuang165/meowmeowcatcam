import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- meme mapping -----------------------------------------------------
// Each gesture maps to one or more meme images. When a gesture has more
// than one image, one is picked at random each time the gesture is newly
// (re)triggered, so repeated gestures don't always show the same frame.
//
// Note: this browser version has no spinCat - it has no optical flow /
// motion detection at all, unlike gesture_meme.py (desktop). That's a
// pre-existing gap, not something this pass changes.
const GESTURE_MEMES = {
  rockstar: ["memes/cat.jpg"],
  default: ["memes/pokercat.jpg"],
  oneFingerUp: ["memes/profcat.jpg", "memes/professorcat.jpg"],
  fist: ["memes/punchcat.jpg"],
  thumbsUp: ["memes/cat_thumbs_up.png"],
  shhh: ["memes/shhcat.jpg"],
  twoFingersTogether: [
    "memes/uwucat.jpg",
    "memes/uwucatt.jpg",
    "memes/fingers together muehehe .jpg",
  ],
  handCoverFace: ["memes/hand cover face .jpg"],
  crashOutCat: ["memes/crashout cat .jpg"],
  twoHandsOnHead: ["memes/two hands on head .jpg"],
  handStretchedOut: ["memes/hand stretched out, palm facing up .jpg"],
  tongueOut: ["memes/cat_tongue.jpeg"],
  sideEyeCat: ["memes/side eye cat.jpg"],
};

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let CFG; // loaded from gesture_config.json - see loadConfig()
let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let votes = {}; // gesture -> decaying vote weight (see updateVotes)
let lastVoteUpdateAt = performance.now();
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, mouthOpen, yawDeg, t }
let lastFaceSeenThisFrame = false;
let lastYawDebug = 0;
let lastMouthOpenDebug = 0;
let handFingerState = {}; // handedness label -> per-finger hysteresis state
let lastHandScoresDebug = {}; // gesture -> score, from the most recent frame with hands

async function loadConfig() {
  // Shared tunables live in gesture_config.json instead of hardcoded here,
  // so this file and gesture_meme.py (desktop) read the same numbers and
  // can't drift out of sync the way they already had (this file never
  // gained gesture_meme.py's spin detection, for instance).
  const res = await fetch("gesture_config.json");
  return res.json();
}

async function init() {
  CFG = await loadConfig();

  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- geometry helpers ---------------------------------------------------
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
// 2D (x, y only) distance. Used for every hand<->face and hand<->hand
// comparison: hand-landmarker z is relative to that hand's own wrist and
// face-landmarker z is relative to the head, so a 3D dist() across two
// different detectors' landmarks mixes two unrelated depths and adds
// noise, not signal. Full 3D dist() is still correct for measurements
// within a single hand's own landmarks (see fingerExtension below).
function dist2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

// 0 at/below lo, 1 at/above hi, linear in between. The building block for
// every soft threshold below - replaces a hard boolean cliff with a score
// that degrades gracefully near the boundary instead of flipping.
function ramp(x, lo, hi) {
  if (hi <= lo) return x >= hi ? 1 : 0;
  return Math.max(0, Math.min(1, (x - lo) / (hi - lo)));
}

// Schmitt-trigger hysteresis: flip on above hi, flip off below lo,
// otherwise hold the previous state. Turns a continuous score into a
// stable boolean that can't flap every frame just because the score is
// hovering near a single cutoff.
function latch(prev, score, hi, lo) {
  if (score >= hi) return true;
  if (score <= lo) return false;
  return prev;
}

// (mcp, pip, dip, tip) landmark indices for each finger's chain. The thumb
// only has three true segments (cmc-mcp-ip-tip); using the same 4-point
// shape as the other fingers keeps fingerExtension generic.
const FINGER_CHAINS = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

// Continuous [0,1] "how extended is this finger" score, computed from hand
// WORLD landmarks (MediaPipe's metric, perspective-corrected 3D estimate)
// rather than image landmarks. angleDeg/dist ratios are rotation-invariant,
// so this stays correct however the hand is turned - including pointed
// straight at the camera, where an image-space angle test would foreshorten
// a straight finger past a fixed cutoff and misread it as curled.
//
// Blends two cues:
//   - curl: total bend across mcp->pip->dip->tip (0 = dead straight)
//   - reach: chord (mcp->tip) over arc (sum of the three segments) - ~1.0
//     straight, ~0.5 fully curled. This ratio is what actually survives
//     foreshortening, since numerator and denominator shrink together
//     under projection - it's weighted higher than curl for that
//     robustness.
function fingerExtension(worldLm, chain, cfg) {
  const [mcp, pip, dip, tip] = chain;
  const seg1 = vec(worldLm[mcp], worldLm[pip]);
  const seg2 = vec(worldLm[pip], worldLm[dip]);
  const seg3 = vec(worldLm[dip], worldLm[tip]);
  const curl = angleDeg(seg1, seg2) + angleDeg(seg2, seg3);

  const arc = dist(worldLm[mcp], worldLm[pip]) + dist(worldLm[pip], worldLm[dip]) + dist(worldLm[dip], worldLm[tip]);
  const chord = dist(worldLm[mcp], worldLm[tip]);
  const reach = arc > 1e-9 ? chord / arc : 0;

  const angleScore = 1 - ramp(curl, cfg.curlAngleLoDeg, cfg.curlAngleHiDeg);
  const reachScore = ramp(reach, cfg.reachLo, cfg.reachHi);
  return cfg.angleWeight * angleScore + cfg.reachWeight * reachScore;
}

// Angle (degrees) between a 2D image-space vector and "straight up" (image
// y grows downward, so up is (0, -1)). Only used for direction tests
// (which way is the thumb/finger pointing) - unlike finger shape, "up" is
// inherently a screen-relative question, so this deliberately uses image
// landmarks rather than hand-world landmarks.
function angleFromVerticalDeg(v2d) {
  return angleDeg({ x: v2d.x, y: v2d.y, z: 0 }, { x: 0, y: -1, z: 0 });
}

// extract the head's left/right turn angle (yaw, degrees) from MediaPipe's
// facial transformation matrix - its own estimate of head pose, far more
// robust than trying to infer turn from landmark distances.
function yawFromTransformMatrix(matrixData) {
  // matrixData is a 16-element row-major 4x4 array; r(row, col) = data[row*4+col]
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

// A stable per-hand key for this frame, from MediaPipe's own handedness
// label, so finger hysteresis state (see classifyHand) persists across
// frames for "the same hand" rather than by array position (which can swap
// if hands cross or one drops out). Suffixed on the rare collision where
// both hands get the same label.
function handStateKeys(handResult) {
  const seen = {};
  return (handResult.handedness || []).map((cats) => {
    const label = cats && cats[0] ? cats[0].categoryName : "hand";
    const n = seen[label] || 0;
    seen[label] = n + 1;
    return n === 0 ? label : `${label}${n}`;
  });
}

function evictStaleHandState(now) {
  for (const key of Object.keys(handFingerState)) {
    if (now - (handFingerState[key].lastSeen || 0) > CFG.smoothing.handStaleMs) {
      delete handFingerState[key];
    }
  }
}

// Build the per-hand feature set decideGesture scores gestures from: a
// continuous extension score per finger (world landmarks - see
// fingerExtension), hysteresis-latched up/down booleans per finger (keyed
// by handedness label so each hand's state persists across frames
// independently), and the image-space points needed for any hand<->face or
// hand<->hand comparison.
function classifyHand(imageLm, worldLm, key, now) {
  const handScaleImg = dist2(imageLm[0], imageLm[9]) || 1e-6;
  const handScaleWorld = dist(worldLm[0], worldLm[9]) || 1e-6;

  const fcfg = CFG.finger;
  const ext = {};
  for (const name of Object.keys(FINGER_CHAINS)) {
    ext[name] = fingerExtension(worldLm, FINGER_CHAINS[name], fcfg);
  }

  const tcfg = CFG.thumb;
  const thumbSpread = dist(worldLm[4], worldLm[17]) / handScaleWorld;
  const thumbSpreadScore = ramp(thumbSpread, tcfg.spreadLo, tcfg.spreadHi);

  // thumb direction (up vs down): image-space only, see angleFromVerticalDeg.
  const thumbDir = vec(imageLm[2], imageLm[4]);
  const thumbUpAngle = angleFromVerticalDeg(thumbDir);
  const thumbUpScore = 1 - ramp(thumbUpAngle, tcfg.upAngleLoDeg, tcfg.upAngleHiDeg);

  const state =
    handFingerState[key] ||
    (handFingerState[key] = { index: false, middle: false, ring: false, pinky: false, thumb: false });
  state.lastSeen = now;
  const up = {};
  for (const name of ["index", "middle", "ring", "pinky"]) {
    up[name] = latch(state[name], ext[name], fcfg.hysteresisHigh, fcfg.hysteresisLow);
    state[name] = up[name];
  }
  state.thumb = latch(state.thumb, thumbSpreadScore, tcfg.hysteresisHigh, tcfg.hysteresisLow);

  const curledCount = ["index", "middle", "ring", "pinky"].filter((name) => !up[name]).length;

  return {
    ext,
    indexUp: up.index,
    middleUp: up.middle,
    ringUp: up.ring,
    pinkyUp: up.pinky,
    thumbOut: state.thumb,
    thumbSpreadScore,
    thumbUpScore,
    curledCount,
    handScaleImg,
    indexTipImg: imageLm[8],
    indexPipImg: imageLm[6],
    wristImg: imageLm[0],
    palmCenterImg: imageLm[9],
  };
}

// a hand is "pointing" if only the index finger is (hysteresis-latched) up.
function isPointing(h) {
  return h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    // how open the mouth is right now - normalized so it doesn't depend on
    // distance from the camera.
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    let yawDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      yawDeg = yawFromTransformMatrix(faceResult.facialTransformationMatrixes[0].data);
    }

    lastFace = { mouthCenter, faceWidth, mouthOpen, yawDeg, t: now };
    lastYawDebug = yawDeg;
    lastMouthOpenDebug = mouthOpen;
  }
  lastFaceSeenThisFrame = sawFace;
}

// Build a {gesture: score in [0,1]} dict for every hand-shape gesture the
// current hand(s) could plausibly be. The coarse routing below (which
// finger-count bucket a hand falls into) mirrors the original cascade's
// genuine precedence - those are mutually exclusive finger configurations,
// not real ties. What's new is that the decision *within* a bucket (fist vs
// thumbsUp; shhh vs oneFingerUp; twoHandsOnHead vs crashOutCat) is a
// continuous, scored comparison instead of a single hard-cliff threshold,
// so decideGesture's caller can pick the winner by margin and treat a close
// call as "ambiguous" rather than snapping to whichever side of the cliff a
// noisy frame landed on.
function computeHandScores(hands, faceIsFresh) {
  const hcfg = CFG.hand;
  const scores = {};

  if (hands.length === 2) {
    const [h0, h1] = hands;
    if (isPointing(h0) && isPointing(h1)) {
      const avgScale = (h0.handScaleImg + h1.handScaleImg) / 2;
      const tipGap = dist2(h0.indexTipImg, h1.indexTipImg) / avgScale;
      // only a candidate at all once the tips are at least roughly close
      // (matching the original's tipGap < tipGapHi cutoff for even
      // considering this gesture) - otherwise two hands each
      // independently pointing somewhere would always add a near-zero
      // entry here and (see below) wrongly suppress the single-hand
      // reading of hands[0].
      if (tipGap < hcfg.tipGapHi) {
        scores.twoFingersTogether = 1 - ramp(tipGap, hcfg.tipGapLo, hcfg.tipGapHi);
      }
    }

    if (faceIsFresh) {
      const { mouthCenter, faceWidth } = lastFace;
      const near0 = 1 - ramp(dist2(h0.palmCenterImg, mouthCenter) / faceWidth, hcfg.nearFaceLo, hcfg.nearFaceHi);
      const near1 = 1 - ramp(dist2(h1.palmCenterImg, mouthCenter) / faceWidth, hcfg.nearFaceLo, hcfg.nearFaceHi);
      const nearBoth = Math.min(near0, near1);
      if (nearBoth > 0) {
        const headTopY = mouthCenter.y - faceWidth * hcfg.headTopMargin;
        const soft = hcfg.headTopSoftness;
        const above0 = ramp((headTopY - h0.palmCenterImg.y) / faceWidth, -soft, soft);
        const above1 = ramp((headTopY - h1.palmCenterImg.y) / faceWidth, -soft, soft);
        const aboveBoth = Math.min(above0, above1);
        scores.twoHandsOnHead = Math.min(nearBoth, aboveBoth);
        scores.crashOutCat = Math.min(nearBoth, 1 - aboveBoth);
      }
    }

    if (Object.keys(scores).length > 0) {
      // a two-hand-specific gesture is genuinely plausible here - this is
      // a decision about the PAIR and shouldn't be diluted by also
      // scoring hands[0] alone, which will often independently look like
      // a plausible single-hand gesture too (both hands doing "uwu" each
      // individually look exactly like oneFingerUp). The original cascade
      // gave the two-hand checks unconditional priority over the
      // single-hand cascade whenever they applied at all; this preserves
      // that rather than letting the two readings tie into "ambiguous".
      return scores;
    }
  }

  // single-hand shapes are evaluated off hands[0] - reached with two hands
  // present only when neither two-hand check above found anything
  // plausible, matching the original cascade's fallthrough.
  const h = hands[0];

  if (h.curledCount === 4) {
    // thumb stuck out from an otherwise-curled fist = thumbs up, rather
    // than a plain fist/punch. Which one wins is now a real contest
    // between "how tucked is the thumb" and "how tucked-and-pointing-up is
    // the thumb" - so a thumb spread out but pointing DOWN scores low on
    // both and doesn't get misread as thumbsUp.
    scores.fist = 1 - h.thumbSpreadScore;
    scores.thumbsUp = Math.min(h.thumbSpreadScore, h.thumbUpScore);
  } else if (h.thumbOut && h.pinkyUp && !h.indexUp && !h.middleUp && !h.ringUp) {
    scores.rockstar = 1;
  } else if (isPointing(h)) {
    // shhh / one-finger-up ("professor cat"): the same hand shape (only
    // the index finger extended), split by three continuous features
    // instead of one hard-cliff distance check - see gesture_config.json.
    const tip = h.indexTipImg;
    const pip = h.indexPipImg;
    const pointingUpScore = 1 - ramp(angleFromVerticalDeg(vec(pip, tip)), hcfg.pointUpAngleLoDeg, hcfg.pointUpAngleHiDeg);

    if (faceIsFresh) {
      const { mouthCenter, faceWidth } = lastFace;
      const d = dist2(tip, mouthCenter) / faceWidth;
      const vertOffset = (mouthCenter.y - tip.y) / faceWidth; // >0 = tip above the mouth

      const nearMouth = 1 - ramp(d, hcfg.shhhDistLo, hcfg.shhhDistHi);
      const atMouthLevel = 1 - ramp(Math.abs(vertOffset), hcfg.shhhVertOffsetLo, hcfg.shhhVertOffsetHi);
      scores.shhh = Math.min(nearMouth, atMouthLevel, pointingUpScore);

      const aboveMouth = ramp(vertOffset, hcfg.shhhVertOffsetLo, hcfg.shhhVertOffsetHi);
      const awayFromFace = ramp(d, hcfg.shhhDistLo, hcfg.shhhDistHi);
      scores.oneFingerUp = Math.min(Math.max(aboveMouth, awayFromFace), pointingUpScore);
    } else {
      // no face to compare against at all - can't possibly be shhh (which
      // is defined by proximity to the mouth), so this is professor cat as
      // long as the finger is actually pointing up rather than sideways or
      // down (the orientation test the original had no version of at all).
      scores.oneFingerUp = pointingUpScore;
    }
  } else {
    // hand covering face: the one hand we see sits roughly where the face
    // last was. Wider tolerance if the face detector has fully lost the
    // face (strong evidence of a real occlusion); tighter if it's still
    // partially tracking through the fingers.
    let closeToFace = 0;
    if (faceIsFresh) {
      const { mouthCenter, faceWidth } = lastFace;
      const d = dist2(h.palmCenterImg, mouthCenter) / faceWidth;
      const threshold = lastFaceSeenThisFrame ? hcfg.handCoverFaceDistFaceSeen : hcfg.handCoverFaceDistFaceLost;
      closeToFace = 1 - ramp(d, threshold - hcfg.handCoverFaceSoftness, threshold);
      scores.handCoverFace = closeToFace;
    }

    if (h.curledCount === 0) {
      // open palm - full credit when there's no face to be near at all,
      // tapering off the closer it gets to the face, so an open palm held
      // directly over the face reads as handCoverFace rather than tying
      // with it (the original checked handCoverFace strictly before
      // handStretchedOut, i.e. "held OUT" specifically meant "not near
      // the face").
      scores.handStretchedOut = 1 - closeToFace;
    }
  }

  return scores;
}

// Argmax the hand-gesture scores, requiring the winner to clear both a
// minimum confidence floor and a margin over the runner-up. A real tie
// (margin not cleared) comes back as "ambiguous" - a non-vote in the
// caller's smoother, so a single boundary-straddling frame holds the
// current gesture instead of snapping to whichever side it landed on. A
// genuinely shapeless hand (nothing clears the floor) falls back to
// side-eye or default exactly as before.
function pickGesture(scores, faceIsFresh, yawDeg) {
  const scfg = CFG.smoothing;
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [bestName, bestScore] = ranked[0] || [null, 0];
  const secondScore = ranked.length > 1 ? ranked[1][1] : 0;

  if (bestName !== null && bestScore >= scfg.minScoreFloor) {
    if (bestScore - secondScore >= scfg.scoreMargin) return bestName;
    return "ambiguous";
  }

  if (faceIsFresh && Math.abs(yawDeg) > CFG.face.sideEyeYawDeg) return "sideEyeCat";
  return "default";
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < CFG.smoothing.faceStaleMs;

  if (!handResult.landmarks || handResult.landmarks.length === 0) {
    // no hands: tongue-out and side-eye are both face-only poses. Tongue-out
    // is the more deliberate shape, so it's checked first.
    lastHandScoresDebug = {};
    if (faceIsFresh && lastFace.mouthOpen > CFG.face.tongueOutMouthOpenRatio) {
      return "tongueOut";
    }
    if (faceIsFresh && Math.abs(lastFace.yawDeg) > CFG.face.sideEyeYawDeg) {
      return "sideEyeCat";
    }
    return "default";
  }

  evictStaleHandState(now);
  const keys = handStateKeys(handResult);
  const hands = handResult.landmarks.map((lm, i) => classifyHand(lm, handResult.worldLandmarks[i], keys[i], now));

  const scores = computeHandScores(hands, faceIsFresh);
  lastHandScoresDebug = scores;
  const yawDeg = faceIsFresh ? lastFace.yawDeg : 0;
  return pickGesture(scores, faceIsFresh, yawDeg);
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

// Time-decayed vote, replacing the old "N consecutive identical frames"
// debounce: each gesture's vote decays continuously with elapsed wall-clock
// time (so latency doesn't drift with frame rate) and a single dissenting
// frame no longer resets progress the way a broken streak used to.
// "ambiguous" - a real tie between two candidate gestures, see pickGesture -
// casts no vote at all, so the display holds instead of flickering toward
// whichever side of the tie a noisy frame landed on.
function updateVotes(gesture, now) {
  const scfg = CFG.smoothing;
  const decay = Math.exp(-(now - lastVoteUpdateAt) / scfg.voteTauMs);
  lastVoteUpdateAt = now;
  for (const g of Object.keys(votes)) {
    votes[g] *= decay;
    if (votes[g] < 1e-3) delete votes[g];
  }
  if (gesture !== "ambiguous") {
    votes[gesture] = (votes[gesture] || 0) + 1;
  }

  let bestGesture = currentGesture;
  let bestVotes = 0;
  for (const [g, v] of Object.entries(votes)) {
    if (v > bestVotes) {
      bestGesture = g;
      bestVotes = v;
    }
  }

  if (bestVotes >= scfg.switchVoteThreshold && bestGesture !== currentGesture) {
    applyGesture(bestGesture);
  }
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);
    updateVotes(gesture, now);

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > CFG.smoothing.defaultFallbackMs && currentGesture !== "default") {
      applyGesture("default");
      votes = {};
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  const lines = [
    `gesture: ${currentGesture}`,
    `yaw: ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)} deg  (side-eye thr +/-${CFG.face.sideEyeYawDeg.toFixed(1)})`,
    `mouthOpen: ${lastMouthOpenDebug.toFixed(2)}  (tongue-out thr ${CFG.face.tongueOutMouthOpenRatio.toFixed(2)})`,
  ];

  // scored-gesture readout: the top two candidates and the margin between
  // them, so a chattery boundary (e.g. shhh vs oneFingerUp) can be tuned by
  // watching these numbers instead of guessing from the meme flicker.
  const ranked = Object.entries(lastHandScoresDebug).sort((a, b) => b[1] - a[1]);
  if (ranked.length > 0) {
    const [topName, topScore] = ranked[0];
    const [secondName, secondScore] = ranked[1] || ["-", 0];
    lines.push(
      `top: ${topName}=${topScore.toFixed(2)}  2nd: ${secondName}=${secondScore.toFixed(2)}  ` +
        `margin=${(topScore - secondScore).toFixed(2)} (thr ${CFG.smoothing.scoreMargin.toFixed(2)}, floor ${CFG.smoothing.minScoreFloor.toFixed(2)})`
    );
  }

  debugHud.textContent = lines.join("\n");
}

init().catch((err) => console.error(err));
