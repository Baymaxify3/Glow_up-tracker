# ─────────────────────────────────────────────────────────────────────
#  Posture Glow-Up Tracker  –  app.py
#  Phase 1 : Analyse posture from a side-view photo
#  Phase 2 : Personalised glow-up plan based on body stats + nutrients
#
#  Uses mediapipe.tasks.vision.PoseLandmarker (Tasks API).
#  Required model file: pose_landmarker_full.task  (must sit next to app.py)
# ─────────────────────────────────────────────────────────────────────

import math
import datetime
import os
import io

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (MUST be the very first Streamlit call) ──────────────
st.set_page_config(
    page_title="Posture Glow-Up Tracker",
    page_icon="🌟",
    layout="centered",
)

# ── MediaPipe Tasks API shortcuts ─────────────────────────────────────
# mediapipe 0.10.30+ on Python 3.13 Windows ships only the Tasks API.
# mp.solutions is gone; we use mp.tasks.vision instead.
BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
RunningMode        = mp.tasks.vision.RunningMode
PoseLandmark       = mp.tasks.vision.PoseLandmark   # enum of landmark indices

# Indices of the six landmarks we care about
LEFT_EAR       = PoseLandmark.LEFT_EAR.value        # 7
RIGHT_EAR      = PoseLandmark.RIGHT_EAR.value       # 8
LEFT_SHOULDER  = PoseLandmark.LEFT_SHOULDER.value   # 11
RIGHT_SHOULDER = PoseLandmark.RIGHT_SHOULDER.value  # 12
LEFT_HIP       = PoseLandmark.LEFT_HIP.value        # 23
RIGHT_HIP      = PoseLandmark.RIGHT_HIP.value       # 24
LEFT_KNEE      = PoseLandmark.LEFT_KNEE.value       # 25
RIGHT_KNEE     = PoseLandmark.RIGHT_KNEE.value      # 26
LEFT_ANKLE     = PoseLandmark.LEFT_ANKLE.value      # 27
RIGHT_ANKLE    = PoseLandmark.RIGHT_ANKLE.value     # 28

# Path to the downloaded model file (must be in the same folder as app.py)
TASK_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_full.task")

# ══════════════════════════════════════════════════════════════════════
#  THRESHOLDS  –  edit here to tune the model
#
#  neck_angle = forward_deviation of ear from directly above shoulder
#    (= 180 - angle at shoulder between neck-line and spine-line)
#    0°  = perfect (ear above shoulder)
#    30° = mild forward head
#    60° = severe forward head
#
#  torso_lean = degrees spine deviates from vertical
#    0-10° = upright, 10-20° = mild lean, >20° = significant lean
# ══════════════════════════════════════════════════════════════════════
THRESHOLDS = {
    # neck_angle : forward deviation of ear from directly above shoulder
    "neck_angle": {
        "good": 15,    # ≤ 15° → green  (good head position)
        "mild": 30,    # 15-30°→ orange (mild forward head)
        "weight": 35,  # > 30° → red    (severe forward head)
    },
    # thoracic_kyphosis : posterior back hump depth as % of torso length
    "thoracic_kyphosis": {
        "good": 28,    # ≤ 28% → green  (normal thoracic curvature)
        "mild": 40,    # 28-40%→ orange (mild rounding / early hump)
        "weight": 35,  # > 40% → red    (pronounced back hump / hyperkyphosis)
    },
    # torso_lean : degrees the shoulder→hip line tilts from vertical
    "torso_lean_standing": {
        "good": 8,     # ≤ 8°  → green  (upright)
        "mild": 18,    # 8-18° → orange (noticeable lean)
        "weight": 30,  # > 18° → red    (significant lean)
    },
    "torso_lean_sitting": {
        "good": 12,    # ≤ 12° → green  (upright sit)
        "mild": 22,    # 12-22°→ orange
        "weight": 30,  # > 22° → red
    },
}

VERDICT_GOOD     = 70    # score ≥ 70  → GOOD POSTURE
VERDICT_MID      = 40    # score ≥ 40  → NEEDS IMPROVEMENT   else POOR
VIS_THRESHOLD    = 0.6   # landmark visibility below this → low confidence
CHANGE_THRESHOLD = 2     # score-point diff = real progress

LOG_FILE_CSV       = "posture_log.csv"
LOG_FILE_XLSX      = "posture_log.xlsx"
NUTRITION_LOG_CSV  = "nutrition_log.csv"
NUTRITION_LOG_XLSX = "nutrition_log.xlsx"

LOG_FILE      = LOG_FILE_CSV
NUTRITION_LOG = NUTRITION_LOG_CSV


# ── Colour maps ───────────────────────────────────────────────────────
BAND_BGR   = {"good": (0, 200, 0), "mild": (0, 165, 255), "poor": (0, 0, 220)}
BAND_EMOJI = {"good": "🟢",        "mild": "🟠",           "poor": "🔴"}
VERDICT_BGR = {
    "GOOD POSTURE":               (0, 200, 0),
    "NEEDS IMPROVEMENT":          (0, 165, 255),
    "POOR POSTURE":               (0, 0, 220),
    "POOR POSTURE (DORSAL HUMP)": (0, 0, 220),
}


# ══════════════════════════════════════════════════════════════════════
#  IMAGE PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════

def pad_to_square(img_bgr):
    """
    Pad an image to a square with white borders, keeping the person
    centred.  Returns (padded_bgr, pad_top, pad_left).

    Why square?  MediaPipe Pose works best on square inputs; a tall
    narrow photo often causes the hip to be cropped, giving low-
    visibility landmarks.
    """
    h, w = img_bgr.shape[:2]
    size = max(h, w)

    pad_top    = (size - h) // 2
    pad_left   = (size - w) // 2
    pad_bottom = size - h - pad_top
    pad_right  = size - w - pad_left

    # Pad with white (255)
    padded = np.pad(
        img_bgr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=255,
    )
    return padded, pad_top, pad_left


def lm_to_padded_px(lm, pad_size):
    """
    Convert a normalised (0-1) MediaPipe landmark to pixel coords
    inside the padded square image.
    """
    return (int(lm.x * pad_size), int(lm.y * pad_size))


# ══════════════════════════════════════════════════════════════════════
#  TASKS API HELPERS
# ══════════════════════════════════════════════════════════════════════

def run_pose(padded_rgb):
    """
    Run PoseLandmarker on a padded RGB image.
    Returns the list of landmarks for the first detected person, or None.

    The Tasks API uses mp.Image (not numpy directly).
    Running mode IMAGE = single photo (not video/webcam).
    """
    if not os.path.exists(TASK_MODEL):
        st.error(
            f"Model file not found: `{TASK_MODEL}`\n\n"
            "Download it with:\n"
            "```\n"
            "Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task "
            "-OutFile pose_landmarker_full.task\n"
            "```\n"
            "Run that command inside the `posture_project` folder."
        )
        st.stop()

    options = PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=TASK_MODEL),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=padded_rgb,          # uint8 RGB numpy array
        )
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None               # no person found

    # result.pose_landmarks is a list-of-lists (one list per person).
    # We asked for num_poses=1, so take index 0.
    return result.pose_landmarks[0]   # list of 33 NormalizedLandmark


def _vis(lm):
    """Return confidence score combining visibility and presence safely."""
    v = getattr(lm, "visibility", None)
    p = getattr(lm, "presence", None)
    if v is not None and p is not None:
        return float(max(v, p))
    if v is not None:
        return float(v)
    if p is not None:
        return float(p)
    return 0.0


def _pres(lm):
    """Alias for backward compatibility."""
    return _vis(lm)


def pick_side(landmarks):
    """
    Accurately choose the camera-facing side for a side-view photo.

    Combines 3 anatomical & computer vision signals:
    1. Facing direction geometry:
       • If nose.x < mid_shoulder.x (facing LEFT)  → RIGHT side faces camera.
       • If nose.x >= mid_shoulder.x (facing RIGHT) → LEFT side faces camera.
    2. MediaPipe 3D Depth (z):
       • Negative z is closer to camera.
    3. Landmark Visibility / Presence:
       • Visible side has higher confidence than the occluded side.

    Returns (ear_lm, shoulder_lm, hip_lm, knee_lm, ankle_lm, side_str, facing_direction).
    """
    NOSE = 0   # MediaPipe landmark index for nose

    nose       = landmarks[NOSE]
    l_sho      = landmarks[LEFT_SHOULDER]
    r_sho      = landmarks[RIGHT_SHOULDER]
    mid_sho_x  = (l_sho.x + r_sho.x) / 2.0

    # Facing direction: where the nose is pointing relative to the shoulder midline
    facing_direction = "left" if nose.x < mid_sho_x else "right"

    # 1. Geometry signal:
    # When facing LEFT, the person's RIGHT shoulder/arm faces the camera.
    # When facing RIGHT, the person's LEFT shoulder/arm faces the camera.
    geo_left  = 1.0 if facing_direction == "right" else 0.0
    geo_right = 1.0 if facing_direction == "left" else 0.0

    # 2. 3D depth signal (smaller z = closer to camera)
    l_z = getattr(l_sho, "z", 0.0) or 0.0
    r_z = getattr(r_sho, "z", 0.0) or 0.0
    z_left  = 1.0 if l_z < r_z else 0.0
    z_right = 1.0 if r_z < l_z else 0.0

    # 3. Visibility / confidence signal
    vis_left  = (_vis(landmarks[LEFT_EAR]) + _vis(landmarks[LEFT_SHOULDER]) + _vis(landmarks[LEFT_HIP]))
    vis_right = (_vis(landmarks[RIGHT_EAR]) + _vis(landmarks[RIGHT_SHOULDER]) + _vis(landmarks[RIGHT_HIP]))
    total_vis = vis_left + vis_right + 1e-6
    vis_left_n  = vis_left / total_vis
    vis_right_n = vis_right / total_vis

    # Weighted combined score
    score_left  = 0.40 * geo_left  + 0.35 * z_left  + 0.25 * vis_left_n
    score_right = 0.40 * geo_right + 0.35 * z_right + 0.25 * vis_right_n

    if score_left >= score_right:
        side_str = "left"
        return (landmarks[LEFT_EAR], landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP],
                landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE], side_str, facing_direction)
    else:
        side_str = "right"
        return (landmarks[RIGHT_EAR], landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP],
                landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE], side_str, facing_direction)


# ══════════════════════════════════════════════════════════════════════
#  POSTURE MATHS
#
#  Two clinically-grounded measurements from a side-view photo:
#
#  1. NECK ANGLE  (ear → shoulder → hip)
#     The angle at the shoulder between the neck line (shoulder→ear)
#     and the spine line (shoulder→hip).
#     • 150–180° = ear stacked above shoulder = GOOD (near straight)
#     • 120–150° = mild forward head
#     •  < 120°  = severe forward head posture
#     We convert to "forward deviation" = 180 - angle, so 0° = perfect.
#
#  2. TORSO LEAN  (shoulder→hip from vertical)
#     Degrees the spine line deviates from vertical.
#     • 0–10°  = upright
#     • 10–20° = mild lean
#     •  > 20° = significant lean
# ══════════════════════════════════════════════════════════════════════

def seg_length_norm(a_lm, b_lm):
    """Euclidean distance in normalised (0-1) coord space."""
    dx = a_lm.x - b_lm.x
    dy = a_lm.y - b_lm.y
    return max(math.sqrt(dx * dx + dy * dy), 0.001)


def calc_neck_angle(ear_lm, shoulder_lm, hip_lm):
    """
    Angle at shoulder between the neck (shoulder→ear) and spine (shoulder→hip).
    Returns forward_deviation = 180 - angle_degrees.
      0°   = ear directly above shoulder = perfect posture
      30°  = mild forward head
      60°+ = severe forward head
    """
    # Vector from shoulder to ear
    nex = ear_lm.x - shoulder_lm.x
    ney = ear_lm.y - shoulder_lm.y   # y increases downward
    # Vector from shoulder to hip
    spx = hip_lm.x - shoulder_lm.x
    spy = hip_lm.y - shoulder_lm.y

    # Angle between the two vectors via dot product
    dot      = nex * spx + ney * spy
    mag_neck = math.sqrt(nex**2 + ney**2) + 1e-9
    mag_sp   = math.sqrt(spx**2 + spy**2) + 1e-9
    cos_a    = max(-1.0, min(1.0, dot / (mag_neck * mag_sp)))
    angle    = math.degrees(math.acos(cos_a))   # 0–180°

    # forward_deviation: 0 = perfect, higher = worse
    return round(180.0 - angle, 1)


def calc_torso_lean(shoulder_lm, hip_lm):
    """Degrees the shoulder→hip line tilts from vertical. 0° = upright."""
    dx = shoulder_lm.x - hip_lm.x
    dy = shoulder_lm.y - hip_lm.y   # y increases downward in image space
    return round(math.degrees(math.atan2(abs(dx), abs(dy))), 1)


def get_lean_direction(shoulder_lm, hip_lm, facing_direction, angle):
    """
    Determines whether torso lean is Forward Slouch, Swayback (backward lean), or Upright.
    """
    if angle <= 3.0:
        return "Upright"
    dx = shoulder_lm.x - hip_lm.x
    # If facing left: shoulder.x < hip.x means forward slouch
    # If facing right: shoulder.x > hip.x means forward slouch
    is_forward = (facing_direction == "left" and dx < 0) or (facing_direction == "right" and dx > 0)
    return "Forward Slouch" if is_forward else "Swayback (Lean Back)"


def calc_cervical_strain(neck_angle):
    """
    Clinically-grounded cervical spine strain model based on Hansraj (2014)
    'Assessment of Stresses in the Cervical Spine Caused by Posture and Position of the Head'.
    Normal upright head: ~5.0 kg (~11 lbs).
    At 15°: ~12.0 kg.
    At 30°: ~18.0 kg.
    At 45°: ~22.0 kg.
    At 60°: ~27.0 kg.
    """
    if neck_angle <= 0:
        return 5.0
    strain = 5.0 + 22.0 * min(neck_angle / 60.0, 1.25)
    return round(strain, 1)


def band(value, cfg):
    """Classify value as 'good', 'mild', or 'poor' (higher = worse)."""
    if value <= cfg["good"]: return "good"
    if value <= cfg["mild"]: return "mild"
    return "poor"


def band_score(b, weight, value, cfg):
    """
    Continuous score within each band — smoother than hard 100/50/0 steps.

    good zone  : score scales from weight  down to weight*0.6
                 as value goes from 0 → good threshold.
    mild zone  : score scales from weight*0.6 down to weight*0.15
                 as value goes from good → mild threshold.
    poor zone  : score scales from weight*0.15 down to 0
                 as value goes from mild → mild*2 (cap at 0).

    This way a value just over the 'good' threshold loses only a few points
    instead of suddenly dropping from full marks to half.
    """
    g, m = cfg["good"], cfg["mild"]
    if value <= g:
        # Perfect to just-at-good-threshold: weight → weight*0.6
        frac  = value / g if g > 0 else 0
        return round(weight * (1.0 - 0.4 * frac), 1)
    elif value <= m:
        # good → mild threshold: weight*0.6 → weight*0.15
        frac  = (value - g) / (m - g)
        return round(weight * (0.6 - 0.45 * frac), 1)
    else:
        # Beyond mild: weight*0.15 → 0  (caps at zero)
        frac  = min((value - m) / m, 1.0)
        return round(max(weight * (0.15 - 0.15 * frac), 0.0), 1)


def detect_thoracic_hump(img_bgr, shoulder_px, hip_px, facing_direction):
    """
    Detects thoracic kyphosis / upper back hump curvature.
    
    In a side view:
    The thoracic spine is located along the posterior (dorsal) side of the torso,
    between the shoulder and hip.
    
    Scans horizontal lines along the upper 75% of the torso (thoracic spine region)
    to trace the dorsal silhouette of the back.
    
    Returns (hump_ratio, max_hump_px, apex_pt).
      • hump_ratio: maximum dorsal displacement as a % of torso length.
      • Normal: ≤ 28%
      • Mild Rounding: 28% - 40%
      • Pronounced Hump (Kyphosis): > 40%
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

    y_start = int(shoulder_px[1] + 0.05 * (hip_px[1] - shoulder_px[1]))
    y_end   = int(shoulder_px[1] + 0.75 * (hip_px[1] - shoulder_px[1]))
    if y_end <= y_start:
        y_start = shoulder_px[1]
        y_end = hip_px[1]

    ys = np.linspace(y_start, y_end, 20).astype(int)
    torso_len = max(math.hypot(hip_px[0] - shoulder_px[0], hip_px[1] - shoulder_px[1]), 1.0)
    direction = 1 if facing_direction == "left" else -1

    spine_xs = []
    back_xs  = []

    for y in ys:
        frac = (y - shoulder_px[1]) / (hip_px[1] - shoulder_px[1] + 1e-6)
        x_spine = int(shoulder_px[0] + frac * (hip_px[0] - shoulder_px[0]))
        spine_xs.append(x_spine)

        border_x = w - 5 if facing_direction == "left" else 5
        bg_ref = np.median(gray[max(0, y-10):min(h, y+10), border_x])
        max_scan = int(torso_len * 0.90)
        best_x = x_spine

        for step in range(5, max_scan):
            x_curr = x_spine + direction * step
            if x_curr < 2 or x_curr >= w - 2:
                break
            grad = sobel_x[y, x_curr]
            diff_from_bg = abs(float(gray[y, x_curr]) - float(bg_ref))
            if diff_from_bg < 25 or grad > 45:
                best_x = x_curr
                if diff_from_bg < 18:
                    break
        back_xs.append(best_x)

    hump_dists = [abs(bx - sx) for bx, sx in zip(back_xs, spine_xs)]
    if len(hump_dists) > 0:
        max_hump_px = max(hump_dists)
        max_idx = np.argmax(hump_dists)
        apex_pt = (back_xs[max_idx], ys[max_idx])
    else:
        max_hump_px = 0
        apex_pt = shoulder_px

    hump_ratio = round((max_hump_px / torso_len) * 100.0, 1)
    return hump_ratio, max_hump_px, apex_pt


def overall_verdict(score, b_hump="good", b_neck="good"):
    if b_hump == "poor":
        return "POOR POSTURE (DORSAL HUMP)"
    if b_neck == "poor":
        return "POOR POSTURE"
    if score >= VERDICT_GOOD:
        if b_hump == "mild":
            return "NEEDS IMPROVEMENT"
        return "GOOD POSTURE"
    if score >= VERDICT_MID:
        return "NEEDS IMPROVEMENT"
    return "POOR POSTURE"


# ══════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════

def draw_normal(img, ear_px, shoulder_px, hip_px, bands_info, score, verdict, knee_px=None, ankle_px=None, apex_pt=None):
    """Skeleton lines + plumb line + thoracic hump curve + colour-coded metric labels + HUD."""
    out    = img.copy()
    h, w   = out.shape[:2]
    colour = VERDICT_BGR.get(verdict, (0, 200, 0))

    # 1. Vertical reference plumb line through hip
    plumb_x = hip_px[0]
    for y in range(0, h, 16):
        cv2.line(out, (plumb_x, y), (plumb_x, min(y + 8, h - 1)), (0, 220, 255), 1)

    # 2. Main skeleton lines
    cv2.line(out, ear_px,      shoulder_px, colour, 3)
    cv2.line(out, shoulder_px, hip_px,      colour, 3)
    if knee_px is not None:
        cv2.line(out, hip_px, knee_px, colour, 3)
        if ankle_px is not None:
            cv2.line(out, knee_px, ankle_px, colour, 3)

    # 3. Thoracic Kyphosis (Back Hump) curvature visualization
    hump_val  = bands_info.get("back_hump_val", 0)
    hump_band = bands_info.get("back_hump", "good")
    if apex_pt is not None and hump_band != "good":
        hump_col = BAND_BGR[hump_band]
        cv2.line(out, shoulder_px, apex_pt, hump_col, 3)
        cv2.line(out, apex_pt, hip_px, hump_col, 3)
        cv2.circle(out, apex_pt, 9, (255, 255, 255), -1)
        cv2.circle(out, apex_pt, 6, hump_col, -1)
        # Text label near apex
        tx_apex = apex_pt[0] + 12 if apex_pt[0] < w - 120 else apex_pt[0] - 130
        cv2.putText(out, f"HUMP ({hump_val}%)", (tx_apex, apex_pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, hump_col, 2)

    # 4. Landmark circles with double rings
    pts = [ear_px, shoulder_px, hip_px]
    if knee_px is not None: pts.append(knee_px)
    if ankle_px is not None: pts.append(ankle_px)
    for pt in pts:
        cv2.circle(out, pt, 9, (255, 255, 255), -1)
        cv2.circle(out, pt, 6, colour, -1)

    # 5. HUD Metric Box (semi-transparent overlay)
    neck_val = bands_info.get("neck_angle_val", bands_info.get("head_offset_val", 0))
    neck_band = bands_info.get("neck_angle", bands_info.get("head_offset", "good"))
    lean_val = bands_info.get("torso_lean_val", 0)
    lean_dir = bands_info.get("lean_direction", "")
    strain   = bands_info.get("cervical_strain_kg", 5.0)

    card_w, card_h = 330, 138
    cx0, cy0 = 15, 15
    overlay = out.copy()
    cv2.rectangle(overlay, (cx0, cy0), (cx0 + card_w, cy0 + card_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.rectangle(out, (cx0, cy0), (cx0 + card_w, cy0 + card_h), colour, 2)

    cv2.putText(out, f"VERDICT: {verdict}", (cx0 + 10, cy0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    cv2.putText(out, f"Neck Angle: {neck_val}deg ({neck_band.upper()})", (cx0 + 10, cy0 + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, BAND_BGR.get(neck_band, (0, 200, 0)), 1)
    cv2.putText(out, f"Back Hump: {hump_val}% ({hump_band.upper()})", (cx0 + 10, cy0 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, BAND_BGR.get(hump_band, (0, 200, 0)), 1)
    lean_text = f"Torso Lean: {lean_val}deg" + (f" ({lean_dir})" if lean_dir else "")
    cv2.putText(out, lean_text, (cx0 + 10, cy0 + 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, BAND_BGR.get(bands_info.get('torso_lean', 'good'), (0, 200, 0)), 1)
    cv2.putText(out, f"Score: {score}/100 | Neck Load: ~{strain}kg", (cx0 + 10, cy0 + 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return out


def draw_debug(img, ear_px, shoulder_px, hip_px,
               ear_vis, shoulder_vis, hip_vis,
               knee_px=None, ankle_px=None, knee_vis=None, ankle_vis=None):
    """
    Debug view: labelled circles on EAR/SHO/HIP with visibility values,
    reference plumb lines.
    """
    out = img.copy()
    h, w = out.shape[:2]

    cv2.line(out, ear_px,      shoulder_px, (255, 100, 0),   2)   # blue
    cv2.line(out, shoulder_px, hip_px,      (255, 255, 255), 2)   # white
    if knee_px is not None:
        cv2.line(out, hip_px, knee_px, (0, 200, 255), 2)
        if ankle_px is not None:
            cv2.line(out, knee_px, ankle_px, (0, 200, 255), 2)

    # Dashed vertical through hip
    for y0 in range(0, h, 20):
        y1 = min(y0 + 10, h - 1)
        cv2.line(out, (hip_px[0], y0), (hip_px[0], y1), (0, 255, 255), 1)

    landmarks_debug = [
        (ear_px,      ear_vis,      "EAR", (255, 100,  0)),
        (shoulder_px, shoulder_vis, "SHO", (100, 255, 100)),
        (hip_px,      hip_vis,      "HIP", (  0, 200, 255)),
    ]
    if knee_px is not None and knee_vis is not None:
        landmarks_debug.append((knee_px, knee_vis, "KNEE", (255, 180, 0)))
    if ankle_px is not None and ankle_vis is not None:
        landmarks_debug.append((ankle_px, ankle_vis, "ANKLE", (200, 100, 255)))

    for px, vis, label, col in landmarks_debug:
        cv2.circle(out, px, 13, col, -1)
        cv2.circle(out, px, 13, (255, 255, 255), 1)
        v_str = f" v={vis:.2f}" if vis is not None else ""
        cv2.putText(out, f"{label}{v_str}",
                    (px[0] + 16, px[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    return out


# ══════════════════════════════════════════════════════════════════════
#  CSV & EXCEL LOG HELPERS
# ══════════════════════════════════════════════════════════════════════

LOG_COLS = ["date", "mode", "head_offset", "neck_angle", "torso_lean", "back_hump", "score"]


def df_to_csv_bytes(df):
    """Convert dataframe to CSV UTF-8 bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df):
    """Convert dataframe to Excel (.xlsx) bytes for download."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def load_log():
    """
    Load posture log from Excel (.xlsx) or CSV (.csv).
    Loads whichever was updated most recently, with automatic fallback.
    """
    df = None
    csv_exists = os.path.exists(LOG_FILE_CSV)
    xlsx_exists = os.path.exists(LOG_FILE_XLSX)

    if xlsx_exists and csv_exists:
        if os.path.getmtime(LOG_FILE_XLSX) >= os.path.getmtime(LOG_FILE_CSV):
            try: df = pd.read_excel(LOG_FILE_XLSX)
            except Exception: df = pd.read_csv(LOG_FILE_CSV)
        else:
            try: df = pd.read_csv(LOG_FILE_CSV)
            except Exception: df = pd.read_excel(LOG_FILE_XLSX)
    elif xlsx_exists:
        try: df = pd.read_excel(LOG_FILE_XLSX)
        except Exception: df = None
    elif csv_exists:
        try: df = pd.read_csv(LOG_FILE_CSV)
        except Exception: df = None

    if df is not None:
        for col in LOG_COLS:
            if col not in df.columns:
                df[col] = np.nan
        # Synchronize neck_angle and head_offset for full compatibility
        if "neck_angle" not in df.columns or df["neck_angle"].isna().all():
            df["neck_angle"] = df["head_offset"]
        if "head_offset" not in df.columns or df["head_offset"].isna().all():
            df["head_offset"] = df["neck_angle"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return pd.DataFrame(columns=LOG_COLS)


def save_to_log(mode, neck_angle, torso_lean, score, back_hump=0.0):
    """Save posture entry to both CSV and Excel (.xlsx) files."""
    df  = load_log()
    row = pd.DataFrame([{
        "date":        datetime.date.today().isoformat(),
        "mode":        mode,
        "head_offset": neck_angle,
        "neck_angle":  neck_angle,
        "torso_lean":  torso_lean,
        "back_hump":   back_hump,
        "score":       score,
    }])
    updated = pd.concat([df, row], ignore_index=True)
    # Save CSV
    try:
        updated.to_csv(LOG_FILE_CSV, index=False)
    except Exception as e:
        st.warning(f"Warning: Could not save to {LOG_FILE_CSV}: {e}")
    # Save Excel
    try:
        updated.to_excel(LOG_FILE_XLSX, index=False)
    except Exception as e:
        st.warning(f"Warning: Could not save to {LOG_FILE_XLSX}: {e}")


def load_nutrition_log():
    """
    Load nutrition log from Excel (.xlsx) or CSV (.csv).
    Loads whichever was updated most recently, with automatic fallback.
    """
    cols = ["date", "height_cm", "weight_kg", "age", "gender",
            "vitamin_d", "iron", "calcium", "magnesium",
            "protein_g", "water_L", "glow_score"]
    df = None
    csv_exists = os.path.exists(NUTRITION_LOG_CSV)
    xlsx_exists = os.path.exists(NUTRITION_LOG_XLSX)

    if xlsx_exists and csv_exists:
        if os.path.getmtime(NUTRITION_LOG_XLSX) >= os.path.getmtime(NUTRITION_LOG_CSV):
            try: df = pd.read_excel(NUTRITION_LOG_XLSX)
            except Exception: df = pd.read_csv(NUTRITION_LOG_CSV)
        else:
            try: df = pd.read_csv(NUTRITION_LOG_CSV)
            except Exception: df = pd.read_excel(NUTRITION_LOG_XLSX)
    elif xlsx_exists:
        try: df = pd.read_excel(NUTRITION_LOG_XLSX)
        except Exception: df = None
    elif csv_exists:
        try: df = pd.read_csv(NUTRITION_LOG_CSV)
        except Exception: df = None

    if df is not None:
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    return pd.DataFrame(columns=cols)


def save_nutrition_log(row_dict):
    """Save nutrition entry to both CSV and Excel (.xlsx) files."""
    df = load_nutrition_log()
    updated = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    # Save CSV
    try:
        updated.to_csv(NUTRITION_LOG_CSV, index=False)
    except Exception as e:
        st.warning(f"Warning: Could not save to {NUTRITION_LOG_CSV}: {e}")
    # Save Excel
    try:
        updated.to_excel(NUTRITION_LOG_XLSX, index=False)
    except Exception as e:
        st.warning(f"Warning: Could not save to {NUTRITION_LOG_XLSX}: {e}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2 HELPERS
# ══════════════════════════════════════════════════════════════════════

def calc_bmi(weight_kg, height_cm):
    return round(weight_kg / (height_cm / 100) ** 2, 1)


def bmi_verdict(bmi):
    if bmi < 18.5: return "Underweight"
    if bmi < 25.0: return "Healthy weight"
    if bmi < 30.0: return "Overweight"
    return "Obese"


def glow_score_from_nutrition(vitamin_d, iron, calcium, magnesium,
                               protein_g, water_L, bmi):
    vd  = min(vitamin_d / 800,  1.0)
    fe  = min(iron      / 18,   1.0)
    ca  = min(calcium   / 1000, 1.0)
    mg  = min(magnesium / 400,  1.0)
    pr  = min(protein_g / 60,   1.0)
    h2o = min(water_L   / 2.5,  1.0)
    bmi_s = (1.0 if 18.5 <= bmi <= 24.9
             else max(0.0, bmi / 18.5) if bmi < 18.5
             else max(0.0, 1.0 - (bmi - 25) / 15))
    return round((vd + fe + ca + mg + pr + h2o + bmi_s) / 7 * 100, 1)


def nutrition_tips(vitamin_d, iron, calcium, magnesium, protein_g, water_L, bmi):
    tips = []
    if vitamin_d < 400:
        tips.append(("☀️", "**Vitamin D low.** 15-20 min morning sun + egg yolks / fortified milk / D3."))
    if iron < 8:
        tips.append(("🥩", "**Iron low.** Red meat, spinach, lentils. Pair with vitamin-C food."))
    if calcium < 700:
        tips.append(("🥛", "**Calcium low.** Dairy, tofu, almonds or leafy greens every day."))
    if magnesium < 250:
        tips.append(("🌰", "**Magnesium low.** Pumpkin seeds, dark chocolate, whole grains."))
    if protein_g < 40:
        tips.append(("🍳", "**Protein low.** Eggs, chicken, beans or paneer at every meal."))
    if water_L < 1.5:
        tips.append(("💧", "**Not enough water.** Aim 2 L / day. Dehydration dulls skin."))
    if bmi < 18.5:
        tips.append(("🍽️", "**BMI under-range.** Nuts, avocado, whole grains, dairy."))
    elif bmi >= 25:
        tips.append(("🏃", "**BMI over-range.** Whole foods, less sugar, 30 min movement daily."))
    return tips


# ══════════════════════════════════════════════════════════════════════
#  APP  –  PHASE 1: POSTURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

st.title("🌟 Posture Glow-Up Tracker")
st.caption("Phase 1 checks your posture → Phase 2 builds your full glow-up plan.")

if "posture_done"  not in st.session_state: st.session_state.posture_done  = False
if "posture_score" not in st.session_state:
    df_init = load_log()
    if len(df_init) > 0 and df_init["score"].notna().any():
        st.session_state.posture_score = float(df_init["score"].dropna().iloc[-1])
        st.session_state.posture_done  = True
    else:
        st.session_state.posture_score = None


st.header("📸 Phase 1 — Posture Check")
st.write("Upload a **side-view** photo. Your ear, shoulder and hip must all be visible.")

col_ctrl1, col_ctrl2 = st.columns([2, 1])
with col_ctrl1:
    mode = st.radio("Posture mode", ["Standing", "Sitting"], horizontal=True)
with col_ctrl2:
    debug_mode = st.checkbox("🔬 Show debug")

uploaded = st.file_uploader("Choose a JPG or PNG", type=["jpg", "jpeg", "png"])

if uploaded is not None:

    # ── 1. Read image ─────────────────────────────────────────────────
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read image. Please upload a valid JPG or PNG.")
        st.stop()

    orig_h, orig_w = img_bgr.shape[:2]

    # ── 2. Pad to square (centred, white borders) ─────────────────────
    # This stops MediaPipe cropping out the hip on tall narrow photos.
    padded_bgr, pad_top, pad_left = pad_to_square(img_bgr)
    pad_size = padded_bgr.shape[0]   # square: pad_size × pad_size

    # Tasks API needs RGB uint8
    padded_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)

    # ── 3. Run PoseLandmarker (Tasks API) ─────────────────────────────
    landmarks = run_pose(padded_rgb)   # returns list of 33 NormalizedLandmark or None

    if landmarks is None:
        st.error(
            "⚠️ No person detected. Try a clearer side-profile photo with "
            "good lighting and your full body from ear to hip visible."
        )
        st.stop()

    # ── 4. Pick the better-visible side (no averaging) ────────────────
    ear_lm, shoulder_lm, hip_lm, knee_lm, ankle_lm, side, facing_dir = pick_side(landmarks)

    # ── 5. Convert landmarks to pixel coords in the PADDED image ──────
    ear_px      = lm_to_padded_px(ear_lm,      pad_size)
    shoulder_px = lm_to_padded_px(shoulder_lm, pad_size)
    hip_px      = lm_to_padded_px(hip_lm,      pad_size)

    knee_vis    = _vis(knee_lm)
    ankle_vis   = _vis(ankle_lm)
    knee_px     = lm_to_padded_px(knee_lm, pad_size) if knee_vis >= 0.4 else None
    ankle_px    = lm_to_padded_px(ankle_lm, pad_size) if ankle_vis >= 0.4 else None

    # ── 6. Confidence check ───────────────────────────────────────────
    low_vis = {
        "Ear":      _vis(ear_lm)      < VIS_THRESHOLD,
        "Shoulder": _vis(shoulder_lm) < VIS_THRESHOLD,
        "Hip":      _vis(hip_lm)      < VIS_THRESHOLD,
    }
    low_confidence = any(low_vis.values())

    # ── 7. Pre-compute measurements ───────────────────────────────────
    neck_angle      = calc_neck_angle(ear_lm, shoulder_lm, hip_lm)
    torso_lean      = calc_torso_lean(shoulder_lm, hip_lm)
    lean_dir        = get_lean_direction(shoulder_lm, hip_lm, facing_dir, torso_lean)
    cervical_strain = calc_cervical_strain(neck_angle)

    # Detect Thoracic Kyphosis (Upper Back Hump)
    hump_ratio, max_hump_px, apex_pt = detect_thoracic_hump(padded_bgr, shoulder_px, hip_px, facing_dir)

    lean_cfg    = THRESHOLDS["torso_lean_standing" if mode == "Standing"
                             else "torso_lean_sitting"]
    b_neck      = band(neck_angle, THRESHOLDS["neck_angle"])
    b_hump      = band(hump_ratio, THRESHOLDS["thoracic_kyphosis"])
    b_lean      = band(torso_lean, lean_cfg)

    score       = round(
        band_score(b_neck, THRESHOLDS["neck_angle"]["weight"],
                   neck_angle, THRESHOLDS["neck_angle"]) +
        band_score(b_hump, THRESHOLDS["thoracic_kyphosis"]["weight"],
                   hump_ratio, THRESHOLDS["thoracic_kyphosis"]) +
        band_score(b_lean, lean_cfg["weight"],
                   torso_lean, lean_cfg), 1)

    verdict     = overall_verdict(score, b_hump=b_hump, b_neck=b_neck)
    bands_info  = {
        "neck_angle":         b_neck, "neck_angle_val":         neck_angle,
        "head_offset":        b_neck, "head_offset_val":        neck_angle,
        "back_hump":          b_hump, "back_hump_val":          hump_ratio,
        "torso_lean":         b_lean, "torso_lean_val":         torso_lean,
        "lean_direction":     lean_dir,
        "cervical_strain_kg": cervical_strain,
    }

    # ── Always-visible raw numbers box ────────────────────────────────
    _nose    = landmarks[0]
    _l_sho   = landmarks[LEFT_SHOULDER]
    _r_sho   = landmarks[RIGHT_SHOULDER]
    _mid_sho = (_l_sho.x + _r_sho.x) / 2.0
    with st.expander("🔢 Raw landmark & biomechanics data (click to expand)", expanded=True):
        st.write(f"**Side used:** {side}  |  **Facing direction:** {facing_dir}")
        st.write(f"**Nose x:** {_nose.x:.4f}  |  **mid-shoulder x:** {_mid_sho:.4f}  |  facing: {facing_dir}")
        st.write(f"**Ear**      → x={ear_lm.x:.4f}, y={ear_lm.y:.4f}, confidence={_vis(ear_lm):.3f}")
        st.write(f"**Shoulder** → x={shoulder_lm.x:.4f}, y={shoulder_lm.y:.4f}, confidence={_vis(shoulder_lm):.3f}")
        st.write(f"**Hip**      → x={hip_lm.x:.4f}, y={hip_lm.y:.4f}, confidence={_vis(hip_lm):.3f}")
        if knee_px: st.write(f"**Knee**     → visible (confidence={knee_vis:.3f})")
        if ankle_px: st.write(f"**Ankle**    → visible (confidence={ankle_vis:.3f})")
        st.write(f"**neck_angle:** {neck_angle}°  (good≤{THRESHOLDS['neck_angle']['good']}°, mild≤{THRESHOLDS['neck_angle']['mild']}°)")
        st.write(f"**back_hump (kyphosis):** {hump_ratio}%  (good≤{THRESHOLDS['thoracic_kyphosis']['good']}%, mild≤{THRESHOLDS['thoracic_kyphosis']['mild']}%)")
        st.write(f"**torso_lean:** {torso_lean}° ({lean_dir})  (good≤{lean_cfg['good']}°, mild≤{lean_cfg['mild']}°)")
        st.write(f"**Cervical Strain:** ~{cervical_strain} kg equivalent head load")
        st.write(f"**Score:** {score}/100  → **{verdict}**")

    # ── 8. Draw on the PADDED image and show it ───────────────────────
    if debug_mode:
        annotated = draw_debug(
            padded_bgr, ear_px, shoulder_px, hip_px,
            _vis(ear_lm), _vis(shoulder_lm), _vis(hip_lm),
            knee_px=knee_px, ankle_px=ankle_px, knee_vis=knee_vis, ankle_vis=ankle_vis,
        )
    else:
        annotated = draw_normal(padded_bgr, ear_px, shoulder_px, hip_px,
                                bands_info, score, verdict,
                                knee_px=knee_px, ankle_px=ankle_px,
                                apex_pt=apex_pt)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption=f"Padded to {pad_size}×{pad_size} px  |  Side: {side}  |  Facing: {facing_dir}",
             width="stretch")

    # ── 9. Confidence warning / stop ──────────────────────────────────
    if low_confidence:
        bad_names = ", ".join(k for k, v in low_vis.items() if v)
        st.error(
            f"🔴 **Low confidence — retake the photo.**  "
            f"Landmarks with low visibility: **{bad_names}**. "
            "Make sure your full side profile is in frame, well-lit, "
            "and the hip is not hidden."
        )
        st.subheader("🔬 Landmark visibility")
        st.dataframe(pd.DataFrame([
            {"Landmark": "Ear",      "visibility": round(_pres(ear_lm),      3),
             "x (px)": ear_px[0],      "y (px)": ear_px[1]},
            {"Landmark": "Shoulder", "visibility": round(_pres(shoulder_lm), 3),
             "x (px)": shoulder_px[0], "y (px)": shoulder_px[1]},
            {"Landmark": "Hip",      "visibility": round(_pres(hip_lm),      3),
             "x (px)": hip_px[0],      "y (px)": hip_px[1]},
        ]), use_container_width=True)
        st.stop()   # do not show measurements or verdict

    # ── 10. Debug table (when debug checkbox is on) ───────────────────
    if debug_mode:
        st.subheader("🔬 Debug Information")
        st.markdown("**Landmark positions & visibility**")
        st.dataframe(pd.DataFrame([
            {"Landmark": "Ear",      "x (px)": ear_px[0],
             "y (px)": ear_px[1],      "visibility": round(_pres(ear_lm),      3)},
            {"Landmark": "Shoulder", "x (px)": shoulder_px[0],
             "y (px)": shoulder_px[1], "visibility": round(_pres(shoulder_lm), 3)},
            {"Landmark": "Hip",      "x (px)": hip_px[0],
             "y (px)": hip_px[1],      "visibility": round(_pres(hip_lm),      3)},
        ]), use_container_width=True)

        st.markdown("**Computed measurements**")
        w_neck = THRESHOLDS["neck_angle"]["weight"]
        w_hump = THRESHOLDS["thoracic_kyphosis"]["weight"]
        w_lean = lean_cfg["weight"]
        st.dataframe(pd.DataFrame([
            {"Measure": "neck_angle (forward deviation)",
             "Value": f"{neck_angle} °",
             "Good threshold": f"≤ {THRESHOLDS['neck_angle']['good']} °",
             "Mild threshold": f"≤ {THRESHOLDS['neck_angle']['mild']} °",
             "Band": b_neck,
             "Points": band_score(b_neck, w_neck, neck_angle, THRESHOLDS["neck_angle"])},
            {"Measure": "thoracic_kyphosis (back hump ratio)",
             "Value": f"{hump_ratio} %",
             "Good threshold": f"≤ {THRESHOLDS['thoracic_kyphosis']['good']} %",
             "Mild threshold": f"≤ {THRESHOLDS['thoracic_kyphosis']['mild']} %",
             "Band": b_hump,
             "Points": band_score(b_hump, w_hump, hump_ratio, THRESHOLDS["thoracic_kyphosis"])},
            {"Measure": f"torso_lean ({mode.lower()})",
             "Value": f"{torso_lean} °",
             "Good threshold": f"≤ {lean_cfg['good']} °",
             "Mild threshold": f"≤ {lean_cfg['mild']} °",
             "Band": b_lean,
             "Points": band_score(b_lean, w_lean, torso_lean, lean_cfg)},
        ]), use_container_width=True)
        st.caption(
            f"Side used: **{side}** | "
            f"Pad: top={pad_top}px left={pad_left}px | "
            f"Score: **{score}/100**"
        )

    # ── 11. Measurement tiles ─────────────────────────────────────────
    st.subheader("📊 Your measurements")

    def measure_row(label, value, unit, b, tip_mild, tip_poor):
        emoji  = BAND_EMOJI[b]
        blabel = {"good": "Good ✅", "mild": "Mild ⚠️", "poor": "Poor 🔴"}[b]
        st.markdown(f"**{label}:** `{value}{unit}` &nbsp; {emoji} **{blabel}**")
        if b == "mild":  st.caption(f"💡 {tip_mild}")
        elif b == "poor": st.caption(f"🔴 {tip_poor}")

    measure_row(
        "Neck Alignment (Forward Head)", neck_angle, f"° (Equivalent Neck Load: ~{cervical_strain} kg)", b_neck,
        tip_mild=f"Head slightly forward (~{cervical_strain} kg cervical strain vs 5 kg normal). Chin tucks: pull chin back, hold 5 sec × 10 reps hourly.",
        tip_poor=(
            f"Forward head posture! Applying ~{cervical_strain} kg of compressive neck load (normal ~5 kg). "
            "Perform chin tucks hourly, raise screen to eye level, and check pillow height."
        ),
    )
    measure_row(
        "Thoracic Kyphosis (Upper Back Hump)", hump_ratio, "% of torso length", b_hump,
        tip_mild="Mild upper back rounding. Perform doorway chest openers and foam roller thoracic extension.",
        tip_poor=(
            f"Pronounced Upper Back Hump ({hump_ratio}% dorsal protrusion)! "
            "Perform prone cobra raises, face pulls, and foam roller extension 10 min daily to open the ribcage."
        ),
    )
    measure_row(
        f"Torso Alignment ({mode} - {lean_dir})", torso_lean, f"° ({lean_dir})", b_lean,
        tip_mild=(
            f"Slight {lean_dir.lower()} (target ≤ {lean_cfg['good']}°). "
            "Engage core and imagine a string pulling your head upward."
        ),
        tip_poor=(
            f"Significant {lean_dir.lower()} (target ≤ {lean_cfg['mild']}°). "
            "Wall-stand: head, shoulders and bum touching wall 2 min daily. "
            "3 × 30-sec planks to strengthen core."
        ),
    )
    if knee_px and ankle_px and mode == "Standing":
        st.markdown("**Lower Kinetic Chain:** `Knee & Ankle In Frame` ✅ &nbsp; 🟢 **Balanced Base**")
        st.caption("Lower body plumb line supports spinal vertical alignment.")

    # ── 12. Overall verdict ───────────────────────────────────────────
    st.divider()
    if "DORSAL HUMP" in verdict:
        st.error(f"🔴 **POOR POSTURE (UPPER BACK HUMP DETECTED)** — Score: {score}/100")
        st.warning(f"⚠️ **Pronounced Thoracic Kyphosis:** The upper back bows backward ({hump_ratio}% of torso length), creating a visible dorsal hump.")
        st.markdown("""
| Action Plan for Back Hump & Rounded Upper Spine | |
|---|---|
| 🧘 **Thoracic Spine Mobility** | Lay upper back across a foam roller for 3–5 min daily with gentle arching. |
| 🚪 **Chest Opener Stretch** | 3 × 30-sec doorway stretch to release tight pectoralis minor dragging shoulders forward. |
| 🦅 **Rhomboid Strengthening** | Prone Cobras & Y-T-W raises (3 × 15 reps) to pull the shoulder blades back. |
| 🦴 **Vertebral Health** | Maintain Calcium (1,000 mg) and Vitamin D (≥800 IU) to protect thoracic spine density. |
        """)
    elif verdict == "GOOD POSTURE":
        st.success(f"✅ **GOOD POSTURE** — Score: {score}/100")
        st.markdown("""
| | |
|---|---|
| 💪 | Core planks and dead bugs — keep your torso stable. |
| 🧘 | 5-min chest-opener stretch daily. |
| 🪑 | Screen at eye level, feet flat on floor. |
| 😴 | Sleep on your back or side, not your stomach. |
        """)
    elif verdict == "NEEDS IMPROVEMENT":
        st.warning(f"⚠️ **NEEDS IMPROVEMENT** — Score: {score}/100")
        st.info("Work on the orange/red measures above. Small daily habits compound fast! 💪")
    else:
        st.error(f"🔴 **POOR POSTURE** — Score: {score}/100")
        st.info("Focus on both red measures. Pick the easiest one first and build the habit. 💪")

    # ── 13. Progress comparison ───────────────────────────────────────
    df_log = load_log()
    if len(df_log) > 0 and df_log["score"].notna().any():
        last_score = df_log["score"].dropna().iloc[-1]
        diff       = score - last_score
        if diff > CHANGE_THRESHOLD:
            st.balloons()
            st.info(f"✨ **Glow UP!** Score improved by {diff:.1f} pts!")
        elif diff < -CHANGE_THRESHOLD:
            st.info(f"📉 **Glow Down** — score dropped {abs(diff):.1f} pts. Keep going!")
        else:
            st.info(f"➡️ **About the same** ({diff:+.1f} pts). Consistency is progress!")

    # ── 14. Save button ───────────────────────────────────────────────
    if st.button("💾 Save posture to my log (CSV & Excel)"):
        save_to_log(mode, neck_angle, torso_lean, score, back_hump=hump_ratio)
        st.success("Saved to posture_log.csv & posture_log.xlsx! 🎉")

    st.session_state.posture_done  = True
    st.session_state.posture_score = score


# ── Progress charts ───────────────────────────────────────────────────
st.divider()
st.header("📈 Posture Progress")

df_log = load_log()
if len(df_log) == 0 or df_log["score"].isna().all():
    st.info("No posture log yet. Upload a photo and save your first result above!")
else:
    df_log["date"] = pd.to_datetime(df_log["date"], errors="coerce")
    df_v = df_log.dropna(subset=["score", "date"])
    if len(df_v) > 0:
        st.metric("Last-7 average", f"{round(df_v.tail(7)['score'].mean(), 1)} / 100")
        st.subheader("Score over time")
        st.line_chart(df_v.set_index("date")[["score"]])
        st.subheader("Daily average score")
        daily = df_v.groupby(df_v["date"].dt.date)["score"].mean().reset_index()
        daily.columns = ["date", "avg_score"]
        st.bar_chart(daily.set_index("date"))
    with st.expander("📋 Full posture log & Excel/CSV Export"):
        st.dataframe(df_log.sort_values("date", ascending=False), use_container_width=True)
        cd1, cd2 = st.columns(2)
        with cd1:
            st.download_button(
                "📥 Download posture_log.xlsx (Excel)",
                data=df_to_excel_bytes(df_log),
                file_name="posture_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_posture_xlsx",
            )
        with cd2:
            st.download_button(
                "📥 Download posture_log.csv (CSV)",
                data=df_to_csv_bytes(df_log),
                file_name="posture_log.csv",
                mime="text/csv",
                key="dl_posture_csv",
            )


# ══════════════════════════════════════════════════════════════════════
#  APP  –  PHASE 2: NUTRITION + BODY GLOW-UP PLAN
# ══════════════════════════════════════════════════════════════════════

st.divider()
st.header("🌿 Phase 2 — Personalised Glow-Up Plan")

if not st.session_state.posture_done and len(load_log()) == 0:
    st.info("👆 Complete Phase 1 first to unlock Phase 2.")
    st.stop()

st.write("Fill in your body stats and daily nutrients to get your **Glow Score**.")

with st.form("nutrition_form"):
    st.subheader("👤 Body Stats")
    c1, c2, c3 = st.columns(3)
    with c1: height_cm = st.number_input("Height (cm)", 100, 250, 165, step=1)
    with c2: weight_kg = st.number_input("Weight (kg)",  20, 200,  60, step=1)
    with c3: age       = st.number_input("Age",           10, 100,  22, step=1)
    gender = st.radio("Gender", ["Female", "Male", "Prefer not to say"], horizontal=True)

    st.subheader("🥗 Daily Nutrient Intake (approximate)")
    c4, c5 = st.columns(2)
    with c4:
        vitamin_d = st.slider("☀️ Vitamin D (IU/day)",  0, 2000,  400, step=50)
        iron      = st.slider("🥩 Iron (mg/day)",        0,   40,    8, step=1)
        calcium   = st.slider("🦴 Calcium (mg/day)",     0, 2000,  700, step=50)
    with c5:
        magnesium = st.slider("🌰 Magnesium (mg/day)",   0,  600,  280, step=10)
        protein_g = st.slider("🍳 Protein (g/day)",      0,  200,   50, step=5)
        water_L   = st.slider("💧 Water (L/day)",       0.0, 5.0,  1.8, step=0.1)
    submitted = st.form_submit_button("✨ Calculate my Glow Score")

if submitted:
    bmi        = calc_bmi(weight_kg, height_cm)
    bmi_cat    = bmi_verdict(bmi)
    glow_score = glow_score_from_nutrition(
        vitamin_d, iron, calcium, magnesium, protein_g, water_L, bmi)
    posture_sc = st.session_state.posture_score
    combined   = round(posture_sc * 0.4 + glow_score * 0.6, 1) if posture_sc else glow_score

    st.subheader("🌟 Your Glow Results")
    g1, g2, g3 = st.columns(3)
    g1.metric("BMI", bmi, bmi_cat)
    g2.metric("Nutrition Score", f"{glow_score} / 100")
    g3.metric("Combined Glow", f"{combined} / 100",
              "Posture + Nutrition" if posture_sc else "Nutrition only")

    prev_log = load_nutrition_log()
    if len(prev_log) > 0 and "glow_score" in prev_log.columns:
        prev = prev_log["glow_score"].dropna()
        if len(prev) > 0:
            diff_g = glow_score - prev.iloc[-1]
            if diff_g > CHANGE_THRESHOLD:
                st.balloons(); st.success(f"✨ **GLOW UP!** Nutrition score rose {diff_g:.1f} pts!")
            elif diff_g < -CHANGE_THRESHOLD:
                st.warning(f"📉 **Glow Down** — dropped {abs(diff_g):.1f} pts.")
            else:
                st.info(f"➡️ **About the same** ({diff_g:+.1f} pts).")

    if glow_score >= 75:
        st.success("🎉 **You're glowing!** Nutritional habits are impressive.")
        st.markdown("""
| | |
|---|---|
| 🥗 | Well-balanced diet — keep the variety! |
| 💧 | Hydration solid — skin and organs thank you. |
| 💪 | Nutrients support healthy muscles and bones. |
| 🌟 | You're building a real glow-up foundation. |
        """)
    elif glow_score >= 50:
        st.info("😊 **Pretty good** — a few tweaks will take you to glowing!")
        for emoji, tip in nutrition_tips(vitamin_d, iron, calcium, magnesium, protein_g, water_L, bmi):
            st.markdown(f"- {emoji} {tip}")
    else:
        st.warning("⚠️ **Needs attention** — your body needs better fuel.")
        for emoji, tip in nutrition_tips(vitamin_d, iron, calcium, magnesium, protein_g, water_L, bmi):
            st.markdown(f"- {emoji} {tip}")
        st.info("💡 Pick 1-2 tips, stick to them 7 days, then add more. 🌱")

    st.divider()
    st.subheader("🌈 Holistic Glow-Up Summary")
    p_label = ("✅ Good"       if posture_sc and posture_sc >= 80 else
               "😊 Moderate"   if posture_sc and posture_sc >= 50 else
               "⚠️ Needs work" if posture_sc else "Not checked yet")
    n_label = ("✅ Great" if glow_score >= 75 else
               "😊 Good"  if glow_score >= 50 else "⚠️ Needs attention")
    b_label = "✅ Healthy" if bmi_cat == "Healthy weight" else f"⚠️ {bmi_cat}"
    st.markdown(f"""
| Area | Status |
|---|---|
| 🧍 Posture | {p_label} ({posture_sc}/100 pts) |
| 🥗 Nutrition | {n_label} ({glow_score}/100 pts) |
| ⚖️ BMI | {b_label} ({bmi}) |
| 🌟 **Overall Glow Score** | **{combined} / 100** |
    """)

    st.subheader("📅 Your 7-Day Glow-Up Action Plan")
    st.markdown("""
| Day | Action |
|---|---|
| Mon | Morning chin-tuck + posture wall-stand (5 min) |
| Tue | Add one calcium-rich food to your meals |
| Wed | 30-min walk + 2 extra glasses of water |
| Thu | Core plank challenge — 3 × 30 sec |
| Fri | Protein-rich meal (eggs / dal / chicken / paneer) |
| Sat | 15-20 min morning sunlight (Vitamin D ☀️) |
| Sun | Rest + reflect — re-take your posture photo 📸 |
    """)

    if st.button("💾 Save nutrition data (CSV & Excel)"):
        save_nutrition_log({
            "date": datetime.date.today().isoformat(),
            "height_cm": height_cm, "weight_kg": weight_kg,
            "age": age, "gender": gender,
            "vitamin_d": vitamin_d, "iron": iron,
            "calcium": calcium, "magnesium": magnesium,
            "protein_g": protein_g, "water_L": water_L,
            "glow_score": glow_score,
        })
        st.success("Nutrition data saved to nutrition_log.csv & nutrition_log.xlsx! 🌿")

    nlog = load_nutrition_log()
    if len(nlog) > 1 and "glow_score" in nlog.columns:
        st.divider()
        st.subheader("📈 Nutrition Glow Score Over Time")
        nlog["date"] = pd.to_datetime(nlog["date"], errors="coerce")
        nlog = nlog.dropna(subset=["date", "glow_score"])
        if len(nlog) > 0:
            st.line_chart(nlog.set_index("date")[["glow_score"]])
        with st.expander("📋 Full nutrition log & Excel/CSV Export"):
            st.dataframe(nlog.sort_values("date", ascending=False), use_container_width=True)
            cn1, cn2 = st.columns(2)
            with cn1:
                st.download_button(
                    "📥 Download nutrition_log.xlsx (Excel)",
                    data=df_to_excel_bytes(nlog),
                    file_name="nutrition_log.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_nutrition_xlsx",
                )
            with cn2:
                st.download_button(
                    "📥 Download nutrition_log.csv (CSV)",
                    data=df_to_csv_bytes(nlog),
                    file_name="nutrition_log.csv",
                    mime="text/csv",
                    key="dl_nutrition_csv",
                )


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3: PERFECT POSTURE OUTCOME & CORRELATION BLUEPRINT
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.header("🏆 Phase 3 — Perfect Posture Outcome (100/100 Blueprint)")
st.caption("Connecting your Posture & Nutrition logs to reach the clinical Gold Standard of Posture.")

df_p = load_log()
df_n = load_nutrition_log()

latest_posture = df_p.dropna(subset=["score"]).iloc[-1] if len(df_p.dropna(subset=["score"])) > 0 else None
latest_nutri   = df_n.dropna(subset=["glow_score"]).iloc[-1] if len(df_n.dropna(subset=["glow_score"])) > 0 else None

col_goal1, col_goal2 = st.columns(2)

with col_goal1:
    st.subheader("🎯 Path to 100/100 Perfect Posture")
    if latest_posture is not None:
        cur_sc = float(latest_posture["score"])
        cur_neck = float(latest_posture.get("neck_angle", latest_posture.get("head_offset", 0)))
        cur_lean = float(latest_posture["torso_lean"])
        cur_hump = float(latest_posture.get("back_hump", 0.0))

        st.metric("Current Posture Score", f"{cur_sc} / 100", f"{round(100 - cur_sc, 1)} pts to Perfect (100)")
        st.progress(min(max(cur_sc / 100.0, 0.0), 1.0))

        # Deviation breakdown
        neck_gap = max(0.0, round(cur_neck - 15.0, 1))
        hump_gap = max(0.0, round(cur_hump - 28.0, 1))
        lean_gap = max(0.0, round(cur_lean - 8.0, 1))

        st.markdown(f"""
- **Neck Alignment**: Current `{cur_neck}°` (Ideal: `≤15°`). Deficit: **{neck_gap}°**
- **Upper Back Hump**: Current `{cur_hump}%` (Ideal: `≤28%`). Deficit: **{hump_gap}%**
- **Spine Tilt**: Current `{cur_lean}°` (Ideal: `≤8°`). Deficit: **{lean_gap}°**
        """)
    else:
        st.info("Upload your posture photo in Phase 1 to see your tailored 100/100 roadmap.")

with col_goal2:
    st.subheader("🧬 Posture & Nutrition Synergy")
    st.markdown("""
Nutrients directly determine your spinal biomechanics and postural endurance:
- **Magnesium**: Relaxes hyperactive suboccipital & neck muscles that pull your head forward.
- **Calcium & Vitamin D**: Rebuilds vertebral bone density to prevent upper back hump (kyphosis).
- **Protein**: Sustains deep spinal stabilizers (multifidus, erector spinae, rhomboids).
- **Water**: Rehydrates spinal discs, maintaining vertebral height and shock absorption.
    """)

# ── Dual Log Correlation Chart ───────────────────────────────────────
if len(df_p) > 0 and len(df_n) > 0:
    st.subheader("📊 Posture vs Nutrition Trends Over Time")
    
    # Merge logs by date
    df_p_clean = df_p.dropna(subset=["date", "score"]).copy()
    df_n_clean = df_n.dropna(subset=["date", "glow_score"]).copy()
    
    df_p_clean["date_only"] = df_p_clean["date"].dt.date
    df_n_clean["date_only"] = df_n_clean["date"].dt.date
    
    p_daily = df_p_clean.groupby("date_only")["score"].mean().reset_index()
    n_daily = df_n_clean.groupby("date_only")["glow_score"].mean().reset_index()
    
    merged_trends = pd.merge(p_daily, n_daily, on="date_only", how="outer").sort_values("date_only")
    merged_trends.rename(columns={"score": "Posture Score", "glow_score": "Nutrition Glow Score", "date_only": "Date"}, inplace=True)
    merged_trends.set_index("Date", inplace=True)
    
    st.line_chart(merged_trends)
    st.caption("Notice how improvements in daily nutrient intake track with increases in posture alignment and stability.")

# ── Custom Posture Prescription ───────────────────────────────────────
st.subheader("📋 Your Clinical Posture Prescription")
rx1, rx2, rx3 = st.columns(3)

with rx1:
    st.markdown("**1. Deep Neck Flexor Training**")
    st.markdown("""
- **Chin Tucks**: 10 reps × 5-sec hold every 2 hours.
- **Chest Doorway Stretch**: 3 × 30 seconds daily to open tight pec muscles.
    """)

with rx2:
    st.markdown("**2. Spinal Stability & Core**")
    st.markdown("""
- **Prone Cobra**: 3 sets × 20 seconds to strengthen mid-trapezius & rhomboids.
- **Deadbugs & Side Planks**: 3 sets × 30 seconds to lock pelvis & spine.
    """)

with rx3:
    st.markdown("**3. Cellular Support**")
    st.markdown("""
- **Hydration**: 2.0–2.5 L daily to rehydrate spinal discs.
- **Micronutrients**: Maintain Vitamin D ≥ 800 IU & Magnesium ≥ 350 mg for muscle release.
    """)

# ── File Sync & External Import Center ────────────────────────────────
with st.expander("📂 Log File Management (Import / Sync Excel & CSV)"):
    st.write("Both `posture_log` and `nutrition_log` are automatically synchronized in both **Excel (.xlsx)** and **CSV (.csv)** formats.")
    st.write(f"- Posture log location: `{LOG_FILE_CSV}` and `{LOG_FILE_XLSX}`")
    st.write(f"- Nutrition log location: `{NUTRITION_LOG_CSV}` and `{NUTRITION_LOG_XLSX}`")

    uploaded_external = st.file_uploader("Upload external Excel (.xlsx) or CSV (.csv) file to import", type=["xlsx", "xls", "csv"], key="ext_uploader")
    if uploaded_external is not None:
        try:
            if uploaded_external.name.endswith(".csv"):
                ext_df = pd.read_csv(uploaded_external)
            else:
                ext_df = pd.read_excel(uploaded_external)

            # Auto-detect whether it's posture or nutrition
            if "neck_angle" in ext_df.columns or "head_offset" in ext_df.columns or "torso_lean" in ext_df.columns:
                existing_p = load_log()
                combined_p = pd.concat([existing_p, ext_df], ignore_index=True).drop_duplicates(subset=["date", "mode"], keep="last")
                combined_p.to_csv(LOG_FILE_CSV, index=False)
                combined_p.to_excel(LOG_FILE_XLSX, index=False)
                st.success(f"✅ Successfully imported {len(ext_df)} posture records! Reload the page to see updated trends.")
            elif "glow_score" in ext_df.columns or "vitamin_d" in ext_df.columns or "calcium" in ext_df.columns:
                existing_n = load_nutrition_log()
                combined_n = pd.concat([existing_n, ext_df], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
                combined_n.to_csv(NUTRITION_LOG_CSV, index=False)
                combined_n.to_excel(NUTRITION_LOG_XLSX, index=False)
                st.success(f"✅ Successfully imported {len(ext_df)} nutrition records! Reload the page to see updated trends.")
            else:
                st.warning("Could not automatically classify file columns. Please ensure it contains posture or nutrition headers.")
        except Exception as err:
            st.error(f"Error importing file: {err}")

