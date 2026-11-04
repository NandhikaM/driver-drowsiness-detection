"""
Driver Drowsiness Detection — v2
Features:
  - MediaPipe Face Mesh (468 landmarks, no dlib/extra model needed)
  - EAR (Eye Aspect Ratio) for eye closure — blink-aware, not triggered by normal blinks
  - PERCLOS (% eye closure over rolling 30-second window) — industry standard metric
  - Blink rate tracking (too few blinks = fatigue signal)
  - MAR (Mouth Aspect Ratio) for yawn detection
  - Head pose estimation (pitch/yaw/roll via solvePnP)
  - Graduated alerts: WARNING → CRITICAL
  - Alert history logging to logs/alerts.csv
  - Thread-safe DetectorState class (no module globals)
  - Both beep + pyttsx3 TTS wake-up call
"""

import cv2
import numpy as np
import os
import time
import threading
import csv
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
# Optional imports — graceful fallbacks
# ─────────────────────────────────────────────
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    MP_AVAILABLE = False
    print("[WARNING] mediapipe not installed. Run: pip install mediapipe")

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 160)
    _tts_engine.setProperty('volume', 1.0)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("[WARNING] pyttsx3 not available. TTS alerts disabled.")

# ─────────────────────────────────────────────
# MediaPipe landmark indices
# ─────────────────────────────────────────────
# Left eye: outer corner=33, inner=133, top=159, bottom=145, top2=158, bottom2=153
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # P1..P6 for EAR
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Mouth for MAR
MOUTH_IDX = [61, 291, 0, 17, 39, 181, 269, 405]  # outer lip corners + top/bottom

# Head pose landmarks
HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]  # nose, chin, left-eye, right-eye, mouth-left, mouth-right

# 3D model points for head pose
MODEL_3D = np.array([
    (0.0,     0.0,    0.0),       # Nose tip
    (0.0,  -330.0,  -65.0),       # Chin
    (-225.0, 170.0, -135.0),      # Left eye corner
    (225.0,  170.0, -135.0),      # Right eye corner
    (-150.0,-150.0, -125.0),      # Left mouth corner
    (150.0, -150.0, -125.0)       # Right mouth corner
], dtype=np.float64)

# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────
EAR_THRESH            = 0.22    # below this = eye closed
BLINK_CONSEC_FRAMES   = 3       # frames for a normal blink (won't trigger alert)
EYE_CLOSED_WARN_SEC   = 1.0     # seconds closed → WARNING
EYE_CLOSED_CRIT_SEC   = 2.5     # seconds closed → CRITICAL

PERCLOS_WINDOW_SEC    = 30      # rolling window for PERCLOS
PERCLOS_WARN_THRESH   = 0.20    # 20% closed in window → warning
PERCLOS_CRIT_THRESH   = 0.35    # 35% closed → critical

BLINK_RATE_WINDOW_SEC = 60      # window for blink rate
BLINK_RATE_LOW        = 8       # blinks/min below this = fatigue signal

MAR_THRESH            = 0.70    # mouth open = yawn
YAWN_CONSEC_SEC       = 1.5     # seconds of high MAR = confirmed yawn

PITCH_WARN            = 18      # degrees head nod
YAW_WARN              = 30      # degrees head turn
HEAD_WARN_SEC         = 2.0     # seconds of bad head pose → alert

BEEP_COOLDOWN         = 3.0     # seconds between beeps
TTS_COOLDOWN          = 8.0     # seconds between TTS calls

# ─────────────────────────────────────────────
# Alert levels
# ─────────────────────────────────────────────
LEVEL_OK       = 0
LEVEL_WARNING  = 1
LEVEL_CRITICAL = 2

LEVEL_LABELS = {LEVEL_OK: "OK", LEVEL_WARNING: "WARNING", LEVEL_CRITICAL: "CRITICAL"}

# ─────────────────────────────────────────────
# Audio alerts
# ─────────────────────────────────────────────
def _beep(level=LEVEL_WARNING):
    """Blocking beep — run in thread."""
    try:
        import winsound
        if level == LEVEL_CRITICAL:
            for _ in range(3):
                winsound.Beep(1200, 300)
                time.sleep(0.1)
        else:
            winsound.Beep(900, 400)
        return
    except ImportError:
        pass
    try:
        import subprocess
        if level == LEVEL_CRITICAL:
            for _ in range(3):
                subprocess.run(["beep", "-f", "1200", "-l", "300"], capture_output=True)
        else:
            subprocess.run(["beep", "-f", "900", "-l", "400"], capture_output=True)
        return
    except Exception:
        pass
    # Terminal bell fallback
    count = 3 if level == LEVEL_CRITICAL else 1
    for _ in range(count):
        print("\a", end="", flush=True)
        time.sleep(0.15)

def _speak(text):
    """Blocking TTS — run in thread."""
    if not TTS_AVAILABLE:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 155)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[TTS] Error: {e}")

TTS_MESSAGES = {
    "eyes_warn":  "Warning. Your eyes are closing. Stay alert.",
    "eyes_crit":  "Wake up! Eyes closed too long! Pull over safely!",
    "yawn":       "Yawning detected. Consider taking a break soon.",
    "head_warn":  "Watch your head position. Stay focused on the road.",
    "perclos":    "Drowsiness pattern detected. You need a rest break.",
    "blink_low":  "Low blink rate detected. You may be entering a microsleep.",
}

# ─────────────────────────────────────────────
# Alert logger
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_PATH  = os.path.join(BASE_DIR, "../logs/alerts.csv")

def _ensure_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "level", "type", "detail"])

def log_alert(level_str, alert_type, detail=""):
    _ensure_log()
    try:
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), level_str, alert_type, detail])
    except Exception:
        pass

# ─────────────────────────────────────────────
# EAR computation
# ─────────────────────────────────────────────
def compute_ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    # EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if C < 1e-6:
        return 0.3
    return (A + B) / (2.0 * C)

def compute_mar(landmarks, w, h):
    """Mouth Aspect Ratio — higher = more open."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH_IDX]
    # corners: 0(left) 1(right), top: 2,4 bottom: 3,5,6,7
    horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    if horiz < 1e-6:
        return 0.0
    vert1 = np.linalg.norm(np.array(pts[2]) - np.array(pts[3]))
    vert2 = np.linalg.norm(np.array(pts[4]) - np.array(pts[5]))
    vert3 = np.linalg.norm(np.array(pts[6]) - np.array(pts[7]))
    return (vert1 + vert2 + vert3) / (3.0 * horiz)

def compute_head_pose(landmarks, w, h):
    """Returns (pitch, yaw, roll) in degrees or None."""
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h) for i in HEAD_POSE_IDX
    ], dtype=np.float64)

    focal  = w
    cam_m  = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    dist   = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(MODEL_3D, image_points, cam_m, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

# ─────────────────────────────────────────────
# Thread-safe Detector State
# ─────────────────────────────────────────────
class DetectorState:
    def __init__(self, fps=30):
        self.fps = fps
        self.lock = threading.Lock()

        # Eye closure timing
        self.eye_closed_start   = None   # timestamp when eyes first closed
        self.eye_was_closed     = False

        # Blink detection
        self.blink_frame_count  = 0      # consecutive closed frames (for blink gate)
        self.blink_timestamps   = deque() # timestamps of confirmed blinks

        # PERCLOS rolling window
        self.perclos_window     = deque() # (timestamp, is_closed) tuples

        # Yawn timing
        self.yawn_start         = None
        self.yawn_active        = False

        # Head pose timing
        self.head_warn_start    = None

        # Alert state
        self.current_level      = LEVEL_OK
        self.active_alerts      = []     # list of dicts with type/msg/level

        # Audio cooldowns
        self.last_beep_time     = 0
        self.last_tts_time      = 0
        self.last_tts_key       = None

        # Stats
        self.session_start      = time.time()
        self.total_blinks       = 0
        self.alert_count        = {"eyes": 0, "yawn": 0, "head": 0, "perclos": 0}

        # MediaPipe
        if MP_AVAILABLE:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None

    def _maybe_beep(self, level):
        now = time.time()
        if now - self.last_beep_time > BEEP_COOLDOWN:
            self.last_beep_time = now
            threading.Thread(target=_beep, args=(level,), daemon=True).start()

    def _maybe_tts(self, key):
        now = time.time()
        cooldown = TTS_COOLDOWN if key != self.last_tts_key else TTS_COOLDOWN * 2
        if now - self.last_tts_time > cooldown:
            self.last_tts_time = now
            self.last_tts_key  = key
            msg = TTS_MESSAGES.get(key, "")
            if msg:
                threading.Thread(target=_speak, args=(msg,), daemon=True).start()

    def _alert(self, alert_type, msg, level, tts_key=None):
        # Update active alerts list
        existing = [a for a in self.active_alerts if a["type"] == alert_type]
        if not existing:
            self.active_alerts.append({"type": alert_type, "msg": msg, "level": level})
            self.alert_count[alert_type] = self.alert_count.get(alert_type, 0) + 1
            log_alert(LEVEL_LABELS[level], alert_type, msg)
        else:
            existing[0]["level"] = level
            existing[0]["msg"]   = msg

        self.current_level = max(self.current_level, level)
        self._maybe_beep(level)
        if tts_key:
            self._maybe_tts(tts_key)

    def _clear_alert(self, alert_type):
        self.active_alerts = [a for a in self.active_alerts if a["type"] != alert_type]

    def get_perclos(self):
        """Return PERCLOS value (0–1) over rolling window."""
        now = time.time()
        cutoff = now - PERCLOS_WINDOW_SEC
        # Prune old entries
        while self.perclos_window and self.perclos_window[0][0] < cutoff:
            self.perclos_window.popleft()
        if not self.perclos_window:
            return 0.0
        closed = sum(1 for _, c in self.perclos_window if c)
        return closed / len(self.perclos_window)

    def get_blink_rate(self):
        """Blinks per minute over rolling window."""
        now = time.time()
        cutoff = now - BLINK_RATE_WINDOW_SEC
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()
        elapsed = min(now - self.session_start, BLINK_RATE_WINDOW_SEC)
        if elapsed < 10:
            return None  # not enough data
        return (len(self.blink_timestamps) / elapsed) * 60

    def get_stats(self):
        return {
            "session_seconds": int(time.time() - self.session_start),
            "total_blinks":    self.total_blinks,
            "perclos":         round(self.get_perclos(), 3),
            "blink_rate":      self.get_blink_rate(),
            "alert_count":     dict(self.alert_count),
            "current_level":   LEVEL_LABELS[self.current_level],
        }

    def process_frame(self, frame):
        """Process a single BGR frame. Returns (annotated_frame, alerts_list, stats_dict)."""
        if not MP_AVAILABLE or self.face_mesh is None:
            cv2.putText(frame, "mediapipe not installed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, [], self.get_stats()

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        now = time.time()
        self.current_level = LEVEL_OK
        self.active_alerts = []

        if not result.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            # Gradually reset counters when no face
            if self.eye_closed_start and (now - self.eye_closed_start) > 5:
                self.eye_closed_start = None
                self.eye_was_closed   = False
            return frame, list(self.active_alerts), self.get_stats()

        lm = result.multi_face_landmarks[0].landmark

        # ── EAR ────────────────────────────────
        ear_l = compute_ear(lm, LEFT_EYE_IDX,  w, h)
        ear_r = compute_ear(lm, RIGHT_EYE_IDX, w, h)
        ear   = (ear_l + ear_r) / 2.0
        eyes_closed = ear < EAR_THRESH

        # PERCLOS recording
        self.perclos_window.append((now, eyes_closed))

        # Blink gate — short closures are blinks, not drowsiness
        if eyes_closed:
            self.blink_frame_count += 1
        else:
            if 1 <= self.blink_frame_count <= BLINK_CONSEC_FRAMES:
                # It was a blink
                self.total_blinks += 1
                self.blink_timestamps.append(now)
            self.blink_frame_count = 0

        # Sustained eye closure (beyond blink duration)
        sustained_closed = eyes_closed and self.blink_frame_count > BLINK_CONSEC_FRAMES

        if sustained_closed:
            if self.eye_closed_start is None:
                self.eye_closed_start = now
            closed_duration = now - self.eye_closed_start
            if closed_duration >= EYE_CLOSED_CRIT_SEC:
                self._alert("eyes", f"EYES CLOSED {closed_duration:.1f}s — WAKE UP!", LEVEL_CRITICAL, "eyes_crit")
            elif closed_duration >= EYE_CLOSED_WARN_SEC:
                self._alert("eyes", f"Eyes closing ({closed_duration:.1f}s)…", LEVEL_WARNING, "eyes_warn")
        else:
            self.eye_closed_start = None
            self.blink_frame_count = max(0, self.blink_frame_count)  # keep count for running blink

        # ── PERCLOS ────────────────────────────
        perclos = self.get_perclos()
        if perclos >= PERCLOS_CRIT_THRESH:
            self._alert("perclos", f"PERCLOS {perclos:.0%} — HIGH DROWSINESS", LEVEL_CRITICAL, "perclos")
        elif perclos >= PERCLOS_WARN_THRESH:
            self._alert("perclos", f"PERCLOS {perclos:.0%} — Drowsy pattern", LEVEL_WARNING)

        # ── Blink rate ──────────────────────────
        blink_rate = self.get_blink_rate()
        if blink_rate is not None and blink_rate < BLINK_RATE_LOW:
            self._alert("blink_low", f"Low blink rate: {blink_rate:.0f}/min", LEVEL_WARNING, "blink_low")

        # ── MAR / Yawn ──────────────────────────
        mar = compute_mar(lm, w, h)
        if mar > MAR_THRESH:
            if self.yawn_start is None:
                self.yawn_start = now
            yawn_dur = now - self.yawn_start
            if yawn_dur >= YAWN_CONSEC_SEC:
                self._alert("yawn", "YAWNING — Take a break!", LEVEL_WARNING, "yawn")
        else:
            self.yawn_start = None
            self._clear_alert("yawn")

        # ── Head pose ───────────────────────────
        pose = compute_head_pose(lm, w, h)
        if pose:
            pitch, yaw, roll = pose
            bad_pose = abs(pitch) > PITCH_WARN or abs(yaw) > YAW_WARN
            if bad_pose:
                if self.head_warn_start is None:
                    self.head_warn_start = now
                if now - self.head_warn_start >= HEAD_WARN_SEC:
                    detail = f"Pitch:{pitch:.0f}° Yaw:{yaw:.0f}°"
                    self._alert("head", f"HEAD POSITION WARNING — {detail}", LEVEL_WARNING, "head_warn")
            else:
                self.head_warn_start = None
                self._clear_alert("head")

        # ── Draw overlays ───────────────────────
        self._draw_frame(frame, lm, w, h, ear, mar, perclos, blink_rate, pose)

        return frame, list(self.active_alerts), self.get_stats()

    def _draw_frame(self, frame, lm, w, h, ear, mar, perclos, blink_rate, pose):
        # Draw eye contours
        for idx_set, color in [(LEFT_EYE_IDX, (0, 255, 200)), (RIGHT_EYE_IDX, (0, 255, 200))]:
            pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in idx_set])
            cv2.polylines(frame, [pts], True, color, 1)

        # Draw mouth contour
        m_pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH_IDX])
        cv2.polylines(frame, [m_pts], True, (255, 200, 50), 1)

        # Draw head pose axis (nose tip arrow)
        if pose:
            pitch, yaw, roll = pose
            nose = (int(lm[1].x * w), int(lm[1].y * h))
            dx   = int(yaw   * 2)
            dy   = int(pitch * 2)
            cv2.arrowedLine(frame, nose, (nose[0]+dx, nose[1]+dy), (255, 80, 80), 2, tipLength=0.3)

        # HUD metrics bar at bottom
        bar_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status text
        blink_str = f"{blink_rate:.0f}" if blink_rate is not None else "--"
        hud = (f"EAR:{ear:.2f}  MAR:{mar:.2f}  "
               f"PERCLOS:{perclos:.0%}  Blinks/min:{blink_str}")
        if pose:
            pitch, yaw, roll = pose
            hud += f"  P:{pitch:.0f}° Y:{yaw:.0f}°"

        cv2.putText(frame, hud, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)

        # Alert overlay banners
        for i, alert in enumerate(self.active_alerts):
            y0 = 50 + i * 55
            color = (0, 60, 200) if alert["level"] == LEVEL_CRITICAL else (0, 130, 220)
            bg    = (0, 0, 180)  if alert["level"] == LEVEL_CRITICAL else (0, 80, 160)
            ov2   = frame.copy()
            cv2.rectangle(ov2, (0, y0 - 35), (w, y0 + 10), bg, -1)
            cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
            prefix = "⚠ " if alert["level"] == LEVEL_WARNING else "🚨 "
            cv2.putText(frame, alert["msg"], (12, y0),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2)

        # Level indicator top-right
        level = self.current_level
        label = LEVEL_LABELS[level]
        lcolor = {LEVEL_OK: (50, 200, 80), LEVEL_WARNING: (0, 180, 255), LEVEL_CRITICAL: (0, 50, 255)}[level]
        cv2.putText(frame, label, (w - 120, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, lcolor, 2)


# ─────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────
def run_standalone():
    print("=" * 55)
    print("  Driver Drowsiness Detection v2")
    print("  Press Q to quit")
    print("=" * 55)

    state = DetectorState()
    cap   = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, alerts, stats = state.process_frame(frame)

        if alerts:
            for a in alerts:
                print(f"[{LEVEL_LABELS[a['level']]}] {a['msg']}")

        cv2.imshow("Driver Drowsiness Detection v2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession Stats:", state.get_stats())


if __name__ == "__main__":
    run_standalone()
