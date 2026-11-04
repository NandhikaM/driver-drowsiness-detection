# 🚗 DrowseGuard — Driver Drowsiness Detection v2

Real-time driver monitoring with a dark HUD dashboard, blink-aware eye tracking, PERCLOS metric, and an AI co-pilot chat assistant.

---

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Face detection | Haar cascade | **MediaPipe Face Mesh** (468 landmarks) |
| Eye detection | CNN on Haar ROIs | **EAR** (Eye Aspect Ratio) from landmarks |
| Blink-aware | ❌ blinks triggered alert | ✅ blinks ignored, only sustained closure alerts |
| Drowsiness metric | Frame count | **PERCLOS** (% eye closure over 30s window) |
| Blink rate | ❌ | ✅ tracked, low rate is a fatigue signal |
| Alerts | Binary (alert/no alert) | **Graduated: WARNING → CRITICAL** |
| Audio | Beep only | **Beep + TTS voice alert** ("Wake up! Pull over!") |
| Alert history | ❌ | ✅ logged to `logs/alerts.csv` |
| Dashboard | Just a camera | **Full dark HUD** with metrics + alert log |
| Chat assistant | ❌ | ✅ AI co-pilot (Anthropic API or rule-based) |
| Thread safety | Global state | **Thread-safe `DetectorState` class** |
| Dependencies | dlib (hard to install) | **No dlib needed** |

---

## 📁 Project Structure

```
drowsiness_v2/
├── models/
│   └── eye_model.h5          ← kept for compatibility (not used in v2)
├── src/
│   ├── app.py                ← Flask web app
│   ├── drowsiness_detector.py← core detection logic
│   └── templates/
│       └── index.html        ← dark HUD dashboard
├── logs/
│   └── alerts.csv            ← auto-created on first alert
└── requirements.txt
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

**Linux extras for TTS beep:**
```bash
sudo apt install beep espeak
```

**Windows:** pyttsx3 uses SAPI5 (built-in), no extras needed.

---

## 🚀 Running

### Web Dashboard (recommended)
```bash
cd src/
python app.py
```
Open **http://localhost:5000** — full HUD with camera, metrics, alerts, chat.

### Standalone OpenCV Window
```bash
cd src/
python drowsiness_detector.py
```

---

## 🔔 Alert Levels

| Level | Trigger | Action |
|---|---|---|
| WARNING | Eyes closed 1s+ OR PERCLOS > 20% OR yawn | Yellow alert + single beep + TTS warning |
| CRITICAL | Eyes closed 2.5s+ OR PERCLOS > 35% | Red alert + triple beep + urgent TTS ("Wake up! Pull over!") |

---

## 🧠 How Blink Detection Works

Normal blinks (≤3 frames, ~100ms) are **ignored** — they do not increment the drowsiness counter.

Only **sustained eye closure** beyond the blink duration triggers the warning/critical pipeline. This eliminates false alerts from normal blinking.

---

## 📊 PERCLOS

PERCLOS (PERcentage of eye CLOSure) is the industry-standard metric for drowsiness:
- Measures % of time eyes are closed over a 30-second rolling window
- > 20% = WARNING
- > 35% = CRITICAL

---

## 🤖 Chat Assistant

The co-pilot chat assistant at **http://localhost:5000** can:
- Engage the driver with questions to keep them alert
- Give rest break advice and safety tips
- Automatically trigger when a CRITICAL alert fires

**With Anthropic API key** (recommended): Set `ANTHROPIC_API_KEY` environment variable for full AI responses.

**Without API key**: Falls back to a smart rule-based response system — no setup needed.

---

## 🎛️ Tuning

In `drowsiness_detector.py`:
```python
EAR_THRESH            = 0.22   # eye closed threshold
BLINK_CONSEC_FRAMES   = 3      # frames that count as a normal blink (not drowsy)
EYE_CLOSED_WARN_SEC   = 1.0    # seconds before WARNING
EYE_CLOSED_CRIT_SEC   = 2.5    # seconds before CRITICAL
PERCLOS_WARN_THRESH   = 0.20   # 20% = warning
PERCLOS_CRIT_THRESH   = 0.35   # 35% = critical
MAR_THRESH            = 0.70   # yawn detection sensitivity
PITCH_WARN            = 18     # head nod degrees
YAW_WARN              = 30     # head turn degrees
```
