"""
Flask Web App — Driver Drowsiness Detection v2
Routes:
  GET  /              → Driver HUD (camera + voice assistant)
  GET  /stats         → Separate stats dashboard
  GET  /video_feed    → MJPEG stream
  GET  /api/status    → JSON: alerts + stats
  GET  /api/log       → JSON: last 50 alert log entries
  POST /api/chat      → Chat assistant (Groq or rule-based fallback)
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import os
import csv
import random

from drowsiness_detector import DetectorState, LOG_PATH, _ensure_log

app   = Flask(__name__)
state = DetectorState()

# ─────────────────────────────────────────────
# Video generator
# ─────────────────────────────────────────────
_cap_lock = threading.Lock()
_cap      = None

def get_cap():
    global _cap
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            _cap = cv2.VideoCapture(0)
            _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return _cap

def generate_frames():
    while True:
        cap = get_cap()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame, _, _ = state.process_frame(frame)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

# ─────────────────────────────────────────────
# Rule-based chat fallback
# ─────────────────────────────────────────────
CHAT_RESPONSES = {
    "tired":   ["I can tell you're feeling tired. How long have you been driving?",
                "It sounds like your body needs rest. Can you pull over somewhere safe soon?"],
    "long":    ["Long drives are exhausting. Are you able to take a 20-minute nap at a rest stop?",
                "Driving more than 2 hours without a break significantly increases drowsiness risk."],
    "coffee":  ["Caffeine can help short-term, but it's not a substitute for sleep. How much have you had?",
                "Coffee takes about 20 minutes to kick in. A short nap plus coffee works even better!"],
    "stop":    ["That's a great idea. Even a 15 to 20 minute nap can restore alertness significantly.",
                "Rest stops are your friend. Your safety is worth the extra time."],
    "music":   ["Music can help! Try upbeat, unfamiliar songs — familiar ones let your brain auto-pilot.",
                "Singing along to music is a great way to stay alert on the road."],
    "window":  ["Fresh air and cooler temperatures do help! Lower the window or turn up the AC.",
                "Cold air activates your alertness reflex. Good short-term strategy."],
    "alone":   ["Driving alone makes it harder to stay alert. Try calling a friend hands-free.",
                "An engaging podcast or audiobook can also help keep your mind active."],
    "how far": ["How far do you still need to go? It might be worth finding a place to rest nearby.",
                "If you're still far away, a rest break now is much safer than pushing through."],
    "default": ["How are you feeling right now?",
                "Stay focused — I'm here with you. What's on your mind?",
                "Keep talking to me. Conversation helps keep you alert.",
                "You're doing great by staying aware. What's your situation?"]
}

def rule_based_chat(user_msg, alerts, stats):
    msg_lower = user_msg.lower()
    for keyword, responses in CHAT_RESPONSES.items():
        if keyword in msg_lower:
            response = random.choice(responses)
            break
    else:
        level = stats.get("current_level", "OK")
        if level == "CRITICAL":
            response = "You seem very drowsy. Please pull over as soon as it's safe. Your safety is the priority."
        elif level == "WARNING":
            response = "I'm noticing some drowsiness signals. How long have you been driving?"
        else:
            response = random.choice(CHAT_RESPONSES["default"])
    if any(a.get("level") == 2 for a in (alerts or [])):
        response += " Please find a safe place to stop soon."
    return response

def groq_chat(user_msg, alerts, stats, history):
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return rule_based_chat(user_msg, alerts, stats)
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        alert_ctx = ""
        if alerts:
            alert_ctx = "Active alerts: " + "; ".join(a["msg"] for a in alerts) + ". "
        system = (
            "You are a caring voice co-pilot for a drowsy driver speaking hands-free. "
            "Keep ALL responses under 2 short spoken sentences — no lists, no bullet points. "
            "Be warm, calm, and direct. "
            "If drowsiness alerts are active, firmly urge them to pull over safely. "
            f"Driver status: {alert_ctx}"
            f"Session: {stats.get('session_seconds', 0) // 60} min, "
            f"PERCLOS: {stats.get('perclos', 0):.0%}, "
            f"Blinks/min: {stats.get('blink_rate') or 'unknown'}."
        )
        messages = [{"role": "system", "content": system}]
        messages += (history or [])[-8:]
        messages.append({"role": "user", "content": user_msg})
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=120,
            messages=messages,
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[Groq] Error: {e}")
        return rule_based_chat(user_msg, alerts, stats)

_chat_histories = {}

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stats")
def stats_page():
    return render_template("stats.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    alerts = [{"type": a["type"], "msg": a["msg"], "level": a["level"]}
              for a in state.active_alerts]
    return jsonify({"alerts": alerts, "stats": state.get_stats(), "level": state.current_level})

@app.route("/api/log")
def api_log():
    _ensure_log()
    entries = []
    try:
        with open(LOG_PATH, "r", newline="") as f:
            entries = list(csv.DictReader(f))[-50:]
    except Exception:
        pass
    return jsonify({"entries": entries})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data       = request.get_json(silent=True) or {}
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    if not user_msg:
        return jsonify({"error": "empty message"}), 400
    history = _chat_histories.setdefault(session_id, [])
    alerts  = [{"msg": a["msg"], "level": a["level"]} for a in state.active_alerts]
    stats   = state.get_stats()
    reply   = groq_chat(user_msg, alerts, stats, history)
    history.append({"role": "user",      "content": user_msg})
    history.append({"role": "assistant", "content": reply})
    if len(history) > 40:
        _chat_histories[session_id] = history[-40:]
    return jsonify({"reply": reply})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print("=" * 55)
    print(f"  DrowseGuard v2")
    print(f"  Driver HUD  -> http://localhost:{args.port}")
    print(f"  Stats       -> http://localhost:{args.port}/stats")
    print("=" * 55)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
