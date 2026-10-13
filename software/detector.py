"""
NapTrap — Drowsiness Detection (Laptop-side)
Captures webcam feed, computes Eye Aspect Ratio via MediaPipe,
sends alert state to ESP32 over serial, and displays a live dashboard.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import serial
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
EAR_THRESHOLD = 0.21        # fallback — overwritten by calibration at startup
CLOSURE_TIME_LIMIT = 1.0    # seconds of closure before ALARM
SERIAL_PORT = "COM3"        # Windows: COM3/COM4  |  Mac: /dev/cu.usbserial-...  |  Linux: /dev/ttyUSB0
SERIAL_BAUD = 115200

# ──────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Eye landmark indices (MediaPipe Face Mesh 468-point model)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def compute_ear(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio for one eye: height / width."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if horiz == 0:
        return 0.3
    return (v1 + v2) / (2.0 * horiz)


# ──────────────────────────────────────────────
# SERIAL (optional — runs without ESP32 too)
# ──────────────────────────────────────────────
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    print(f"Serial connected on {SERIAL_PORT}")
except Exception:
    ser = None
    print("No serial connection — running without ESP32")

# ──────────────────────────────────────────────
# CSV LOG
# ──────────────────────────────────────────────
# Build the path relative to THIS script's location, not the current
# working directory — so it works whether you run `python detector.py`
# from software/, or launch it via VS Code's debugger from the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
log_path = os.path.join(LOGS_DIR, f"naptrap_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
log_file = open(log_path, "w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "ear", "state"])

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Try VideoCapture(1).")
    exit()

# ──────────────────────────────────────────────
# CALIBRATION — sets EAR_THRESHOLD per person
# ──────────────────────────────────────────────
print("\n=== EAR CALIBRATION ===")
print("Step 1: Look at the camera with eyes OPEN normally.")
print("        Press 's' when ready to sample.\n")

open_ears = []
closed_ears = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear = 0.0
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        ear = (compute_ear(lm, LEFT_EYE, w, h) + compute_ear(lm, RIGHT_EYE, w, h)) / 2.0

    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if len(open_ears) < 20:
        cv2.putText(frame, "Eyes OPEN - press 's' to sample", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif len(closed_ears) < 20:
        cv2.putText(frame, "Eyes CLOSED - press 's' to sample", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("NapTrap - Calibration", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(open_ears) < 20:
            # Collect 20 frames with eyes open
            for _ in range(20):
                ret, f = cap.read()
                f = cv2.flip(f, 1)
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    open_ears.append((compute_ear(lm, LEFT_EYE, w, h) +
                                      compute_ear(lm, RIGHT_EYE, w, h)) / 2.0)
                time.sleep(0.05)
            print(f"  Open EAR average: {np.mean(open_ears):.3f}")
            print("\nStep 2: Now CLOSE your eyes and press 's'.")
        elif len(closed_ears) < 20:
            # Collect 20 frames with eyes closed
            for _ in range(20):
                ret, f = cap.read()
                f = cv2.flip(f, 1)
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    closed_ears.append((compute_ear(lm, LEFT_EYE, w, h) +
                                         compute_ear(lm, RIGHT_EYE, w, h)) / 2.0)
                time.sleep(0.05)
            print(f"  Closed EAR average: {np.mean(closed_ears):.3f}")
            break

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Threshold = midpoint between open and closed
open_avg = np.mean(open_ears)
closed_avg = np.mean(closed_ears)
EAR_THRESHOLD = (open_avg + closed_avg) / 2.0

print(f"\n  Calibrated threshold: {EAR_THRESHOLD:.3f}")
print(f"  (open={open_avg:.3f}, closed={closed_avg:.3f})")
print("\nCalibration done — starting detection...\n")
cv2.destroyAllWindows()

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
eyes_closed_since = None
state = "OK"
ear_history = []

print("Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear = 0.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        left_ear = compute_ear(lm, LEFT_EYE, w, h)
        right_ear = compute_ear(lm, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0

        # Draw eye landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            x_pt, y_pt = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (x_pt, y_pt), 2, (0, 255, 0), -1)

        # Drowsiness logic
        if ear < EAR_THRESHOLD:
            if eyes_closed_since is None:
                eyes_closed_since = time.time()
            elapsed = time.time() - eyes_closed_since
            if elapsed >= CLOSURE_TIME_LIMIT:
                state = "ALARM"
            else:
                state = "WARN"
        else:
            eyes_closed_since = None
            state = "OK"
    else:
        state = "NO_FACE"

    # Send state to ESP32
    if ser:
        ser.write(f"{state}\n".encode())

    # Log to CSV
    csv_writer.writerow([datetime.now().isoformat(), f"{ear:.4f}", state])

    # ── Dashboard overlay ──
    color = {
        "OK": (0, 200, 0),
        "WARN": (0, 200, 255),
        "ALARM": (0, 0, 255),
        "NO_FACE": (128, 128, 128),
    }[state]

    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"State: {state}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if state == "ALARM":
        cv2.putText(frame, "!!! WAKE UP !!!", (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ── EAR rolling graph ──
    ear_history.append(ear)
    if len(ear_history) > w:
        ear_history = ear_history[-w:]

    graph_h = 100
    graph_y = h - graph_h - 10
    cv2.rectangle(frame, (0, graph_y), (w, h), (0, 0, 0), -1)
    thresh_y = graph_y + graph_h - int(EAR_THRESHOLD / 0.4 * graph_h)
    cv2.line(frame, (0, thresh_y), (w, thresh_y), (0, 0, 255), 1)
    cv2.putText(frame, "threshold", (5, thresh_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for i in range(1, len(ear_history)):
        y1 = graph_y + graph_h - int(ear_history[i - 1] / 0.4 * graph_h)
        y2 = graph_y + graph_h - int(ear_history[i] / 0.4 * graph_h)
        cv2.line(frame, (i - 1, y1), (i, y2), (0, 255, 0), 1)

    cv2.imshow("NapTrap - Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
if ser:
    ser.close()
print(f"Log saved to {log_path}")