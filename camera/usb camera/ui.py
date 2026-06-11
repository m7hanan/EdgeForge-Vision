from flask import Flask, jsonify, request, render_template, Response, send_file
import threading
import time
import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from reportlab.pdfgen import canvas

# 🔥 ADDED (GPIO)
import Jetson.GPIO as GPIO

app = Flask(__name__)

MODEL_PATH = "models/newmodel.engine"
CAM_INDEX = 0

CONF = 0.20
IOU = 0.50
IMGSZ = 640

STREAM_DELAY = 0.07
JPEG_QUALITY = 70

MAX_MISSED_FRAMES = 180

RECORD_DIR = "record"
os.makedirs(RECORD_DIR, exist_ok=True)

# 🔥 ADDED (GPIO setup)
LED_PIN = 7
GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

state = {
    "running": False,
    "recording": False,
    "count": 0,
    "target": 2000,
    "last": "idle",
    "line_y_ratio": 0.25,
    "limit_reached": False,   # 🔥 ADDED
    "roi": {
        "x1": 142,
        "y1": 82,
        "x2": 466,
        "y2": 478
    }
}

lock = threading.Lock()

latest_jpeg = None
frame_lock = threading.Lock()

last_y = {}
last_seen = {}
counted_ids = set()
frame_idx = 0

cap = None
model = None

video_writer = None
record_file = None


def init_once():
    global cap, model

    if model is None:
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        print("Model loaded")

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def reset_tracking():
    global last_y, last_seen, counted_ids, frame_idx
    last_y = {}
    last_seen = {}
    counted_ids = set()
    frame_idx = 0


def start_recording(w, h):
    global video_writer, record_file

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = os.path.join(RECORD_DIR, f"{ts}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(record_file, fourcc, 20, (w, h))


def stop_recording():
    global video_writer

    if video_writer is not None:
        video_writer.release()
        video_writer = None


def create_pdf():

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(RECORD_DIR, f"{ts}_report.pdf")

    c = canvas.Canvas(pdf_path)

    c.setFont("Helvetica", 16)
    c.drawString(100, 800, "Coconut Counting Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 760, f"Date: {datetime.now()}")
    c.drawString(100, 730, f"Total Coconut Count: {state['count']}")
    c.drawString(100, 700, f"Target: {state['target']}")

    c.save()

    return pdf_path


def vision_worker():

    global latest_jpeg, frame_idx

    init_once()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    print(f"Camera started {w}x{h}")

    while True:

        ok, frame = cap.read()

        if not ok:
            time.sleep(0.01)
            continue

        frame_idx += 1

        with lock:
            running = state["running"]
            recording = state["recording"]
            target = state["target"]
            current_count = state["count"]
            line_ratio = state["line_y_ratio"]
            roi = state["roi"]
            limit_reached = state["limit_reached"]  # 🔥 ADDED

        rx1 = int(roi["x1"])
        ry1 = int(roi["y1"])
        rx2 = int(roi["x2"])
        ry2 = int(roi["y2"])

        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)

        line_y = int(h * line_ratio)
        cv2.line(frame, (0, line_y), (w, line_y), (0,255,255), 3)

        # 🔥 MODIFIED CONDITION (stop when limit reached)
        if running and not limit_reached:

            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                verbose=False
            )

            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:

                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)

                new_count = current_count

                for (x1, y1, x2, y2), tid in zip(boxes, ids):

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

                    last_seen[tid] = frame_idx

                    if tid not in last_y:
                        last_y[tid] = cy
                        continue

                    prev_y = last_y[tid]
                    last_y[tid] = cy

                    if prev_y > line_y and cy <= line_y and tid not in counted_ids:
                        new_count += 1
                        counted_ids.add(tid)

                # 🔥 LIMIT CHECK
                if new_count >= target:
                    new_count = target
                    GPIO.output(LED_PIN, GPIO.HIGH)

                    with lock:
                        state["limit_reached"] = True
                        state["running"] = False
                        state["last"] = "limit reached"

                with lock:
                    state["count"] = new_count

        cv2.putText(frame, f"COUNT: {state['count']}/{target}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        if recording and video_writer is not None:
            video_writer.write(frame)

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if ret:
            with frame_lock:
                latest_jpeg = buf.tobytes()


def gen_stream():
    while True:

        with frame_lock:
            frame_bytes = latest_jpeg

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(STREAM_DELAY)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    with lock:
        return jsonify(state)


@app.post("/api/start")
def api_start():
    with lock:
        state["running"] = True
        state["last"] = "started"
    return ("",204)


@app.post("/api/stop")
def api_stop():
    with lock:
        state["running"] = False
        state["last"] = "stopped"
    return ("",204)


@app.post("/api/reset")
def api_reset():
    with lock:
        state["count"] = 0
        state["limit_reached"] = False  # 🔥 ADDED
        state["last"] = "reset"
    GPIO.output(LED_PIN, GPIO.LOW)  # 🔥 ADDED
    reset_tracking()
    return ("",204)


@app.post("/api/target")
def api_target():
    data = request.get_json(force=True) or {}
    t = int(data.get("target", 2000))
    t = max(1, t)
    with lock:
        state["target"] = t
        state["last"] = f"target set to {t}"
    return ("",204)


@app.post("/api/line")
def api_line():
    data = request.get_json(force=True) or {}
    v = float(data.get("line_y_ratio", 0.25))
    v = max(0.05, min(0.95, v))
    with lock:
        state["line_y_ratio"] = v
        state["last"] = f"line set to {v:.2f}"
    return ("",204)


@app.post("/api/record_start")
def record_start():
    global cap
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_recording(w, h)
    with lock:
        state["recording"] = True
        state["last"] = "recording started"
    return ("",204)


@app.post("/api/record_stop")
def record_stop():
    stop_recording()
    with lock:
        state["recording"] = False
        state["last"] = "recording stopped"
    return ("",204)


@app.route("/download_report")
def download_report():
    pdf = create_pdf()
    return send_file(pdf, as_attachment=True)


if __name__ == "__main__":
    try:
        threading.Thread(target=vision_worker, daemon=True).start()
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        GPIO.cleanup()  
