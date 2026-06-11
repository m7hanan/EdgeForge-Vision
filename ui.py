from flask import Flask, jsonify, request, render_template, Response, send_file
import threading, time, os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from reportlab.pdfgen import canvas
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import cv2
import Jetson.GPIO as GPIO
import atexit

app = Flask(__name__)

MODEL_PATH = "models/best.engine"
CONF = 0.20

RECORD_DIR = "record"
os.makedirs(RECORD_DIR, exist_ok=True)

# Initialize GStreamer Core Core Libraries
Gst.init(None)

# Configure Physical Jetson Control Pins Natively
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.output(7, GPIO.LOW)

state = {
    "running": False,
    "recording": False,
    "count": 0,
    "target": 2000,
    "line_y_ratio": 0.25,
    "last": "System Idle",
    "roi": {"x1": 142, "y1": 82, "x2": 466, "y2": 478}
}

lock = threading.Lock()
latest_jpeg = None
frame_lock = threading.Lock()

# Initialize TensorRT compiled model weights explicitly for detection task
model = YOLO(MODEL_PATH, task='detect')
video_writer = None

last_y = {}
counted_ids = set()


# ---------------- HEADLESS HEADLESS GSTREAMER C270 USB PIPELINE ----------------
def start_camera():
    # Tuned specifically for SSH/headless environments. 
    # Bypasses local display checks by routing frames directly into an internal appsink memory buffer.
    gst_str = (
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
        "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true"
    )
    pipeline = Gst.parse_launch(gst_str)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    return pipeline, sink


def get_frame(sink):
    sample = sink.emit("pull-sample")
    if sample is None:
        return None
        
    buf = sample.get_buffer()
    caps = sample.get_caps()
    
    h = caps.get_structure(0).get_value('height')
    w = caps.get_structure(0).get_value('width')
    
    # Directly extract frame allocations from memory mapping bounds
    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        return None
        
    # Generate frame and copy immediately to preserve memory safety bounds
    frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8).copy()
    buf.unmap(map_info)
    return frame


# ---------------- OUTPUT MEDIA WRITERS ----------------
def start_recording(w, h):
    global video_writer
    path = os.path.join(RECORD_DIR, datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4")
    video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

def stop_recording():
    global video_writer
    if video_writer:
        video_writer.release()
        video_writer = None


# ---------------- MAIN VISION WORKER THREAD ----------------
def vision_worker():
    global latest_jpeg
    try:
        pipeline, sink = start_camera()
    except Exception as e:
        print(f"❌ Critical Failure: GStreamer cannot open Logitech C270 device nodes: {e}")
        return

    while True:
        frame = get_frame(sink)
        if frame is None:
            time.sleep(0.01)
            continue

        h, w, _ = frame.shape
        with lock:
            running = state["running"]
            recording = state["recording"]
            line_ratio = state["line_y_ratio"]
            roi = state["roi"]
            target = state["target"]

        rx1, ry1, rx2, ry2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

        # Render Active Regions and counting limits
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
        line_y = int(h * line_ratio)
        cv2.line(frame, (rx1, line_y), (rx2, line_y), (0, 255, 255), 2)

        if running:
            # Process frames via native Tracking weights targeting Orin Nano GPU (Device 0)
            results = model.track(frame, persist=True, conf=CONF, verbose=False)
            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), tid in zip(boxes, ids):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    if tid not in last_y:
                        last_y[tid] = cy
                        continue

                    prev_y = last_y[tid]
                    last_y[tid] = cy

                    # Industrial Line Crossing Validation Matrix (Top-to-Bottom Tracking Direction)
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        if prev_y > line_y and cy <= line_y and tid not in counted_ids:
                            with lock:
                                state["count"] += 1
                                state["last"] = datetime.now().strftime("%H:%M:%S")
                                current_count = state["count"]

                            counted_ids.add(tid)
                            print(f"🎯 COCONUT STACKED: {current_count}")

                            # Fire physical relay trigger when complete batch is logged
                            if current_count >= target:
                                GPIO.output(7, GPIO.HIGH)

        if recording and video_writer:
            video_writer.write(frame)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_jpeg = buf.tobytes()
        time.sleep(0.01)


# ---------------- WEB HOST ROUTING CONTROLLERS ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(gen_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_stream():
    while True:
        with frame_lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.get("/api/status")
def status():
    with lock:
        return jsonify(state)

@app.post("/api/start")
def start():
    with lock:
        state["running"] = True
    return ("", 204)

@app.post("/api/stop")
def stop():
    with lock:
        state["running"] = False
    return ("", 204)

@app.post("/api/reset")
def reset():
    with lock:
        state["count"] = 0
        state["last"] = "Batch Cleared"
    last_y.clear()
    counted_ids.clear()
    GPIO.output(7, GPIO.LOW)
    return ("", 204)

@app.post("/api/target")
def set_target():
    data = request.get_json(force=True)
    with lock:
        state["target"] = int(data.get("target", 2000))
    return ("", 204)

@app.post("/api/line")
def set_line():
    data = request.get_json(force=True)
    with lock:
        state["line_y_ratio"] = float(data.get("line_y_ratio", 0.25))
    return ("", 204)

@app.post("/api/record_start")
def rec_start():
    start_recording(640, 480)
    with lock:
        state["recording"] = True
    return ("", 204)

@app.post("/api/record_stop")
def rec_stop():
    stop_recording()
    with lock:
        state["recording"] = False
    return ("", 204)


# ---------------- DOCUMENT EXTRACTION ENGINE ----------------
@app.route("/download_report")
def download_report():
    pdf_path = "industrial_report.pdf"
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "INDUSTRIAL COCONUT COUNTER PRODUCTION REPORT")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with lock:
        c.drawString(50, 680, f"Total Counted Quantity: {state['count']} units")
        c.drawString(50, 660, f"Target Setpoint Limit: {state['target']} units")
        c.drawString(50, 640, f"Last Logged Event: {state['last']}")
    c.save()
    return send_file(pdf_path, as_attachment=True)

def cleanup():
    try:
        GPIO.output(7, GPIO.LOW)
        GPIO.cleanup()
    except:
        pass

atexit.register(cleanup)

if __name__ == "__main__":
    threading.Thread(target=vision_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)