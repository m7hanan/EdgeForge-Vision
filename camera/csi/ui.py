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

Gst.init(None)

# GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.output(7, GPIO.LOW)

state = {
    "running": False,
    "recording": False,
    "count": 0,
    "target": 2000,
    "line_y_ratio": 0.25,
    "roi": {"x1":142,"y1":82,"x2":466,"y2":478}
}

lock = threading.Lock()
latest_jpeg = None
frame_lock = threading.Lock()

model = YOLO(MODEL_PATH)
video_writer = None

last_y = {}
counted_ids = set()


# ---------------- CAMERA ----------------
def start_camera():
    pipeline = Gst.parse_launch(
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
    )
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

    frame = np.ndarray((h, w, 3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8)

    return frame.copy()


# ---------------- RECORD ----------------
def start_recording(w, h):
    global video_writer
    path = os.path.join(RECORD_DIR, datetime.now().strftime("%Y%m%d_%H%M%S")+".mp4")
    video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))


def stop_recording():
    global video_writer
    if video_writer:
        video_writer.release()
        video_writer = None


# ---------------- VISION ----------------
def vision_worker():
    global latest_jpeg

    pipeline, sink = start_camera()

    while True:
        frame = get_frame(sink)
        if frame is None:
            continue

        h, w, _ = frame.shape

        with lock:
            running = state["running"]
            recording = state["recording"]
            line_ratio = state["line_y_ratio"]
            roi = state["roi"]
            target = state["target"]

        rx1, ry1, rx2, ry2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

        # draw ROI
        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(255,0,255),2)

        line_y = int(h * line_ratio)
        cv2.line(frame,(0,line_y),(w,line_y),(0,255,255),2)

        if running:
            results = model.track(frame, persist=True, conf=CONF)
            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)

                for (x1,y1,x2,y2), tid in zip(boxes, ids):

                    cx = int((x1+x2)/2)
                    cy = int((y1+y2)/2)

                    # draw box ONLY inside ROI
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)

                    # tracking always active
                    if tid not in last_y:
                        last_y[tid] = cy
                        continue

                    prev_y = last_y[tid]
                    last_y[tid] = cy

                    # ✅ COUNT ONLY when inside ROI
                    if (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):

                        if prev_y > line_y+5 and cy <= line_y-5 and tid not in counted_ids:

                            with lock:
                                state["count"] += 1
                                current_count = state["count"]

                            counted_ids.add(tid)

                            print("COUNT:", current_count)

                            # GPIO trigger
                            if current_count >= target:
                                GPIO.output(7, GPIO.HIGH)

        # display
        with lock:
            text = f"COUNT: {state['count']}/{state['target']}"

        cv2.putText(frame, text, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        if recording and video_writer:
            video_writer.write(frame)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

        with frame_lock:
            latest_jpeg = buf.tobytes()

        time.sleep(0.01)


# ---------------- STREAM ----------------
def gen_stream():
    while True:
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def status():
    with lock:
        return jsonify(state)


@app.post("/api/start")
def start():
    with lock:
        state["running"] = True
    return ("",204)


@app.post("/api/stop")
def stop():
    with lock:
        state["running"] = False
    return ("",204)


@app.post("/api/reset")
def reset():
    with lock:
        state["count"] = 0

    last_y.clear()
    counted_ids.clear()
    GPIO.output(7, GPIO.LOW)

    return ("",204)


@app.post("/api/target")
def set_target():
    data = request.get_json(force=True)
    with lock:
        state["target"] = int(data.get("target",2000))
    return ("",204)


@app.post("/api/line")
def set_line():
    data = request.get_json(force=True)
    with lock:
        state["line_y_ratio"] = float(data.get("line_y_ratio",0.25))
    return ("",204)


@app.post("/api/record_start")
def rec_start():
    start_recording(640,480)
    with lock:
        state["recording"] = True
    return ("",204)


@app.post("/api/record_stop")
def rec_stop():
    stop_recording()
    with lock:
        state["recording"] = False
    return ("",204)


# ---------------- CLEANUP ----------------
def cleanup():
    GPIO.output(7, GPIO.LOW)
    GPIO.cleanup()

atexit.register(cleanup)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=vision_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
