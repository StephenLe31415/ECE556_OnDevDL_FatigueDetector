import cv2
import time
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ================= Configuration =================
EYE_SIZE   = 24
MOUTH_SIZE = 24

EYE_MODEL_PATH   = "eye_model.tflite"
MOUTH_MODEL_PATH = "mouth_model.tflite"

FACE_DETECT_INTERVAL = 4   # run face detection every N frames
INFER_INTERVAL       = 2   # run inference every N frames

# Confidence thresholds — how far above zero_point the raw int8
# score must be before flipping the label (tune these if too sensitive)
EYE_THRESHOLD  = 10
YAWN_THRESHOLD = 2

# =================================================

# ================= Load Models =================
eye_interpreter   = tflite.Interpreter(model_path=EYE_MODEL_PATH)
mouth_interpreter = tflite.Interpreter(model_path=MOUTH_MODEL_PATH)
eye_interpreter.allocate_tensors()
mouth_interpreter.allocate_tensors()

eye_input    = eye_interpreter.get_input_details()
eye_output   = eye_interpreter.get_output_details()
mouth_input  = mouth_interpreter.get_input_details()
mouth_output = mouth_interpreter.get_output_details()

# For Dense(1, sigmoid) quantized to int8:
# zero_point is where probability = 0.5 sits in int8 space
# raw > zero_point  →  prob > 0.5  →  label 1 (Open / No Yawn)
# raw < zero_point  →  prob < 0.5  →  label 0 (Closed / Yawn)
eye_zero_point   = eye_output[0]["quantization"][1]
mouth_zero_point = mouth_output[0]["quantization"][1]

print(f"Eye   zero point : {eye_zero_point}")
print(f"Mouth zero point : {mouth_zero_point}")

# ================= Preprocess =================
def preprocess_eye(img_gray, size, input_details):
    img = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA)

    # grayscale → fake RGB (this matches your eye training)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    scale, zero_point = input_details[0]["quantization"]

    img = img.astype(np.float32) / 255.0
    img = img / scale + zero_point
    img = np.clip(img, -128, 127).astype(np.int8)

    return np.expand_dims(img, axis=0)


def preprocess_mouth(img_rgb, size, input_details):
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

    scale, zero_point = input_details[0]["quantization"]

    img = img.astype(np.float32) / 255.0
    img = img / scale + zero_point
    img = np.clip(img, -128, 127).astype(np.int8)

    return np.expand_dims(img, axis=0)

# ================= Inference =================
def predict_eye(interpreter, input_details, output_details, img_gray, size):
    inp = preprocess_eye(img_gray, size, input_details)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0][0]

def predict_mouth(interpreter, input_details, output_details, img_rgb, size):
    inp = preprocess_mouth(img_rgb, size, input_details)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0][0]

# ================= Camera =================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
picam2.set_controls({
    "AwbEnable": True,
    "AwbMode":   0,     # auto
    "AeEnable":  True
})

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

print("Dual-model drowsiness detection running...")

# ================= Runtime State =================
prev_time   = 0
fps         = 0
alpha       = 0.9
frame_count = 0
faces       = []

# Cached labels (reused between infer frames)
left_label  = "Open"
right_label = "Open"
mouth_label = "No Yawn"

# =================================================

while True:
    frame_count += 1

    # ── FPS ──────────────────────────────────────────────────────
    cur       = time.perf_counter()
    fps       = alpha * fps + (1 - alpha) * (1 / (cur - prev_time) if prev_time else 0)
    prev_time = cur

    frame      = picam2.capture_array()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # ── Face detection in Grayscale (downscaled + interval) ───────────────────
    if frame_count % FACE_DETECT_INTERVAL == 0:
        small = cv2.resize(gray_frame, (320, 240))
        detected = face_cascade.detectMultiScale(small, 1.3, 5)

        if len(detected) > 0:
            scale_x = gray_frame.shape[1] / 320
            scale_y = gray_frame.shape[0] / 240

            faces = [
                (int(x*scale_x), int(y*scale_y),
                int(w*scale_x), int(h*scale_y))
                for (x, y, w, h) in detected
            ]

    # ── Process each face ────────────────────────────────────────
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_frame[y : y+h, x : x+w]
        roi_rgb  = frame[y : y+h, x : x+w]  

        # Crop regions
        upper     = roi_gray[int(h * 0.22):int(h * 0.50), :]
        left_eye  = upper[:, int(w * 0.18):int(w * 0.41)]
        right_eye = upper[:, int(w * 0.60):int(w * 0.80)]

        cv2.imshow("Left Eye", left_eye)
        cv2.imshow("Right Eye", right_eye)

        lower = roi_rgb[int(h * 0.60):int(h * 1.1), :]    # Only use lower half for mouth to avoid noisy upper face
        mouth = lower[:, int(w * 0.25):int(w * 0.85)]

        cv2.imshow("Mouth", mouth)

        # ── Inference (interval) ──────────────────────────────────
        if frame_count % INFER_INTERVAL == 0:
            try:
                # Eyes
                left_raw  = predict_eye(eye_interpreter, eye_input, eye_output, left_eye, EYE_SIZE)
                right_raw = predict_eye(eye_interpreter, eye_input, eye_output, right_eye, EYE_SIZE)

                # Mouth
                mouth_raw = predict_mouth(mouth_interpreter, mouth_input, mouth_output, mouth, MOUTH_SIZE)

                # ===== Eye decision =====
                left_label  = "Open" if (left_raw  - eye_zero_point) > (EYE_THRESHOLD + 20) else "Closed"
                right_label = "Open" if (right_raw - eye_zero_point) > EYE_THRESHOLD else "Closed"

                # ===== Mouth decision =====
                mouth_label = "No Yawn" if (mouth_raw - mouth_zero_point) > YAWN_THRESHOLD else "Yawn"

            except Exception as e:
                print(f"Inference error: {e}")
                continue

        # ── Final decision ────────────────────────────────────────
        eyes_closed = (left_label == "Closed" and right_label == "Closed")
        yawning     = (mouth_label == "Yawn")

        if eyes_closed or yawning:
            status, color = "DROWSY",      (0, 0, 255)
        else:
            status, color = "Alert",       (0, 255, 0)

        # ── Draw ─────────────────────────────────────────────────
        cv2.putText(frame, status,
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"L:{left_label}  R:{right_label}  M:{mouth_label}",
                    (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── FPS overlay ──────────────────────────────────────────────
    cv2.rectangle(frame, (5, 5), (170, 45), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

picam2.stop()
cv2.destroyAllWindows()