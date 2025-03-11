from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

MODEL_PATH = "weights.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    print(f"✅ Model class names: {model.names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_image():
    if not model:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file selected."}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.jpg")
    file.save(filepath)
    print(f"✅ Image saved: {filepath}")

    if not os.path.exists(filepath):
        return jsonify({"error": "File save failed."}), 500

    try:
        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            return jsonify({"error": "Failed to read image with OpenCV."}), 400

        results = model.predict(image_bgr, save=False)
        predictions = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                conf = round(float(conf), 2)
                label = model.names[int(cls)] if cls in model.names else f"class_{int(cls)}"
                predictions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf, "class": int(cls), "label": label})

                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text = f"{label} ({conf:.2f})"
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
                text_bg_x = x1
                text_bg_y = text_y - text_size[1] - 5
                text_bg_w = text_size[0] + 10
                text_bg_h = text_size[1] + 10
                cv2.rectangle(image_bgr, (text_bg_x, text_bg_y), (text_bg_x + text_bg_w, text_bg_y + text_bg_h), (0, 0, 0), -1)
                cv2.putText(image_bgr, text, (text_x + 5, text_y + text_size[1] - 5), font, font_scale, color, font_thickness)

        processed_img_path = os.path.join(app.config["PROCESSED_FOLDER"], "processed.jpg")
        cv2.imwrite(processed_img_path, image_bgr)
        print(f"✅ Processed image saved: {processed_img_path}")
        return jsonify({"image_url": f"/processed-image?t={int(os.path.getmtime(processed_img_path))}", "predictions": predictions})

    except Exception as e:
        print(f"❌ Processing error: {e}")
        return jsonify({"error": f"Image processing error: {e}"}), 500

@app.route("/processed-image")
def get_processed_image():
    return send_from_directory(app.config["PROCESSED_FOLDER"], "processed.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)