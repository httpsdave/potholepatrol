from flask import Flask, request, render_template, jsonify, send_file
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import io
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Folder setup
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load trained YOLO model
model = YOLO("weights.pt")  # Ensure this is your trained model

# Route for Main page
@app.route("/")
def index():
    return render_template("index.html")

# Route for analyzing images
@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    # Read image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image_np = np.array(image)  # Convert to NumPy array (H, W, 3) format
    
    # Ensure correct format for YOLO
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR if needed

    # Run YOLO inference
    results = model.predict(image_bgr, save=False)

    predictions = []
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        if boxes is None:
            continue  # Skip if no detections

        for box in boxes.data.tolist():
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
            conf = round(float(box[4]), 2)  # Confidence score
            cls = int(box[5])  # Class index

            # Ensure correct way to get class name
            label = model.names[cls] if cls in model.names else f"class_{cls}"
            
            print(f"Detected: {label} with confidence {conf}")

            # Only process pothole class
            if label.lower() == "pothole":  
                predictions.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf, "class": cls, "label": label
                })

                # Draw bounding box
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image_np, f"{label} ({conf})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, "processed.png")
    cv2.imwrite(processed_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    return jsonify({"image_url": "/processed_image", "predictions": predictions})

# Route to serve processed image
@app.route("/processed_image")
def processed_image():
    processed_image_path = os.path.join(PROCESSED_FOLDER, "processed.png")
    return send_file(processed_image_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
