from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)


CORS(app)

UPLOAD_FOLDER = "uploads"

MODEL_DIR = "saved_model"

print(f"Model directory: {os.path.abspath(MODEL_DIR)}")
if os.path.exists(MODEL_DIR):
    print("Directory contents:", os.listdir(MODEL_DIR))
    if "variables" in os.listdir(MODEL_DIR):
        print("Variables directory contents:", os.listdir(os.path.join(MODEL_DIR, "variables")))
else:
    print("ERROR: SavedModel directory not found!")

try:
    imported = tf.saved_model.load(MODEL_DIR)
    print("Available signatures:", list(imported.signatures.keys()))
    predict_fn = imported.signatures["serving_default"]      
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load model:", e)
    raise

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print("SAVED")
        img = Image.open(filepath).convert("RGB").resize((256, 182))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        print("Input tensor shape:", input_tensor.shape)
        print("Input tensor ")
        outputs = predict_fn(input_tensor)
        print("Output keys:", outputs.keys())
        predictions = next(iter(outputs.values())).numpy()
        predicted_class = np.argmax(predictions)
        
        class_names = [
            "Normal",
            "Myocardial Infarction", 
            "History of MI",
            "Abnormal Heartbeat",
        ]
        result = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

        recommendation = {
            "Normal": "Your ECG appears normal. Maintain a healthy lifestyle.",
            "Myocardial Infarction": "Signs of heart attack detected. Seek emergency care immediately.",
            "History of MI": "Signs of past heart attack. Follow up with a cardiologist.",
            "Abnormal Heartbeat": "Irregular heartbeat detected. Please consult a doctor.",
        }

        return jsonify({
            "prediction": result,
            "recommendation": recommendation[result]
        }), 200

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)