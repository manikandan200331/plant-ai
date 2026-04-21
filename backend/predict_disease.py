import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(
    os.path.join(BASE_DIR, "model", "plant_model.h5"),
    compile=False
)
# Class labels
classes = [
    "tomato_healthy",
    "tomato_early_blight",
    "tomato_late_blight",
    "potato_healthy",
    "potato_early_blight",
    "potato_late_blight",
    "pepper_bell_healthy",
    "pepper_bell_bacterial_spot"
]

def predict_plant(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128,128))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    pred = model.predict(x,verbose=0)
    class_index = np.argmax(pred)
    plant_disease = classes[class_index]
    confidence = float(pred[0][class_index] * 100)

    # Affected percentage
    img_cv = cv2.imread(img_path)

    if img_cv is None:
       print("Image not loaded properly")
       return "error", 0, 0, "Image read failed", "-", "-"
    
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
    total_pixels = img_cv.shape[0] * img_cv.shape[1]
    affected_pixels = np.sum(thresh == 255)
    affected_percent = float((affected_pixels/total_pixels)*100)
    if "healthy" in plant_disease.lower():  # fix healthy leaf
        affected_percent = 0

    # Reason & remedies
    reasons = { 
        "tomato_healthy": "Leaf is healthy. No disease symptoms detected.",
        "tomato_early_blight": "Fungal disease caused by Alternaria solani. Appears as brown spots with rings.",
        "tomato_late_blight": "Caused by Phytophthora infestans. Spreads quickly in wet and cool weather.",
        "potato_healthy": "Leaf is healthy. Plant growth is normal.",
        "potato_early_blight": "Fungal infection caused by Alternaria solani due to warm humid conditions.",
        "potato_late_blight": "Disease caused by Phytophthora infestans during rainy or cool climate.",
        "pepper_bell_healthy": "No disease detected. Plant leaf is healthy.",
        "pepper_bell_bacterial_spot": "Bacterial infection caused by Xanthomonas. Spread through water splashes."
    }
    reason = reasons.get(plant_disease.lower(), "Reason not available")

    remedies = {
        "tomato_healthy": {"organic":"No treatment required.", "chemical":"No chemical needed."},
        "tomato_early_blight": {"organic":"Remove infected leaves & spray neem oil.", "chemical":"Apply Mancozeb or Chlorothalonil."},
        "tomato_late_blight": {"organic":"Remove infected parts & improve air circulation.", "chemical":"Apply copper fungicide."},
        "potato_healthy": {"organic":"Maintain irrigation.", "chemical":"No chemical required."},
        "potato_early_blight": {"organic":"Remove infected leaves & spray neem oil.", "chemical":"Use Mancozeb."},
        "potato_late_blight": {"organic":"Remove infected plants.", "chemical":"Apply copper fungicide."},
        "pepper_bell_healthy": {"organic":"Maintain good soil nutrients.", "chemical":"No chemical required."},
        "pepper_bell_bacterial_spot": {"organic":"Remove infected leaves & avoid overhead watering.", "chemical":"Apply copper bactericide."}
    }
    remedy_data = remedies.get(plant_disease.lower(), {"organic":"Consult expert","chemical":"Consult expert"})
    organic_remedy = remedy_data["organic"]
    chemical_remedy = remedy_data["chemical"]

    return plant_disease, round(confidence,2), round(affected_percent,2), reason, organic_remedy, chemical_remedy