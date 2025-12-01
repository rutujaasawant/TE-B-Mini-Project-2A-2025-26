
#########################################################################dontknow
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from flask import Response
# ------------------- CONFIG -------------------
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "flower_classifier_40class_10epochs_32batch.h5"


# ------------------- LOAD MODEL -------------------
model = load_model(MODEL_PATH)

# Load class labels from training directory
train_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
    r'C:\ai_project\codes\flower_data\train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
class_labels = list(train_gen.class_indices.keys())


# Load flower info JSON
JSON_INFO_PATH = "flower_info.json"
with open(JSON_INFO_PATH, "r") as f:
    flower_info = json.load(f)

# Load flower cure JSON
FLOWER_CURE_PATH = "flowercure.json"
with open(FLOWER_CURE_PATH, "r") as f:
    flower_cure_info = json.load(f)

# ------------------- FEATURE EXTRACTION MODEL -------------------
input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = model.layers[0](input_tensor)  # MobileNetV2 base
for layer in model.layers[1:-1]:
    x = layer(x)
feature_model = Model(inputs=input_tensor, outputs=x)

# Precompute embeddings for all flowers for visual similarity
flower_embeddings = {}
for flower in class_labels:
    class_dir = os.path.join(r'C:\ai_project\codes\flower_data\train', flower)
    img_files = os.listdir(class_dir)
    if img_files:
        img_path = os.path.join(class_dir, img_files[0])
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        emb = feature_model.predict(img_array)
        flower_embeddings[flower] = emb

# ------------------- FLASK APP -------------------
app = Flask(__name__)
CORS(app)  # Enable CORS
db = mysql.connector.connect(
    host="localhost",
    user="root",             # your MySQL username
    password="mysql@root",  # your MySQL password
    database="floralq_db"      # the database you created
)
cursor = db.cursor()
# ------------------- HELPER FUNCTIONS -------------------
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    predicted_label = class_labels[class_idx]
    confidence = float(prediction[0][class_idx])  # convert to Python float
    return predicted_label, confidence

def get_visual_similar(img_path, top_n=5):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    emb = feature_model.predict(img_array)

    sims = {}
    for flower, f_emb in flower_embeddings.items():
        sim = float(cosine_similarity(emb, f_emb)[0][0])  # convert to Python float
        sims[flower] = sim

    top_flowers = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"class": f[0], "similarity": f[1]} for f in top_flowers]

def get_property_similar(predicted_label, top_n=5):
    """Return flowers with at least 3 matching care properties as objects (class only)."""
    target_props = flower_info[predicted_label].get("care", {})
    sims = []
    for flower, info in flower_info.items():
        if flower == predicted_label:
            continue
        flower_props = info.get("care", {})
        match_count = sum(1 for key in target_props if key in flower_props and target_props[key] == flower_props[key])
        if match_count >= 3:
            sims.append({"class": flower})
    if not sims:  # No matching flowers
       sims.append({"class": "No matching property flowers found"})
    return sims[:top_n]

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    # try:
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")

    if not name or not email or not message:
        return jsonify({"error": "All fields are required"}), 400

    query = "INSERT INTO feedback (name, email, message) VALUES (%s, %s, %s)"
    cursor.execute(query, (name, email, message))
    db.commit()

    return jsonify({"message": "Feedback submitted successfully!"})
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

# ------------------- ROUTES -------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    temp_path = os.path.join("temp.jpg")
    file.save(temp_path)

    # try:
    predicted_label, confidence = predict_flower(temp_path)
    info = flower_info.get(predicted_label, {})

    # Visual similarity
    visual_similar = get_visual_similar(temp_path)

    # Property similarity
    property_similar = get_property_similar(predicted_label)

    # Flower Cure info
    care_info = flower_cure_info.get(predicted_label, {})
    print("predicted_label :", predicted_label)
    response = {
        "label": predicted_label,
        "confidence": confidence,
        "info": info,
        "care_info": care_info,
        "similar_classes": visual_similar,
        "property_similar": property_similar
    }

    return jsonify(response)
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    # finally:
    #     if os.path.exists(temp_path):
    #         os.remove(temp_path)

# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True)
