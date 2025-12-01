from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from collections import defaultdict
import os
import heapq

# ----------------------------
# Parameters
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_DIR = r'C:\ai_project\codes\flower_data\train'
MODEL_PATH = "flower_classifier_40class_10epochs_32batch.h5"
FEATURES_FILE = "train_image_features.npy"  # file to save embeddings
PATHS_FILE = "train_image_paths.npy"  # file to save image paths

# ----------------------------
# Load Sequential model
# ----------------------------
seq_model = load_model(MODEL_PATH)

# ----------------------------
# Correct feature extraction
# ----------------------------
# MobileNetV2 base is the first layer in your Sequential model
mobilenet_base = seq_model.layers[0]

# GlobalAveragePooling2D layer is the second layer in your Sequential model
gap_layer = seq_model.layers[1]

# Create functional input
input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Pass input through MobileNetV2 + GAP
x = mobilenet_base(input_tensor)
features = gap_layer(x)

# Feature extraction model
feature_model = Model(inputs=input_tensor, outputs=features)

# ----------------------------
# Get class labels
# ----------------------------
train_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
class_labels = list(train_gen.class_indices.keys())

# ----------------------------
# Precompute or load features
# ----------------------------
if os.path.exists(FEATURES_FILE) and os.path.exists(PATHS_FILE):
    print("Loading precomputed features...")
    image_features = np.load(FEATURES_FILE)
    image_paths = np.load(PATHS_FILE)
else:
    print("Extracting features for all training images...")
    image_features_list = []
    image_paths_list = []
    for class_name in os.listdir(TRAIN_DIR):
        class_path = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            feat = feature_model.predict(img_array)
            image_features_list.append(feat.flatten())
            image_paths_list.append(img_path)

    image_features = np.array(image_features_list)
    image_paths = np.array(image_paths_list)

    np.save(FEATURES_FILE, image_features)
    np.save(PATHS_FILE, image_paths)
    print("Feature extraction completed and saved!")


# ----------------------------
# Predict flower class
# ----------------------------
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = seq_model.predict(img_array)
    class_idx = np.argmax(prediction)

    predicted_label = class_labels[class_idx]
    confidence = float(prediction[0][class_idx])

    return predicted_label, confidence


# ----------------------------
# Suggest visually similar flowers
# ----------------------------
# def get_similar_flowers(img_path, top_n=5):
#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#
#     query_features = feature_model.predict(img_array).flatten().reshape(1, -1)
#
#     similarities = cosine_similarity(query_features, image_features)[0]
#
#     top_indices = similarities.argsort()[-top_n:][::-1]
#     top_similar = [(os.path.basename(image_paths[i]), similarities[i]) for i in top_indices]
#
#     return top_similar
def get_similar_flowers_by_class(img_path, top_n=5):
    """
    Returns top N visually similar flower classes with average similarity
    """
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features
    query_features = feature_model.predict(img_array).flatten().reshape(1, -1)

    # Compute similarity with all training images
    similarities = cosine_similarity(query_features, image_features)[0]

    # Aggregate similarities by class
    class_sim_dict = defaultdict(list)
    for path, sim in zip(image_paths, similarities):
        class_name = os.path.basename(os.path.dirname(path))
        class_sim_dict[class_name].append(sim)

    # Compute average similarity per class
    class_avg_sim = {cls: np.mean(sims) for cls, sims in class_sim_dict.items()}

    # Sort classes by similarity
    top_classes = sorted(class_avg_sim.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_classes


# ----------------------------

# ----------------------------
# Combined function: predict + suggest
# ----------------------------
# def predict_and_suggest(img_path, top_n=5):
#     predicted_label, confidence = predict_flower(img_path)
#     similar_flowers = get_similar_flowers(img_path, top_n=top_n)
#     return predicted_label, confidence, similar_flowers
#
def predict_and_suggest_by_class(img_path, top_n=5):
    predicted_label, confidence = predict_flower(img_path)
    similar_classes = get_similar_flowers_by_class(img_path, top_n=top_n)
    return predicted_label, confidence, similar_classes

# ----------------------------
# Example usage
# ----------------------------
# img_path = r"C:\ai_project\codes\flower_data\test\rose\rose1.jpg"
# predicted_label, confidence, similar_flowers = predict_and_suggest(img_path, top_n=5)
#
# print(f"Predicted Flower: {predicted_label} (Confidence: {confidence:.2f})")
# print("Top 5 visually similar flowers:")
# for fname, score in similar_flowers:
#     print(f"{fname} -> Similarity: {score:.2f}")

# img_path = r"C:\ai_project\codes\flower_data\test\rose\rose1.jpg"
# predicted_label, confidence, similar_classes = predict_and_suggest_by_class(img_path, top_n=5)
#
# print(f"Predicted Flower: {predicted_label} (Confidence: {confidence:.2f})")
# print("Top visually similar flower classes:")
# for cls, score in similar_classes:
#     print(f"{cls} -> Average Similarity: {score:.2f}")