# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# import numpy as np
# import json
#
# # Load trained model
# model = load_model("flower_classifier_40class_10epochs_32batch.h5")
#
# IMG_SIZE = 224
#
# BATCH_SIZE = 32
#
# train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
#     r'C:\ai_project\codes\flower_data\train',
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )
#
# def predict_flower(img_path):
#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img = image.img_to_array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#
#     prediction = model.predict(img)
#     class_idx = np.argmax(prediction)
#     class_labels = list(train_gen.class_indices.keys())
#
#     return class_labels[class_idx], round(float(prediction[0][class_idx]), 2)
#
#
# # Example
# # print(predict_flower(r"C:\ai_project\codes\images_testing\orchids\orchids_00029.jpg"))

######################################## NEW CODE FOR API ############################################



# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
#
# IMG_SIZE = 224
# BATCH_SIZE = 32
#
# # Load trained model
# model = load_model("flower_classifier_40class_10epochs_32batch.h5")
#
# # Recreate your training generator to get class labels
# train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
#     r'C:\ai_project\codes\flower_data\train',
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )
#
# # Extract class labels in correct order
# class_labels = list(train_gen.class_indices.keys())
#
# def predict_flower(img_path):
#     """
#     Predicts flower class and returns (predicted_label, confidence)
#     """
#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#
#     prediction = model.predict(img_array)
#     class_idx = np.argmax(prediction)
#
#     predicted_label = class_labels[class_idx]
#     confidence = float(prediction[0][class_idx])
#
#     return predicted_label, confidence
#
#
#
# # flower_name, score = predict_flower(r"C:\ai_project\codes\flower_data\test\banana\banana11.jpg")
# # print(flower_name)
# # print(score)
#############################################################################below trail
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32

# Load trained model
model = load_model("flower_classifier_40class_10epochs_32batch.h5")

# Recreate your training generator to get class labels
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    r'C:\ai_project\codes\flower_data\train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Extract class labels in correct order
class_labels = list(train_gen.class_indices.keys())

def predict_flower(img_path):
    """
    Predicts flower class and returns (predicted_label, confidence)
    Also formats label for JSON lookup (lowercase, underscores)
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # Get predicted label and confidence
    predicted_label = class_labels[class_idx]
    confidence = float(prediction[0][class_idx])

    # Format label for JSON lookup (replace spaces with underscores, lowercase)
    formatted_label = predicted_label.lower().replace(" ", "_")

    return formatted_label, confidence

