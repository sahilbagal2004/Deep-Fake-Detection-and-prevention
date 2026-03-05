import cv2
import numpy as np
import tensorflow as tf
import os

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(frame):
    frame = cv2.resize(frame, (224,224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def detect_video(path):

    # read image
    frame = cv2.imread(path)

    if frame is None:
        return "Error reading image"

    frame = preprocess(frame)

    prediction = model.predict(frame)[0][0]

    print("Prediction Score:", prediction)

    # classification
    if prediction >= 0.5:
        return "Real Image"
    else:
        return "Fake Image"