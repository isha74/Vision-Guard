# helper 
import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

MODEL_PATH = "face_model.pkl"
LABELS_PATH = "labels.pkl"

def load_images_from_folder(dataset_path):
    X, y = [], []
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100))  # resize for uniformity
            X.append(img.flatten())  # flatten to vector
            y.append(person_name)
    return np.array(X), np.array(y)

def train_model(dataset_path="Dataset"):
    print("[INFO] Loading dataset...")
    X, y = load_images_from_folder(dataset_path)

    print("[INFO] Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("[INFO] Training model...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X, y_enc)

    print("[INFO] Saving model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABELS_PATH)
    print("[INFO] Training complete!")

def load_model():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABELS_PATH)
    return model, le

def predict_face(image_path):
    model, le = load_model()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100)).flatten().reshape(1, -1)
    pred = model.predict(img)[0]
    proba = model.predict_proba(img).max()
    return le.inverse_transform([pred])[0], proba
