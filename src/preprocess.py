
import cv2
import sys
import dlib
import os
import numpy as np

face_detector = dlib.get_frontal_face_detector()

def preprocess_image(file_path):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) > 0:
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (64, 64))
            return face_img / 255.0
    return None

def load_dataset(dataset_path):
    images = []
    labels = []
    for label, category in enumerate(['sabikali', 'sabikasiz']):
        category_path = os.path.join(dataset_path, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            img = preprocess_image(file_path)
            if img is not None:
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)
