# src/predict.py

from tensorflow.keras.models import load_model
from preprocess import preprocess_image

def predict_image(file_path):
    model = load_model('saved_model/sabikali_model.h5')
    img = preprocess_image(file_path)
    if img is not None:
        img = img.reshape(1, 64, 64, 1)
        prediction = model.predict(img)
        sabikali_prob = prediction[0][0]
        sabikasiz_prob = prediction[0][1]
        print(f'Sabıkalı olma olasılığı: {sabikali_prob*100:.2f}%')
        print(f'Sabıkasız olma olasılığı: {sabikasiz_prob*100:.2f}%')
    else:
        print("Yüz algılanamadı.")

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    predict_image(file_path)
