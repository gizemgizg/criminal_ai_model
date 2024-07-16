# src/evaluate.py

from tensorflow.keras.models import load_model
from preprocess import load_dataset

def evaluate_model():
    X, y = load_dataset('data')
    X = X.reshape(-1, 64, 64, 1)
    y = to_categorical(y, num_classes=2)

    model = load_model('saved_model/sabikali_model.h5')
    loss, accuracy = model.evaluate(X, y)
    print(f'Model Accuracy: {accuracy*100:.2f}%')

if __name__ == "__main__":
    evaluate_model()
