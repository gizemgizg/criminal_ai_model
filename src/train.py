# src/train.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from preprocess import load_dataset

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    X, y = load_dataset('data')
    X = X.reshape(-1, 64, 64, 1)
    y = to_categorical(y, num_classes=2)

    model = create_model()
    model.fit(X, y, epochs=10, validation_split=0.2)
    model.save('saved_model/sabikali_model.h5')

if __name__ == "__main__":
    train_model()
