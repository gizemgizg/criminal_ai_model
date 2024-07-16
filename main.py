# main.py

import sys
from src import train, evaluate, predict

def main():
    if len(sys.argv) < 2:
        print("Usage: main.py [train|evaluate|predict] [file_path (for predict)]")
        return

    command = sys.argv[1]

    if command == "train":
        train.train_model()
    elif command == "evaluate":
        evaluate.evaluate_model()
    elif command == "predict" and len(sys.argv) == 3:
        file_path = sys.argv[2]
        predict.predict_image(file_path)
    else:
        print("Invalid command or missing file path for predict.")

if __name__ == "__main__":
    main()
