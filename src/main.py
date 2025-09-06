import sys
import cv2
from utils import train_model, predict_face
from detection import realtime_detection

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py [train | predict <image_path> | realtime]")
        return

    command = sys.argv[1]

    if command == "train":
        train_model("Dataset")
    elif command == "predict" and len(sys.argv) == 3:
        image_path = sys.argv[2]
        name, proba = predict_face(image_path)
        print(f"Predicted: {name} (confidence: {proba:.2f})")
    elif command == "realtime":
        realtime_detection()
    else:
        print("Invalid command.")
        
def predict_face(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img = cv2.resize(img, (100, 100)).flatten().reshape(1, -1)
    ...

if __name__ == "__main__":
    main()
