import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import load_class_indices


def predict_image(image_path, model_path=None, class_indices_path=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if model_path is None:
        model_path = os.path.join(project_root, "models", "nutmeg_mobilenetv2.keras")

    if class_indices_path is None:
        class_indices_path = os.path.join(project_root, "models", "class_indices.json")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image path does not exist: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first!")

    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Class mapping file not found at: {class_indices_path}")

    # Load Model & Class Index Mapping
    model = tf.keras.models.load_model(model_path)
    class_to_idx, idx_to_class = load_class_indices(class_indices_path)

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    input_tensor = np.expand_dims(img_preprocessed, axis=0)

    # Predict
    probabilities = model.predict(input_tensor, verbose=0)[0]
    top_class_idx = int(np.argmax(probabilities))
    top_class_name = idx_to_class[top_class_idx]
    top_confidence = float(probabilities[top_class_idx] * 100.0)

    # Print Formatted Results
    print(f"\nPredicted Class : {top_class_name}")
    print(f"Confidence      : {top_confidence:.2f}%\n")
    print("Class Probabilities")
    print("-------------------")

    for i in sorted(idx_to_class.keys()):
        cname = idx_to_class[i]
        prob = float(probabilities[i] * 100.0)
        print(f"{cname:<16}: {prob:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Predict Nutmeg Condition using MobileNetV2")
    parser.add_argument("--image", type=str, required=True, help="Path to input nutmeg image")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .keras file")
    args = parser.parse_args()

    predict_image(args.image, model_path=args.model)


if __name__ == "__main__":
    main()
