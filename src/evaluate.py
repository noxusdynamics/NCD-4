import os
import sys
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dataset import build_tf_dataset
from utils import load_class_indices, plot_confusion_matrix


def evaluate_model():
    print("==================================================")
    print("      NUTMEG HARVEST NCD4 - MODEL EVALUATION      ")
    print("==================================================\n")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", "nutmeg_mobilenetv2.keras")
    test_split_path = os.path.join(project_root, "models", "test_split.json")
    class_indices_path = os.path.join(project_root, "models", "class_indices.json")
    results_dir = os.path.join(project_root, "results")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first!")

    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Test split file not found at {test_split_path}.")

    # Load Model & Test Split Info
    print(f"[INFO] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    with open(test_split_path, "r") as f:
        test_split = json.load(f)

    test_paths = np.array(test_split["paths"])
    y_true = np.array(test_split["labels"])

    class_to_idx, idx_to_class = load_class_indices(class_indices_path)
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    # Build test dataset (unaugmented, unshuffled)
    test_ds = build_tf_dataset(test_paths, y_true, batch_size=16, is_training=False)

    print("[INFO] Evaluating on unseen test set...")
    predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred, target_names=target_names, digits=4
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n--------------------------------------------------")
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print("--------------------------------------------------\n")
    print("Classification Report:\n")
    print(report_str)

    # Save Classification Report
    os.makedirs(results_dir, exist_ok=True)
    report_file_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write("Nutmeg Harvest NCD4 - MobileNetV2 Test Set Evaluation\n")
        f.write("====================================================\n\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"[INFO] Saved classification report to {report_file_path}")

    # Plot & Save Confusion Matrix
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, target_names, cm_path)

    print("\n[SUCCESS] Evaluation complete!")
    return acc, report_str, cm


if __name__ == "__main__":
    evaluate_model()
