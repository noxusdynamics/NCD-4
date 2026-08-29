import os
import json
import matplotlib.pyplot as plt
import numpy as np


def save_class_indices(class_indices, filepath):
    """Save class name to index mapping dictionary as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(class_indices, f, indent=4)
    print(f"[INFO] Saved class indices to {filepath}")


def load_class_indices(filepath):
    """Load class mapping JSON and return (class_to_index, index_to_class)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Class indices file not found at: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    # data is {"Harvested": 0, "Not Harvested": 1, "Decayed": 2, "Intermediate": 3}
    class_to_idx = {str(k): int(v) for k, v in data.items()}
    idx_to_class = {int(v): str(k) for k, v in data.items()}
    return class_to_idx, idx_to_class


def plot_training_history(history_dict, save_dir):
    """Plot and save Training & Validation Accuracy and Loss graphs."""
    os.makedirs(save_dir, exist_ok=True)

    acc = history_dict.get("accuracy", [])
    val_acc = history_dict.get("val_accuracy", [])
    loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, "bo-", label="Training Accuracy")
    plt.plot(epochs, val_acc, "ro-", label="Validation Accuracy")
    plt.title("Nutmeg MobileNetV2 - Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved accuracy plot to {acc_path}")

    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, "bo-", label="Training Loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation Loss")
    plt.title("Nutmeg MobileNetV2 - Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved loss plot to {loss_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix heatmap using Matplotlib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Nutmeg Harvest NCD4", fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                weight="bold",
            )

    plt.tight_layout()
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {save_path}")
