import os
import sys
import tensorflow as tf

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dataset import verify_dataset, get_stratified_splits, build_tf_dataset
from model import build_mobilenetv2_model, unfreeze_for_finetuning
from utils import save_class_indices, plot_training_history


def train_pipeline(data_dir=None, epochs_phase1=15, epochs_phase2=15, batch_size=16):
    print("==================================================")
    print("       NUTMEG HARVEST NCD4 - TRAINING PIPELINE    ")
    print("==================================================\n")

    # Step 1: Verify dataset
    dataset_dir, image_paths, image_labels, class_to_idx = verify_dataset(data_dir)

    # Save class indices mapping
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    class_indices_path = os.path.join(models_dir, "class_indices.json")
    save_class_indices(class_to_idx, class_indices_path)

    # Step 2: Stratified Split (70% Train, 15% Val, 15% Test)
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = get_stratified_splits(
        image_paths, image_labels
    )

    # Save test set paths for evaluation script
    test_split_info = {"paths": test_p.tolist(), "labels": test_l.tolist()}
    import json

    with open(os.path.join(models_dir, "test_split.json"), "w") as f:
        json.dump(test_split_info, f, indent=4)

    # Step 3: Build TensorFlow tf.data pipelines
    train_ds = build_tf_dataset(
        train_p, train_l, batch_size=batch_size, is_training=True
    )
    val_ds = build_tf_dataset(
        val_p, val_l, batch_size=batch_size, is_training=False
    )

    # Step 4: Build Model (Phase 1 - Base Frozen)
    model, base_model = build_mobilenetv2_model(
        num_classes=4, input_shape=(224, 224, 3), learning_rate=1e-3
    )

    model_save_path = os.path.join(models_dir, "nutmeg_mobilenetv2.keras")

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("\n--- PHASE 1: Training Classification Head (Base Frozen) ---")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=callbacks_phase1,
    )

    # Step 5: Phase 2 - Fine-Tuning Deeper Layers
    print("\n--- PHASE 2: Fine-Tuning Deeper MobileNetV2 Layers ---")
    model = unfreeze_for_finetuning(
        model, base_model, fine_tune_at=100, learning_rate=1e-5
    )

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1 + epochs_phase2,
        initial_epoch=history_phase1.epoch[-1] + 1 if history_phase1.epoch else 0,
        callbacks=callbacks_phase2,
    )

    # Combine History for Plotting
    combined_history = {}
    for key in history_phase1.history.keys():
        combined_history[key] = (
            history_phase1.history[key] + history_phase2.history[key]
        )

    print("\n[INFO] Saving training history plots...")
    plot_training_history(combined_history, results_dir)
    print("\n[SUCCESS] Training pipeline completed successfully!")


if __name__ == "__main__":
    train_pipeline()
