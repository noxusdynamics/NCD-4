import os
import glob
import hashlib
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Standard 4 Class Order
CLASS_NAMES = ["Harvested", "Not Harvested", "Decayed", "Intermediate"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Map directory folder names (case-insensitive) to official class names
FOLDER_TO_CLASS = {
    "harvest_img": "Harvested",
    "harvested": "Harvested",
    "harvest_img/": "Harvested",
    "not_harvested_img": "Not Harvested",
    "not_harvested": "Not Harvested",
    "decayed_img": "Decayed",
    "decayed": "Decayed",
    "intermediate_img": "Intermediate",
    "intermediate": "Intermediate",
}


def find_dataset_dir(custom_path=None):
    """Find dataset root directory containing class subfolders."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frames", "M1")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frames", "m1")),
        r"C:\Users\LOQ\OneDrive\Desktop\nutmeg_harvest_ncd4\frames\m1",
        r"C:\Users\LOQ\OneDrive\Desktop\Nutmeg_Harvest_NCD4\Frames\M1",
    ]

    for p in possible_paths:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not find dataset directory. Please verify folder location."
    )


def verify_dataset(dataset_dir=None):
    """Perform dataset verification: check path, folder existence, counts, corruption, duplicates."""
    dataset_dir = find_dataset_dir(dataset_dir)
    print("Dataset Verification")
    print("---------------------")
    print(f"Path: {dataset_dir}")

    subdirs = [
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    class_counts = {c: 0 for c in CLASS_NAMES}
    image_paths = []
    image_labels = []
    corrupted_files = []
    md5_hashes = {}
    duplicates = []
    supported_formats = {".jpg", ".jpeg", ".png", ".bmp"}

    for subdir in subdirs:
        norm_name = subdir.lower()
        if norm_name in FOLDER_TO_CLASS:
            official_class = FOLDER_TO_CLASS[norm_name]
        else:
            continue

        cpath = os.path.join(dataset_dir, subdir)
        files = [
            f for f in os.listdir(cpath) if os.path.splitext(f)[1].lower() in supported_formats
        ]

        for fname in files:
            fpath = os.path.join(cpath, fname)
            # Verify readability
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception:
                corrupted_files.append(fpath)
                continue

            # Verify duplicates via MD5
            try:
                with open(fpath, "rb") as fp:
                    h = hashlib.md5(fp.read()).hexdigest()
                    if h in md5_hashes:
                        duplicates.append((fpath, md5_hashes[h]))
                    else:
                        md5_hashes[h] = fpath
            except Exception:
                pass

            class_counts[official_class] += 1
            image_paths.append(fpath)
            image_labels.append(CLASS_TO_INDEX[official_class])

    for cname in CLASS_NAMES:
        print(f"{cname:<16}: {class_counts[cname]}")

    total_images = sum(class_counts.values())
    print(f"Total           : {total_images}")

    if corrupted_files:
        print(f"[WARNING] Corrupted/Unreadable Images: {len(corrupted_files)}")
    if duplicates:
        print(f"[INFO] Duplicate Images Found: {len(duplicates)}")

    print("---------------------\n")
    return dataset_dir, image_paths, image_labels, CLASS_TO_INDEX


def get_stratified_splits(image_paths, image_labels, seed=42):
    """Perform 70% Train, 15% Validation, 15% Test stratified split."""
    image_paths = np.array(image_paths)
    image_labels = np.array(image_labels)

    # First split: 70% Train, 30% Temp (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, image_labels, test_size=0.30, random_state=seed, stratify=image_labels
    )

    # Second split: Split 30% Temp equally into 15% Val and 15% Test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )

    print(
        f"[INFO] Dataset Split -> Train: {len(train_paths)} (70%), Val: {len(val_paths)} (15%), Test: {len(test_paths)} (15%)"
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def load_and_preprocess_image(path, label, img_size=(224, 224)):
    """Load image, resize, convert RGB, and apply MobileNetV2 preprocess_input."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def create_data_augmentation_layer():
    """Create mild data augmentation layer for training."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),  # ~15 degrees max
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )


def build_tf_dataset(paths, labels, batch_size=16, is_training=False, img_size=(224, 224)):
    """Build tf.data.Dataset pipeline with preprocessing and optional training augmentation."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=42)

    dataset = dataset.map(
        lambda p, l: load_and_preprocess_image(p, l, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if is_training:
        aug_layer = create_data_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
