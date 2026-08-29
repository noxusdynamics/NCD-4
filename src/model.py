import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model


def build_mobilenetv2_model(
    num_classes=4, input_shape=(224, 224, 3), learning_rate=1e-3
):
    """Build MobileNetV2 classification model with base frozen initially."""
    # Base pretrained model
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze base model parameters initially
    base_model.trainable = False

    # Input layer
    inputs = Input(shape=input_shape)

    # Base model output
    x = base_model(inputs, training=False)

    # Classification head
    x = GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x = Dropout(0.3, name="top_dropout_1")(x)
    x = Dense(128, activation="relu", name="dense_128")(x)
    x = Dropout(0.2, name="top_dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Nutmeg_MobileNetV2")

    # Compile with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[INFO] Built MobileNetV2 model with frozen base.")
    return model, base_model


def unfreeze_for_finetuning(model, base_model, fine_tune_at=100, learning_rate=1e-5):
    """Unfreeze deeper layers of MobileNetV2 base model and recompile with small learning rate."""
    base_model.trainable = True

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    total_layers = len(base_model.layers)
    trainable_layers = total_layers - fine_tune_at
    print(
        f"[INFO] Unfrozen deeper MobileNetV2 base layers from index {fine_tune_at} to {total_layers} ({trainable_layers} trainable layers)."
    )

    # Recompile model with very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
