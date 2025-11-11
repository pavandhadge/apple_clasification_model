import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
    SeparableConv2D,
)
from tensorflow.keras.models import Sequential


def fast_preprocessing(image):
    """
    Lightweight preprocessing - much faster than complex CLAHE.
    Only basic normalization for speed.
    """
    # Convert to float and normalize to [0, 1]
    img = image.astype(np.float32) / 255.0

    # Simple adaptive brightness normalization
    mean_val = np.mean(img)

    if mean_val > 0.7:  # Too bright
        img = img * 0.85
    elif mean_val < 0.3:  # Too dark
        img = img * 1.2

    img = np.clip(img, 0, 1)

    return img

def build_model(input_shape=(128, 128, 3)):
    """
    Streamlined model that trains faster while maintaining accuracy.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Block 2
        SeparableConv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        SeparableConv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 4
        SeparableConv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.35),

        # Classification
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    return model

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None

def predict(model, image):
    img = cv2.resize(image, (128, 128))
    img = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img, axis=0)
    pred = model.predict(img_array)
    return pred
