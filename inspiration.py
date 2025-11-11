import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============ GPU SETUP ============
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU found, using CPU.")

# ============ PATHS ============
dataset_dir = '/content/drive/MyDrive/appledataset'
model_save_path = '/content/drive/MyDrive/apple_quality_fast.h5'

# ============ OPTIMIZED SETTINGS ============
img_height, img_width = 128, 128
batch_size = 32  # Increased from 16 for faster training

# ============ VIDEO SMOOTHING ============
window = deque(maxlen=7)

def smoothed_prediction(new_pred):
    window.append(new_pred)
    avg = np.mean(window, axis=0)
    return np.argmax(avg)

# ============ SIMPLIFIED FAST PREPROCESSING ============
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

# ============ SIMPLIFIED AUGMENTATION ============
# Removed heavy preprocessing for faster training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Simple rescaling instead of custom preprocessing
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],  # Reduced range
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

# ============ DATA LOADERS WITH MULTIPROCESSING ============
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'valid')
test_dir = os.path.join(dataset_dir, 'test')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Check data balance
print("\n📊 Dataset Information:")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Class distribution: {train_generator.class_indices}")

# Check for class imbalance
class_counts = np.bincount(train_generator.classes)
print(f"Training class distribution: {class_counts}")
if max(class_counts) / min(class_counts) > 2:
    print("⚠️ WARNING: Significant class imbalance detected!")

# ============ SIMPLER, FASTER MODEL ============
def build_fast_model():
    """
    Streamlined model that trains faster while maintaining accuracy.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=(img_height, img_width, 3)),
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

# ============ LEARNING RATE WARMUP ============
def lr_warmup_schedule(epoch, lr):
    """
    Learning rate warmup to prevent early divergence.
    Starts low, increases to target, then decays.
    """
    if epoch < 3:  # Warmup phase
        return 0.0001 * (10 ** (epoch / 2))
    elif epoch < 10:  # High learning rate phase
        return 0.01
    elif epoch < 20:  # Decay phase 1
        return 0.001
    else:  # Decay phase 2
        return 0.0001

# ============ TRAIN MODEL ============
if os.path.exists(model_save_path):
    print("\n📦 Loading existing model...")
    model = load_model(model_save_path)
else:
    print("\n🚀 Building fast training model...")
    model = build_fast_model()

    print("\n📊 Model Summary:")
    model.summary()
    print(f"\n💾 Parameters: {model.count_params():,}")

    # Compile with HIGHER initial learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Will be adjusted by scheduler
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # ============ CALLBACKS ============
    lr_scheduler = LearningRateScheduler(lr_warmup_schedule, verbose=1)

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # ============ TRAINING ============
    epochs = 25

    print(f"\n🏋️ Training with warmup schedule...")
    print(f"Batch size: {batch_size}")
    print(f"Expected time per epoch after first: 1-2 minutes")
    print(f"⏰ First epoch will be slower (10-20x) due to compilation - this is NORMAL\n")

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[lr_scheduler, early_stop, checkpoint],
        verbose=1,
        # removed workers and use_multiprocessing arguments
        # workers=4,  # Use 4 CPU cores for data loading
        # use_multiprocessing=True,  # Speed up data loading
        # max_queue_size=16  # Buffer size
    )

    print(f"\n✅ Training complete!")

    # Show final performance
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n📈 Final Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"📈 Final Val Accuracy: {final_val_acc*100:.2f}%")
    print(f"📊 Train/Val Gap: {(final_train_acc - final_val_acc)*100:.2f}%")

# ============ EVALUATION ============
print("\n" + "="*60)
print("📊 TEST SET EVALUATION")
print("="*60)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

if test_acc < 0.70:
    print("\n⚠️ Low accuracy. Possible issues:")
    print("  1. Class imbalance - check distribution above")
    print("  2. Images too similar between classes")
    print("  3. Mislabeled data")
    print("  4. Need more diverse training data")

# ============ FAST PREDICTION FUNCTIONS ============
def predict_image(image_path):
    """Fast image prediction with simplified preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0  # Simple normalization
    img_array = np.expand_dims(img, axis=0)

    pred = model.predict(img_array, verbose=0)
    confidence = np.max(pred)
    class_idx = np.argmax(pred)
    label = list(train_generator.class_indices.keys())[class_idx]

    print(f"\n🖼️  {os.path.basename(image_path)}")
    print(f"   → {label} ({confidence:.1%})")

    return label, confidence

def predict_video(video_path, frame_skip=10):
    """Fast video prediction."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return None

    predictions = []
    window.clear()
    frame_count = 0

    print(f"\n🎥 {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (img_width, img_height))
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            frame_array = np.expand_dims(frame_normalized, axis=0)

            pred = model.predict(frame_array, verbose=0)
            smooth_class = smoothed_prediction(pred[0])
            predictions.append(smooth_class)

        frame_count += 1

    cap.release()

    if not predictions:
        print("⚠️ No frames")
        return None

    final_class = np.bincount(predictions).argmax()
    label = list(train_generator.class_indices.keys())[final_class]

    print(f"   → {label} ({len(predictions)} frames)")

    return label

# ============ USAGE ============
print("\n" + "="*60)
print("✅ READY FOR INFERENCE")
print("="*60)
print("\nUsage:")
print("  predict_image('path/to/image.jpg')")
print("  predict_video('path/to/video.mp4')")
print("="*60)
