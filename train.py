import argparse
import os

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model


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

def get_data_generators(dataset_dir, batch_size, img_size):
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.6, 1.4],
        fill_mode='nearest'
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen

def train(args):
    img_size = (128, 128)
    train_gen, val_gen = get_data_generators(args.dataset, args.batch_size, img_size)

    model = build_model(input_shape=(img_size[0], img_size[1], 3))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        LearningRateScheduler(lr_warmup_schedule, verbose=1),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            args.save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset directory with train/val subfolders")
    parser.add_argument("--save_path", default="model/apple_model.h5", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    train(args)
