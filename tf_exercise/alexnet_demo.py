import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_alexnet(input_shape=(224, 224, 3), num_classes=10):
    model = models.Sequential()

    # Layer 1: Convolutional, Batch Normalization, ReLU, and Max Pooling
    model.add(
        layers.Conv2D(
            96, kernel_size=11, strides=4, padding="same", input_shape=input_shape
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=3, strides=2))

    # Layer 2: Convolutional, Batch Normalization, ReLU, and Max Pooling
    model.add(layers.Conv2D(256, kernel_size=5, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=3, strides=2))

    # Layer 3: Convolutional, Batch Normalization, ReLU (no pooling here)
    model.add(layers.Conv2D(384, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    # Layer 4: Convolutional, Batch Normalization, ReLU (no pooling here)
    model.add(layers.Conv2D(384, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    # Layer 5: Convolutional, Batch Normalization, ReLU, and Max Pooling
    model.add(layers.Conv2D(256, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=3, strides=2))

    # Flattening the layers and adding dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
