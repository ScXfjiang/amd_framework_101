import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from demo_cnn import build_demo_cnn

model = build_demo_cnn()
model.compile(
    optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# use random data for demonstration
train_images = np.random.random((100, 224, 224, 3))
train_labels = np.random.randint(10, size=(100,))
train_labels = tf.keras.utils.to_categorical(train_labels, 10)

test_images = np.random.random((20, 224, 224, 3))
test_labels = np.random.randint(10, size=(20,))
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# training
model.fit(
    train_images,
    train_labels,
    batch_size=32,
    epochs=10,
    validation_data=(test_images, test_labels),
)

model.save('./demo.h5')

# inference
new_images = np.random.random((10, 224, 224, 3))  # Simulate 10 new images
predictions = model.predict(new_images)
labels = np.argmax(predictions, axis=1)
print(labels)