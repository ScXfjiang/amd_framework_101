import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('./demo.h5')
predictions = model.predict(np.random.random((10, 224, 224, 3)))
labels = np.argmax(predictions, axis=1)
print(labels)