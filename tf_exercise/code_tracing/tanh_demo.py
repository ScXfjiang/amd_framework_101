import tensorflow as tf

x = tf.random.uniform((2, 2))
y = tf.keras.activations.tanh(x)
print(y)