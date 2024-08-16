import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=./generated --xla_dump_hlo_as_dot'
os.environ['TF_DUMP_GRAPH_PREFIX'] = './generated'

import tensorflow as tf
tf.debugging.set_log_device_placement(True)

@tf.function(jit_compile=True)
def fused_conv_bias_relu(x):
    y = tf.nn.conv2d(x, tf.random.normal((3, 3, 3, 64)), strides=(1, 1), padding="SAME", data_format="NCHW")
    y = tf.nn.bias_add(y, tf.random.normal((64,)), data_format="NCHW")
    y = tf.nn.relu(y)
    return y

x = tf.random.normal((10, 3, 128, 128))
y = fused_conv_bias_relu(x)
print(y.shape)
