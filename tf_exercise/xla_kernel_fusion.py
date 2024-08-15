import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
os.environ['XLA_FLAGS'] = '--xla_dump_to=./generated'
os.environ['TF_DUMP_GRAPH_PREFIX'] = './generated'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_clustering_debug'

import tensorflow as tf
tf.debugging.set_log_device_placement(True)

@tf.function(jit_compile=True)
def fused_conv_bias_relu(x):
    y = tf.nn.conv2d(x, tf.random.normal((3, 3, 3, 64)), strides=(1, 1), padding="SAME", data_format="NCHW")
    y = tf.nn.bias_add(y, tf.random.normal((64,)), data_format="NCHW")
    y = tf.nn.relu(y)
    return y

x = tf.random.normal((10, 3, 128, 128))
tf.profiler.experimental.start('logdir')
y = fused_conv_bias_relu(x)
tf.profiler.experimental.stop()
print(y.shape)
