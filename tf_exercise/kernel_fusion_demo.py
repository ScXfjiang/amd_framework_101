import os

os.environ["TF_CPP_VMODULE"] = "gpu_fusion_pass=2"  # set log level
os.environ["TF_ROCM_FUSION_ENABLE"] = "1"  # enable kernel fusion


import tensorflow as tf
import numpy as np

tf.get_logger().setLevel(5)

use_nhwc = True

N, H, W, C = 1, 2, 3, 3
k, r, s, c = 3, 2, 2, C

if use_nhwc:
    x_format = "NHWC"
    x_format_keras = "channels_last"
    bias_format = "N...C"
    x_shape = (N, H, W, C)
    channel_axis = -1
else:
    x_format = "NCHW"
    x_format_keras = "channels_first"
    bias_format = "NC..."
    x_shape = (N, C, H, W)
    channel_axis = 1

f_np = np.random.random([r, s, c, k]).astype(np.float32)
f = tf.Variable(f_np)
b_np = np.random.random([k]).astype(np.float32)
b = tf.Variable(b_np)


@tf.function(jit_compile=True)
def fused_conv_bias_relu(x):
    y = tf.nn.conv2d(x, f, strides=(1, 1), padding="SAME", data_format=x_format)
    y = tf.nn.bias_add(y, b, data_format=bias_format)
    y = tf.nn.relu(y)
    return y


inputs = tf.random.normal(x_shape)
outputs = fused_conv_bias_relu(inputs)
print(outputs)
