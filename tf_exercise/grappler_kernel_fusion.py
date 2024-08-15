import os
# enable kernel fusion for rocm
os.environ["TF_ROCM_FUSION_ENABLE"] = "1"
# set log level for gpu_fusion_pass
os.environ["TF_CPP_VMODULE"] = "gpu_fusion_pass=2"

import tensorflow as tf

@tf.function()
def fused_conv_bias_relu(x):
    y = tf.nn.conv2d(x, tf.random.normal((3, 3, 3, 64)), strides=(1, 1), padding="SAME", data_format="NCHW")
    y = tf.nn.bias_add(y, tf.random.normal((64,)), data_format="NCHW")
    y = tf.nn.relu(y)
    return y

# graph execution error
# No registered '_ROCmFusedConvolutionBiasActivation' OpKernel
if False:
    x = tf.random.normal((10, 3, 128, 128))
    y = fused_conv_bias_relu(x)
    print(y.shape)


@tf.function()
def fused_conv_bias(x):
    y = tf.nn.conv2d(x, tf.random.normal((3, 3, 3, 64)), strides=(1, 1), padding="SAME", data_format="NCHW")
    y = tf.nn.bias_add(y, tf.random.normal((64,)), data_format="NCHW")
    return y

x = tf.random.normal((10, 3, 128, 128))
y = fused_conv_bias(x)
print(y.shape)
