import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'
os.environ['HIP_VISIBLE_DEVICES'] = '2'
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./generated "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_html "
    # "--xla_dump_hlo_pass_re=.*"
)

import tensorflow as tf
import numpy as np

@tf.function(jit_compile=True)
def fp8_matmul(x_fp8, y_fp8, scale_x, scale_y, scale_z):
    # 1. dequantize x_fp8 and y_fp8 to fp16
    x_fp16_unscaled = tf.cast(x_fp8, tf.float16) * scale_x
    y_fp16_unscaled = tf.cast(y_fp8, tf.float16) * scale_y
    # 2. perform matmul in fp16
    z_fp16_unscaled = tf.matmul(x_fp16_unscaled, y_fp16_unscaled)
    z_max = tf.reduce_max(tf.abs(z_fp16_unscaled))
    # 3. quantize z_fp16 to fp8
    z_fp8 = tf.cast(z_fp16_unscaled / scale_z, tf.dtypes.experimental.float8_e4m3fnuz)
    return z_fp8, z_max

@tf.function(jit_compile=False)
def fp16_matmul(x_fp16, y_fp16):
    return tf.matmul(x_fp16, y_fp16)

if __name__ == "__main__":
    x_fp16 = tf.constant([[2.0, 2.0], [2.0, 2.0]], tf.float16)
    y_fp16 = tf.constant([[4.0, 4.0], [4.0, 4.0]], tf.float16)
    z_fp16 = fp16_matmul(x_fp16, y_fp16)
    print(z_fp16)

    x_fp8 = tf.cast(x_fp16, tf.dtypes.experimental.float8_e4m3fnuz)
    y_fp8 = tf.cast(y_fp16, tf.dtypes.experimental.float8_e4m3fnuz) 
    scale_x = tf.constant(2.0, dtype=tf.float16)
    scale_y = tf.constant(2.0, dtype=tf.float16)
    scale_z = tf.constant(4.0, dtype=tf.float16)
    z_fp8, z_max = fp8_matmul(x_fp8, y_fp8, scale_x, scale_y, scale_z)
    print(z_fp8)

    assert(np.allclose(z_fp16, tf.cast(z_fp8, tf.float16)))
