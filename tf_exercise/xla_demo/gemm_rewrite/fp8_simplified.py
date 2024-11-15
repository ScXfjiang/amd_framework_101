import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'
os.environ['HIP_VISIBLE_DEVICES'] = '2'
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./generated "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_html "
    "--xla_dump_hlo_pass_re=.*"
    # "--xla_gpu_gemm_rewrite_size_threshold=0"
)

import tensorflow as tf
import numpy as np

@tf.function(jit_compile=True)
def fp8_matmul(x_fp8, y_fp8, scale_x, scale_y):
    # 1. dequantize x_fp8 and y_fp8 to fp16
    x_fp16_unscaled = tf.cast(x_fp8, tf.float16) * scale_x
    y_fp16_unscaled = tf.cast(y_fp8, tf.float16) * scale_y
    # 2. perform matmul in fp16
    z_fp16_unscaled = tf.matmul(x_fp16_unscaled, y_fp16_unscaled)
    return z_fp16_unscaled

if __name__ == "__main__":
    x_fp16 = tf.random.uniform([16, 32], dtype=tf.float16)
    y_fp16 = tf.random.uniform([32, 16], dtype=tf.float16)
    # initialize x_fp8 and y_fp8
    x_fp8 = tf.cast(x_fp16, tf.dtypes.experimental.float8_e4m3fnuz)
    y_fp8 = tf.cast(y_fp16, tf.dtypes.experimental.float8_e4m3fnuz)
    # initialize scale_x and scale_y
    scale_x = tf.constant(2.0, dtype=tf.float16)
    scale_y = tf.constant(2.0, dtype=tf.float16)
    # do fp8 matmul
    z_fp8 = fp8_matmul(x_fp8, y_fp8, scale_x, scale_y)
    print(z_fp8)
