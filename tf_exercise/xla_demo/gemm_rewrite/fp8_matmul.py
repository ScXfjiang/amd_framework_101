import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'
os.environ['HIP_VISIBLE_DEVICES'] = '2'
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./generated "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_html "
    "--xla_dump_hlo_pass_re=.*"
)

import tensorflow as tf

@tf.function(jit_compile=True)
def fp8_matmul(x_fp8, y_fp8, scale_x, scale_y, scale_z):
    # 1. dequantize x_fp8 and y_fp8
    x_fp16_unscaled = tf.cast(x_fp8, tf.float16) * scale_x
    y_fp16_unscaled = tf.cast(y_fp8, tf.float16) * scale_y
    # 2. perform matmul in fp16
    z_fp16_unscaled = tf.matmul(x_fp16_unscaled, y_fp16_unscaled)
    # 3. quantize z_fp16 and calculate new scale
    z_fp8 = tf.cast(z_fp16_unscaled / scale_z, tf.dtypes.experimental.float8_e4m3fnuz)
    max_e4m3fnuz = 240
    new_z_scale = max_e4m3fnuz / tf.reduce_max(tf.abs(z_fp16_unscaled))
    return z_fp8, new_z_scale


if __name__ == "__main__":
    x_fp8 = tf.constant([[1.0, 2.0], [3.0, 4.0]], tf.dtypes.experimental.float8_e4m3fnuz)
    y_fp8 = tf.constant([[5.0, 6.0], [7.0, 8.0]], tf.dtypes.experimental.float8_e4m3fnuz)
    z_fp8, new_z_scale = fp8_matmul(x_fp8, y_fp8, 2.0, 2.0, 2.0)
    print(z_fp8)
    print(new_z_scale)
