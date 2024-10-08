import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./generated --xla_dump_hlo_as_dot --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_hlo_pass_re=.*"
)

import tensorflow as tf

# print("test nanoo fp8 tensor creation...")
# with tf.device('/GPU:0'):
#     x = tf.constant([1.0, 2.0, 3.0], tf.dtypes.experimental.float8_e4m3fnuz)
#     print(x)

# print("cast op - cast from fp32 to nanoo fp8...")
# with tf.device('/GPU:0'):
#     x = tf.constant([1.0, 2.0, 3.0], tf.float32)
#     y = tf.cast(x, tf.dtypes.experimental.float8_e4m3fnuz)
#     print(y)
#     z = tf.cast(y, tf.float32)
#     print(z)

print("matmul demo")
@tf.function(jit_compile=True)
def matmul_demo():
    m1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], tf.dtypes.experimental.float8_e4m3fnuz)
    m2 = tf.constant([[5.0, 6.0], [7.0, 8.0]], tf.dtypes.experimental.float8_e4m3fnuz)
    return tf.matmul(m1, m2)

res = matmul_demo()
print(res)
