import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '2'
os.environ["XLA_FLAGS"] = (
    "--xla_dump_to=./generated --xla_dump_hlo_as_dot --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_hlo_pass_re=.*"
)

import tensorflow as tf

# (kMultiply (kCustomCall:gemm A B) C) is folding C (provided it's a constant)
# into an alpha parameter of the custom call.
@tf.function(jit_compile=True)
def matmul_demo():
    m1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], tf.float32)
    m2 = tf.constant([[5.0, 6.0], [7.0, 8.0]], tf.float32)
    c = tf.constant(3.0, tf.float32)
    return c * tf.matmul(m1, m2)

res = matmul_demo()
print(res)
