import jax
import jax.numpy as jnp
import timeit


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


seed = jax.random.key(2024)
x = jax.random.normal(seed, (1_000_000,))
execution_time = timeit.timeit(lambda: selu(x).block_until_ready(), number=10)
average_time_ms = (execution_time / 10) * 1000
print(f"Average execution time over 10 runs: {average_time_ms:.3f} ms")
