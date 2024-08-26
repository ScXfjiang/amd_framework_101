#include <hip/hip_runtime.h>
#include <stdio.h>

constexpr int error_exit_code = -1;

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                      \
  {                                                                                               \
    const hipError_t error = condition;                                                           \
    if (error != hipSuccess) {                                                                    \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " << __FILE__ \
                << ':' << __LINE__ << std::endl;                                                  \
      std::exit(error_exit_code);                                                                 \
    }                                                                                             \
  }

__global__ void AddKernel(float *c, const float *a, const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) { asm volatile("v_add_f32 %0, %1, %2" : "=v"(c[i]) : "v"(a[i]), "v"(b[i])); }
}

int main() {
  const int n = 5;
  float a[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
  float b[n] = {10.0, 10.0, 10.0, 10.0, 10.0};
  float c[n] = {0};

  float *d_a = 0;
  float *d_b = 0;
  float *d_c = 0;

  // Allocate GPU buffers for three vectors (two input, one output)
  hipMalloc((void **)&d_c, n * sizeof(float));
  hipMalloc((void **)&d_a, n * sizeof(float));
  hipMalloc((void **)&d_b, n * sizeof(float));

  // Copy input vectors from host memory to GPU buffers
  hipMemcpy(d_a, a, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_b, b, n * sizeof(float), hipMemcpyHostToDevice);

  // Launch a kernel on the GPU with one thread for each element
  AddKernel<<<1, n>>>(d_c, d_a, d_b, n);

  // Copy output vector from GPU buffer to host memory
  hipMemcpy(c, d_c, n * sizeof(int), hipMemcpyDeviceToHost);

  printf("Results:\n");
  for (int i = 0; i < n; i++) { printf("%f + %f = %f\n", a[i], b[i], c[i]); }

  // Cleanup
  hipFree(d_c);
  hipFree(d_a);
  hipFree(d_b);

  return 0;
}
