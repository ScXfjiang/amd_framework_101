#include <hip/hip_runtime.h>
#include <iostream>

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

  // allocate memory on the host (CPU)
  float a[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
  float b[n] = {10.0, 10.0, 10.0, 10.0, 10.0};
  float c[n] = {0};

  // allocate memory on the device (GPU)
  float *d_a = 0;
  float *d_b = 0;
  float *d_c = 0;
  HIP_CHECK(hipMalloc((void **)&d_c, n * sizeof(float)));
  HIP_CHECK(hipMalloc((void **)&d_a, n * sizeof(float)));
  HIP_CHECK(hipMalloc((void **)&d_b, n * sizeof(float)));

  // copy data from host to device
  HIP_CHECK(hipMemcpy(d_a, a, n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, b, n * sizeof(float), hipMemcpyHostToDevice));

  // launch the kernel
  AddKernel<<<1, n>>>(d_c, d_a, d_b, n);

  // copy result back to host
  HIP_CHECK(hipMemcpy(c, d_c, n * sizeof(int), hipMemcpyDeviceToHost));

  printf("Results:\n");
  for (int i = 0; i < n; i++) { printf("%f + %f = %f\n", a[i], b[i], c[i]); }
  
  HIP_CHECK(hipFree(d_c));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));

  return 0;
}
