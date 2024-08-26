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

__global__ void SqrtKernel(float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { output[idx] = __builtin_amdgcn_sqrt(input[idx]); }
}

int main() {
  int n = 1024;
  size_t size = n * sizeof(float);

  // allocate memory on the host (CPU)
  float *h_input = (float *)malloc(size);
  float *h_output = (float *)malloc(size);
  for (int i = 0; i < n; i++) { h_input[i] = i; }

  // allocate memory on the device (GPU)
  float *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, n * sizeof(float)));

  // copy data from host to device
  HIP_CHECK(hipMemcpy(d_input, h_input, n * sizeof(float), hipMemcpyHostToDevice));

  // launch the kernel
  SqrtKernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);

  // copy result back to host
  HIP_CHECK(hipMemcpy(h_output, d_output, n * sizeof(float), hipMemcpyDeviceToHost));

  // verify the result
  for (int i = 0; i < 10; i++) {
    std::cout << "sin(" << h_input[i] << ") = " << h_output[i] << std::endl;
  }

  // free device memory
  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  // free host memory
  free(h_input);
  free(h_output);

  return 0;
}
