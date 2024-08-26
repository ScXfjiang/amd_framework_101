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

__global__ void AddArrays(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { c[idx] = a[idx] + b[idx]; }
}

int main() {
  int n = 1024;
  size_t size = n * sizeof(float);

  // allocate memory on the host (CPU)
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);
  for (int i = 0; i < n; i++) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  // allocate memory on the device (GPU)
  float *d_a, *d_b, *d_c;
  HIP_CHECK(hipMalloc((void **)&d_a, size));
  HIP_CHECK(hipMalloc((void **)&d_b, size));
  HIP_CHECK(hipMalloc((void **)&d_c, size));

  // copy from host to device
  HIP_CHECK(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));

  hipEvent_t start, stop;
  float elapsed_time;

  // create events
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // record the start event
  HIP_CHECK(hipEventRecord(start, 0));

  // kernel launch
  int num_threads = 256;
  int num_blocks = (n + num_threads - 1) / num_threads;
  AddArrays<<<num_blocks, num_threads>>>(d_a, d_b, d_c, n);

  // record the stop event
  HIP_CHECK(hipEventRecord(stop, 0));

  // wait for the stop event to complete
  HIP_CHECK(hipEventSynchronize(stop));

  // calculate elapsed time
  HIP_CHECK(hipEventElapsedTime(&elapsed_time, start, stop));

  std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  // copy from device to host
  HIP_CHECK(hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost));

  // verify the result
  bool success = true;
  for (int i = 0; i < n; i++) {
    if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
      success = false;
      std::cout << "Error at index " << i << ": " << h_c[i] << " != " << h_a[i] + h_b[i]
                << std::endl;
      break;
    }
  }
  if (success) { std::cout << "Arrays added successfully!" << std::endl; }

  // free device memory
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  // free host memory
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}