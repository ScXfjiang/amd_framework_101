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

__global__ void LoopKernel() {
  for (int i = 0; i < 1000; ++i) {}
}

int main() {
  hipEvent_t start, stop;
  float elapsed_time;

  // create events
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // record the start event
  HIP_CHECK(hipEventRecord(start, 0));

  // lanuch kernel
  LoopKernel<<<1, 1>>>();

  // record the stop event
  HIP_CHECK(hipEventRecord(stop, 0));

  // Wait for the stop event to complete
  HIP_CHECK(hipEventSynchronize(stop));

  // calculate elapsed time
  HIP_CHECK(hipEventElapsedTime(&elapsed_time, start, stop));

  std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

  return 0;
}