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

__global__ void hello_world() {
  unsigned int block_idx = blockIdx.x;
  unsigned int thread_idx = threadIdx.x;
  printf("Hello World from Thread Block %u Thread %u!\n", block_idx, thread_idx);
}

int main() {
  hello_world<<<2,                // # of thread blocks
                5,                // # of threads within each thread block
                0,                // shared memory
                hipStreamDefault  // gpu stream
                >>>();
  // wait for compute device to finish
  HIP_CHECK(hipDeviceSynchronize());
}