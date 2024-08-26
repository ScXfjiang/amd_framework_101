#include <hip/hip_runtime.h>
#include <iostream>

__global__ void SineKernel(float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { output[idx] = __sinf(input[idx]); }
}

int main() {
  int n = 1024;
  float *input, *output;
  float *d_input, *d_output;

  // Allocate host memory
  input = new float[n];
  output = new float[n];

  // Initialize input data
  for (int i = 0; i < n; i++) {
    input[i] = i * 0.1f;  // Example values
  }

  // Allocate device memory
  hipMalloc(&d_input, n * sizeof(float));
  hipMalloc(&d_output, n * sizeof(float));

  // Copy data from host to device
  hipMemcpy(d_input, input, n * sizeof(float), hipMemcpyHostToDevice);

  // Launch the kernel
  SineKernel<<<(n + 255) / 256, 256>>>(d_input, d_output, n);

  // Copy result back to host
  hipMemcpy(output, d_output, n * sizeof(float), hipMemcpyDeviceToHost);

  // Print some of the results
  for (int i = 0; i < 10; i++) {
    std::cout << "sin(" << input[i] << ") = " << output[i] << std::endl;
  }

  // Free device memory
  hipFree(d_input);
  hipFree(d_output);

  // Free host memory
  delete[] input;
  delete[] output;

  return 0;
}
