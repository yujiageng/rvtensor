/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/avpooling.hpp"

namespace RVTensor {

CPUAVPoolingOp::sptr CPUAVPoolingOp::create() {
  return std::make_shared<CPUAVPoolingOp>();
}

CPUAVPoolingOp::sptr CPUAVPoolingOp::create(RamTensor::sptr input,
                                            RamTensor::sptr output) {
  CPUAVPoolingOp::sptr ptr = std::make_shared<CPUAVPoolingOp>(input, output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUAVPoolingOp::CPUAVPoolingOp() : Operation({}, {}) {}

inline CPUAVPoolingOp::CPUAVPoolingOp(RamTensor::sptr input,
                                      RamTensor::sptr output)
    : Operation({input}, {output}) {}

inline CPUAVPoolingOp::~CPUAVPoolingOp() {}

inline void CPUAVPoolingOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  float* input = reinterpret_cast<float*>(input_tensor->data_ptr);
  float* output = reinterpret_cast<float*>(output_tensor->data_ptr);

  // TODO: complete it

  int b, i, k;
  int c = input_tensor->channel;
  int h = input_tensor->height;
  int w = input_tensor->width;
  int batch = input_tensor->n_batch;
  // 8 * 8 *64
  for (int b = 0; b < batch; b++) {
    for (k = 0; k < c; ++k) {
      int out_index = k + b * batch;
      output[out_index] = 0;
      for (i = 0; i < h * w; ++i) {
        int in_index = i + h * w * (k + b * batch);
        output[out_index] += input[in_index];
      }
      output[out_index] /= h * w;
    }
  }
}
}  // namespace RVTensor
