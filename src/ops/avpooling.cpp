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

void CPUAVPoolingOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  float* input = reinterpret_cast<float*>(input_tensor->data_ptr);
  float* output = reinterpret_cast<float*>(output_tensor->data_ptr);

  int ci = input_tensor->channel;
  int hi = input_tensor->height;
  int wi = input_tensor->width;
  int ni = input_tensor->n_batch;
  int co = output_tensor->channel;
  int ho = output_tensor->height;
  int wo = output_tensor->width;
  int no = output_tensor->n_batch;
  assert((ni == no) && (ci == co) && (ho == 1) && (wo == 1));

  for (int n = 0; n < no; n++) {
      for (int c = 0; c < co; c++) {
         int out_index = n * co + c;
         output[out_index] = 0;
         for (int i = 0; i < hi * wi; i++) {
             int in_index = n * ci * hi * wi + c * hi * wi + i;
             output[out_index] += input[in_index];
         }
         output[out_index] /= hi * wi;
      }
  }

}


}  // namespace RVTensor
