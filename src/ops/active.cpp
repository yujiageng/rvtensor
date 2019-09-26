/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */
#include "include/ops/active.hpp"
#include "math.h"

namespace RVTensor {

CPUActiveOp::sptr CPUActiveOp::create() {
  return std::make_shared<CPUActiveOp>();
}

CPUActiveOp::sptr CPUActiveOp::create(ActiveType active_type,
                                      RamTensor::sptr input,
                                      RamTensor::sptr output) {
  CPUActiveOp::sptr ptr =
      std::make_shared<CPUActiveOp>(active_type, input, output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUActiveOp::CPUActiveOp() : Operation({}, {}), param_(ACTIVE_SIGMOID) {}

inline CPUActiveOp::CPUActiveOp(ActiveType active_type, RamTensor::sptr input,
                                RamTensor::sptr output)
    : Operation({input}, {output}), param_(active_type) {}

inline CPUActiveOp::~CPUActiveOp() {}
// 对feature map 进行激活
inline void CPUActiveOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t*>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t*>(output_tensor->data_ptr);

  // TODO: complete it
  if (!param_) {
    softmax(input, input_tensor->trueSize, output);
  } else {
    relu(input, input_tensor->trueSize, output);
  }
}
void relu(uint8_t* inputs, int n, uint8_t* outputs) {
  int i;
  for (i = 0; i < n; ++i) {
    if (inputs[i] > 0) {
      outputs[i] = inputs[i];
    } else {
      outputs[i] = 0;
    }
  }
}
void softmax(uint8_t* input, int n, uint8_t* output) {
  int i;
  float sum = 0;
  float largest = 0;
  for (i = 0; i < n; ++i) {
    if (input[i] > largest) largest = input[i];
  }
  for (i = 0; i < n; ++i) {
    float e = exp(input[i] - largest);
    sum += e;
    output[i] = e;
  }
  for (i = 0; i < n; ++i) {
    output[i] /= sum;
  }
}

}  // namespace RVTensor
