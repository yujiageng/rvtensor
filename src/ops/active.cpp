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
void CPUActiveOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  float* input = reinterpret_cast<float*>(input_tensor->data_ptr);
  float* output = reinterpret_cast<float*>(output_tensor->data_ptr);

  // TODO: complete it
  if (param_ == ACTIVE_SOFTMAX) {
    softmax(input, input_tensor->count(), output);
  } else if (param_ == ACTIVE_RELU) {
    relu(input, input_tensor->count(), output);
  } else {
    throw std::runtime_error("Cannot find active function!");
  }

}
void CPUActiveOp::relu(float* inputs, int n, float* outputs) {
  int i;
  for (i = 0; i < n; ++i) {
    if (inputs[i] > 0) {
      outputs[i] = inputs[i];
    } else {
      outputs[i] = 0;
    }
  }
}

void CPUActiveOp::softmax(float* input, int n, float* output) {
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
