/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/bn.hpp"

namespace RVTensor {

CPUBnOp::sptr CPUBnOp::create() {
  return std::make_shared<CPUBnOp>();
}

CPUBnOp::sptr CPUBnOp::create(BatchNormParam bn_param, RamTensor::sptr input,
                             RamTensor::sptr output) {
  CPUBnOp::sptr ptr = std::make_shared<CPUBnOp>(bn_param, input, output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUBnOp::CPUBnOp() : Operation({}, {}), param_({{}, {}, 0.001}) {}

inline CPUBnOp::CPUBnOp(BatchNormParam bn_param, RamTensor::sptr input,
                            RamTensor::sptr output)
                          : Operation({input}, {output}), param_(bn_param) {}

inline CPUBnOp::~CPUBnOp() {}

inline void CPUBnOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t *>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t *>(output_tensor->data_ptr);

  // TODO: complete it
}

}  // namespace RVTensor
