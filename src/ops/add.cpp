/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/add.hpp"

namespace RVTensor {

CPUAddOp::sptr CPUAddOp::create() {
  return std::make_shared<CPUAddOp>();
}

CPUAddOp::sptr CPUAddOp::create(RamTensor::sptr input1, RamTensor::sptr input2,
                                RamTensor::sptr output) {
  CPUAddOp::sptr ptr = std::make_shared<CPUAddOp>(input1, input2, output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUAddOp::CPUAddOp() : Operation({}, {}) {}

inline CPUAddOp::CPUAddOp(RamTensor::sptr input1, RamTensor::sptr input2,
                          RamTensor::sptr output)
                          : Operation({input1, input2}, {output}) {}

inline CPUAddOp::~CPUAddOp() {}

inline void CPUAddOp::forward_compute() {
  auto input_tensor1 = getInputs()[0];
  auto input_tensor2 = getInputs()[1];
  auto output_tensor = getOutputs()[0];

  uint8_t* input1 = reinterpret_cast<uint8_t *>(input_tensor1->data_ptr);
  uint8_t* input2 = reinterpret_cast<uint8_t *>(input_tensor2->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t *>(output_tensor->data_ptr);

  // TODO: complete it
}

}  // namespace RVTensor
