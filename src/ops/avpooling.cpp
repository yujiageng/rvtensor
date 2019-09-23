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

  uint8_t* input = reinterpret_cast<uint8_t *>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t *>(output_tensor->data_ptr);

  // TODO: complete it
}

}  // namespace RVTensor
