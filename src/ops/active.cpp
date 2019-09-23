/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/active.hpp"

namespace RVTensor {

CPUActiveOp::sptr CPUActiveOp::create() {
  return std::make_shared<CPUActiveOp>();
}

CPUActiveOp::sptr CPUActiveOp::create(ActiveType active_type,
                                      RamTensor::sptr input,
                                      RamTensor::sptr output) {
  CPUActiveOp::sptr ptr = std::make_shared<CPUActiveOp>(active_type, input,
                                                        output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUActiveOp::CPUActiveOp() : Operation({}, {}),
                                param_(ACTIVE_SIGMOID) {}

inline CPUActiveOp::CPUActiveOp(ActiveType active_type, RamTensor::sptr input,
                            RamTensor::sptr output)
                          : Operation({input}, {output}), param_(active_type) {}

inline CPUActiveOp::~CPUActiveOp() {}

inline void CPUActiveOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t *>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t *>(output_tensor->data_ptr);

  // TODO: complete it
}

}  // namespace RVTensor
