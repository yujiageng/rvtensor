/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/fc.hpp"

namespace RVTensor {

CPUFCOp::sptr CPUFCOp::create() {
  return std::make_shared<CPUFCOp>();
}

CPUFCOp::sptr CPUFCOp::create(RamTensor::sptr input, RamTensor::sptr output,
                             FlashTensor::sptr weight,
                             FlashTensor::sptr bias) {
  CPUFCOp::sptr ptr = std::make_shared<CPUFCOp>(input, output, weight, bias);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUFCOp::CPUFCOp() : Operation({}, {}),
                                weight_(nullptr), bias_(nullptr) {}

inline CPUFCOp::CPUFCOp(RamTensor::sptr input, RamTensor::sptr output,
                            FlashTensor::sptr weight,
                            FlashTensor::sptr bias)
                          : Operation({input}, {output}),
                            weight_(weight), bias_(bias) {}

inline CPUFCOp::~CPUFCOp() {}

inline void CPUFCOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t *>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t *>(output_tensor->data_ptr);
  uint8_t* weight = reinterpret_cast<uint8_t *>(weight_->data_ptr);
  uint8_t* bias = bias_ ?
                    reinterpret_cast<uint8_t *>(bias_->data_ptr) : nullptr;

  // TODO: complete it
}

}  // namespace RVTensor
