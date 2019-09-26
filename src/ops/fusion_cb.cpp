/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/fusion_cb.hpp"

namespace RVTensor {

CPUFusionCBOp::sptr CPUFusionCBOp::create() {
  return std::make_shared<CPUFusionCBOp>();
}

CPUFusionCBOp::sptr CPUFusionCBOp::create(
    ConvParam conv_param, BatchNormParam bn_param, RamTensor::sptr input,
    RamTensor::sptr output, FlashTensor::sptr weight, FlashTensor::sptr bias) {
  CPUFusionCBOp::sptr ptr = std::make_shared<CPUFusionCBOp>(
      conv_param, bn_param, input, output, weight, bias);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUFusionCBOp::CPUFusionCBOp()
    : Operation({}, {}),
      conv_param_({0, 0, 1, 1, 0, 0, false}),
      bn_param_({{}, {}, 0.001}),
      weight_(nullptr),
      bias_(nullptr) {}

inline CPUFusionCBOp::CPUFusionCBOp(
    ConvParam conv_param, BatchNormParam bn_param, RamTensor::sptr input,
    RamTensor::sptr output, FlashTensor::sptr weight, FlashTensor::sptr bias)
    : Operation({input}, {output}),
      conv_param_(conv_param),
      bn_param_(bn_param),
      weight_(weight),
      bias_(bias) {}

inline CPUFusionCBOp::~CPUFusionCBOp() {}

inline void CPUFusionCBOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t*>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t*>(output_tensor->data_ptr);
  uint8_t* weight = reinterpret_cast<uint8_t*>(weight_->data_ptr);
  uint8_t* bias = bias_ ? reinterpret_cast<uint8_t*>(bias_->data_ptr) : nullptr;

  //[TODO] create a tmp_output
  auto conv = CPUConvOp::create(conv_param_, input_tensor, output_tensor, weight_, bias_);
  conv->forward_compute();
  //[TODO] use tmp_output as input
  auto bn= CPUBnOp::create(bn_param_, input_tensor, output_tensor);
  bn->forward_compute();
}

}  // namespace RVTensor
