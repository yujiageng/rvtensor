/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/fusion_cba.hpp"

namespace RVTensor {

CPUFusionCBAOp::sptr CPUFusionCBAOp::create() {
  return std::make_shared<CPUFusionCBAOp>();
}

CPUFusionCBAOp::sptr CPUFusionCBAOp::create(
    ConvParam conv_param, BatchNormParam bn_param, ActiveType active_type,
    RamTensor::sptr input, RamTensor::sptr output, FlashTensor::sptr weight,
    FlashTensor::sptr bias) {
  CPUFusionCBAOp::sptr ptr = std::make_shared<CPUFusionCBAOp>(
      conv_param, bn_param, active_type, input, output, weight, bias);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUFusionCBAOp::CPUFusionCBAOp()
    : Operation({}, {}),
      conv_param_({0, 0, 1, 1, 0, 0, false}),
      bn_param_({{}, {}, 0.001}),
      active_type_(ACTIVE_SIGMOID),
      weight_(nullptr),
      bias_(nullptr) {}

inline CPUFusionCBAOp::CPUFusionCBAOp(
    ConvParam conv_param, BatchNormParam bn_param, ActiveType active_type,
    RamTensor::sptr input, RamTensor::sptr output, FlashTensor::sptr weight,
    FlashTensor::sptr bias)
    : Operation({input}, {output}),
      conv_param_(conv_param),
      bn_param_(bn_param),
      active_type_(active_type),
      weight_(weight),
      bias_(bias) {}

inline CPUFusionCBAOp::~CPUFusionCBAOp() {}

inline void CPUFusionCBAOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t*>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t*>(output_tensor->data_ptr);
  uint8_t* weight = reinterpret_cast<uint8_t*>(weight_->data_ptr);
  uint8_t* bias = bias_ ? reinterpret_cast<uint8_t*>(bias_->data_ptr) : nullptr;

  // TODO: complete it
  CPUConvOp conv(conv_param_, input, output, weight, bias);
  conv.forward_compute();
  CPUBnOp BN(bn_param_, input, output);
  BN.forward_compute();
  CPUActiveOp activate(active_type_, input, output);
  activate.forward_compute();
}

}  // namespace RVTensor
