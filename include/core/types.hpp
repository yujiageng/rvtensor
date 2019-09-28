/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_TYPES_HPP_
#define INCLUDE_CORE_TYPES_HPP_

#include <vector>
#include "include/core/tensor.hpp"

namespace RVTensor {

struct ConvParam {
  /// stride
  int sw;
  int sh;
  /// dilated 扩展的w, h
  int dw;
  int dh;
  /// add pad
  int pw;
  int ph;
  /// quantization int8 or float
  bool quantized;
};

struct BatchNormParam {
  std::vector<float> gamma;
  std::vector<float> beta;
  std::vector<float> mean;
  std::vector<float> variance;
  float epsilon;
};

enum ActiveType {
  ACTIVE_SIGMOID = 0,
  ACTIVE_RELU = 1,
};

struct bn_model_data {
    FlashTensor::sptr bn_beta_ptr;
    FlashTensor::sptr bn_gamma_ptr;
    FlashTensor::sptr bn_mean_ptr;
    FlashTensor::sptr bn_variance_ptr;
};

struct conv_model_data {
    FlashTensor::sptr conv_kernel_ptr;
    FlashTensor::sptr conv_bias_ptr;
};

}  // namespace RVTensor

#endif  // INCLUDE_CORE_TYPES_HPP_
