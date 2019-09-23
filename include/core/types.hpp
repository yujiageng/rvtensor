/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_TYPES_HPP_
#define INCLUDE_CORE_TYPES_HPP_

#include<vector>

namespace RVTensor {

struct ConvParam {
  /// stride
  int sw;
  int sh;
  /// dilated
  int dw;
  int dh;
  /// add pad
  int pw;
  int ph;
  /// quantization int8 or float
  bool quantized;
};

struct BatchNormParam {
  std::vector<float> mean;
  std::vector<float> variance;
  float epsilon;
};

enum ActiveType {
  ACTIVE_SIGMOID = 0,
};

}  // namespace RVTensor

#endif  // INCLUDE_CORE_TYPES_HPP_
