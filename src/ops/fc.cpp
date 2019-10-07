/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */
#include "include/ops/fc.hpp"
#include "include/ops/active.hpp"
#include "math.h"
#include <float.h>

namespace RVTensor {

CPUFCOp::sptr CPUFCOp::create() { return std::make_shared<CPUFCOp>(); }

CPUFCOp::sptr CPUFCOp::create(RamTensor::sptr input, RamTensor::sptr output,
                              FlashTensor::sptr weight,
                              FlashTensor::sptr bias) {
  CPUFCOp::sptr ptr = std::make_shared<CPUFCOp>(input, output, weight, bias);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUFCOp::CPUFCOp()
    : Operation({}, {}), weight_(nullptr), bias_(nullptr) {}

inline CPUFCOp::CPUFCOp(RamTensor::sptr input, RamTensor::sptr output,
                        FlashTensor::sptr weight, FlashTensor::sptr bias)
    : Operation({input}, {output}), weight_(weight), bias_(bias) {}

inline CPUFCOp::~CPUFCOp() {}

inline void CPUFCOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  float *input = reinterpret_cast<float *>(input_tensor->data_ptr);
  float *output = reinterpret_cast<float *>(output_tensor->data_ptr);
  float *weight = reinterpret_cast<float *>(weight_->data_ptr);
  float *bias =
      bias_ ? reinterpret_cast<float *>(bias_->data_ptr) : nullptr;

  // TODO: complete it
  int n = input_tensor->n_batch;
  int k = input_tensor->count() / n;
  int m = output_tensor->count() / n;

  for (int ni = 0; ni < n; ni++) {
      float* out = output + ni * m;
      float largest = -FLT_MAX;

      // fc
      for (int mi = 0; mi < m; mi++) {
          float sum = bias[mi];
          for (int ki = 0; ki < k; ki++) {
            int in_index = ni * k + ki;
            int m_index = mi * k + ki;
            sum += input[in_index] * weight[m_index];
          }
          out[mi] = sum;
          if (out[mi] > largest) largest = out[mi];
      }

      // softmax
      float sum = 0;
      for (int mi = 0; mi < m; mi++) {
        float e = exp(out[mi] - largest);
        sum += e;
        out[mi] = e;
      }
      for (int mi = 0; mi < m; mi++) {
        out[mi] /= sum;
      }
  }

  // printf("******************************************\n");
  // for (int i = 0; i < output_tensor->count(); i++)
  //     printf("output[%d] = %f\n", i, output[i]);
  // printf("******************************************\n");
}

}  // namespace RVTensor
