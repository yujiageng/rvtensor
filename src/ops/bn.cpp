/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */
#include "include/ops/bn.hpp"
#include "math.h"

namespace RVTensor {

CPUBnOp::sptr CPUBnOp::create() { return std::make_shared<CPUBnOp>(); }

CPUBnOp::sptr CPUBnOp::create(BnModelData bn_param, RamTensor::sptr input,
                              RamTensor::sptr output) {
  CPUBnOp::sptr ptr = std::make_shared<CPUBnOp>(bn_param, input, output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUBnOp::CPUBnOp() : Operation({}, {}),
                            param_({nullptr, nullptr, nullptr, nullptr}),
                            weight_(nullptr),bias_(nullptr),scales_(nullptr) {}

inline CPUBnOp::CPUBnOp(BnModelData bn_param, RamTensor::sptr input,
                        RamTensor::sptr output)
    : Operation({input}, {output}), param_(bn_param) {}

inline CPUBnOp::~CPUBnOp() {
  if (scales_) free(scales_);
}

inline void CPUBnOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  float *input = reinterpret_cast<float *>(input_tensor->data_ptr);
  float *output = reinterpret_cast<float *>(output_tensor->data_ptr);

  float *biases =
      bias_ ? reinterpret_cast<float *>(bias_->data_ptr) : nullptr;
  // TODO: complete it

  int input_batch = input_tensor->n_batch;
  int input_c = input_tensor->channel;
  int input_h = input_tensor->height;
  int input_w = input_tensor->width;

  int output_batch = output_tensor->n_batch;
  int output_c = output_tensor->channel;
  int output_h = output_tensor->height;
  int output_w = output_tensor->width;
  // bn_param;
  float* mean = (float*)(param_.bn_mean_ptr->data_ptr);
  float* variance = (float*)(param_.bn_variance_ptr->data_ptr);
  float *scales = (float*)calloc(output_c, sizeof(float));
  if (!scales) { scales_ = scales;}

  // memset(scales, 1, output_c);
  // 将 input copy 到 output
  copy_cpu(input_tensor->count(), input, 1, output, 1);
  // 归一化
  normalize_cpu(output, &mean[0], &variance[0], input_batch, output_c,
                output_h * output_w);
  // scales大小为out_c的数组，值全是1
  scale_bias(output, scales, input_batch, output_c, output_h * output_w);
  add_bias(output, biases, input_batch, output_c, output_h * output_w);
}

inline void CPUBnOp::copy_cpu(int N, float *X, int INCX, float *Y,
                              int INCY) {
  int i;
  for (i = 0; i < N; ++i) Y[i * INCY] = X[i * INCX];
}
//归一化
inline void CPUBnOp::normalize_cpu(float *x, float *mean, float *variance,
                                   int batch, int filters, int spatial) {
  int b, f, i;
  for (b = 0; b < batch; ++b) {
    for (f = 0; f < filters; ++f) {
      for (i = 0; i < spatial; ++i) {
        int index = b * filters * spatial + f * spatial + i;
        //公式中的ε=.000001f
        x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
      }
    }
  }
}
inline void CPUBnOp::scale_bias(float *output, float *scales, int batch,
                                int n, int size) {
  // scales 全是1
  int i, j, b;
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        output[(b * n + i) * size + j] *= scales[i];
      }
    }
  }
}
inline void CPUBnOp::add_bias(float *output, float *biases, int batch,
                              int n, int size) {
  int i, j, b;
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        output[(b * n + i) * size + j] += biases[i];
      }
    }
  }
}

}  // namespace RVTensor
