/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */
#include "include/ops/fc.hpp"
#include "include/ops/active.hpp"
#include "math.h"

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
  int m = input_tensor->n_batch;
  int k = input_tensor->count();
  int n = output_tensor->count();

  multl(m, n, k, input, k, weight, k, output, n);

  add_bias(output, bias, m, n, 1);

  softmax(output, n);
}
inline void CPUFCOp::multl(int M, int N, int K, float *A, int lda, float *B,
                           int ldb, float *C, int ldc) {
  int i, j, k;
  // M=batch，每个样本有N个输出
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float sum = 0;
      // K是inputs，即输入个数
      for (k = 0; k < K; ++k) {
        //输入项和权重项对应相乘相加
        sum += A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}
inline void CPUFCOp::softmax(float *input, int n) {
  int i;
  float sum = 0;
  float largest = 0;
  for (i = 0; i < n; ++i) {
    if (input[i] > largest) largest = input[i];
  }
  for (i = 0; i < n; ++i) {
    float e = exp(input[i] - largest);
    sum += e;
    input[i] = e;
  }
  for (i = 0; i < n; ++i) {
    input[i] /= sum;
  }
}
inline void CPUFCOp::add_bias(float *output, float *biases, int batch,
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
