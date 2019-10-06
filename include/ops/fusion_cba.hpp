/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_FUSION_CBA_HPP_
#define INCLUDE_OPS_FUSION_CBA_HPP_

#include <memory>
#include <vector>
#include "include/core/operation.hpp"
#include "include/core/tensor.hpp"
#include "include/core/types.hpp"
#include "include/ops/active.hpp"
#include "include/ops/bn.hpp"
#include "include/ops/conv.hpp"
namespace RVTensor {

class CPUFusionCBAOp : public Operation {
 public:
  using sptr = std::shared_ptr<CPUFusionCBAOp>;
  static sptr create();
  static sptr create(ConvParam conv_param, BnModelData bn_param,
                     ActiveType active_type, RamTensor::sptr input,
                     RamTensor::sptr output, FlashTensor::sptr weight,
                     FlashTensor::sptr bias = nullptr);

  /**
   * Constructor & Deconstructor
   */
  CPUFusionCBAOp();
  CPUFusionCBAOp(ConvParam conv_param, BnModelData bn_param,
                 ActiveType active_type, RamTensor::sptr input,
                 RamTensor::sptr output, FlashTensor::sptr weight,
                 FlashTensor::sptr bias = nullptr);
  ~CPUFusionCBAOp();
  CPUFusionCBAOp& operator=(const CPUFusionCBAOp& fusion_cba_op);

  /**
   * 卷积
   */
  template <typename T>
  void coppersmith_winograd(T* matA, float* matB, float* matC, int M,
                            int N, int K, int strideA, int strideB,
                            int strideC);

  template <typename T>
  float im2col_get_pixel(T* im, int height, int width, int channels,
                         int row, int col, int channel, int pad);

  template <typename T>
  void im2col_cpu(T* data_im, int channels, int height, int width,
                  int ksize, int stride, int pad, T* data_col);

  template <typename T>
  void mm_generate(T* matA, float* matB, float* matC, const int M,
                   const int N, const int K, const int strideA,
                   const int strideB, const int strideC);

  // BN
  void normalize_cpu(float* x, float* mean, float* variance,
                     float* gamma, float*beta,
                     int batch, int filters, int spatial);
  void scale_bias(float* output, float* scales, int batch, int n, int size);

  void add_bias(float* output, float* biases, int batch, int n, int size);

  void copy_cpu(int N, float* X, int INCX, float* Y, int INCY);

  // 激活
  void relu(float* input, int size);
  /**
   * inference
   */
  void forward_compute() override;

 private:
  /// conv paramter
  ConvParam conv_param_;
  /// bn paramter
  BnModelData bn_param_;
  /// active type
  ActiveType active_type_;
  /// model data: weight
  FlashTensor::sptr weight_;
  /// model data: bias
  FlashTensor::sptr bias_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_FUSION_CBA_HPP_
