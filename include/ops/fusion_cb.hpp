/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_FUSION_CB_HPP_
#define INCLUDE_OPS_FUSION_CB_HPP_

#include <memory>
#include <vector>
#include "include/core/operation.hpp"
#include "include/core/tensor.hpp"
#include "include/core/types.hpp"
#include "include/ops/bn.hpp"
#include "include/ops/conv.hpp"

namespace RVTensor {

class CPUFusionCBOp : public Operation {
 public:
  using sptr = std::shared_ptr<CPUFusionCBOp>;
  static sptr create();
  static sptr create(ConvParam conv_param, BatchNormParam bn_param,
                     RamTensor::sptr input, RamTensor::sptr output,
                     FlashTensor::sptr weight,
                     FlashTensor::sptr bias = nullptr);

  /**
   * Constructor & Deconstructor
   */
  CPUFusionCBOp();
  CPUFusionCBOp(ConvParam conv_param, BatchNormParam bn_param,
                RamTensor::sptr input, RamTensor::sptr output,
                FlashTensor::sptr weight, FlashTensor::sptr bias = nullptr);
  ~CPUFusionCBOp();
  CPUFusionCBOp& operator=(const CPUFusionCBOp& fusion_cb_op);

  /**
   * 卷积
   */
  void coppersmith_winograd(uint8_t* matA, uint8_t* matB, uint8_t* matC, int M,
                            int N, int K, int strideA, int strideB,
                            int strideC);

  float im2col_get_pixel(uint8_t* im, int height, int width, int channels,
                         int row, int col, int channel, int pad);

  void im2col_cpu(uint8_t* data_im, int channels, int height, int width,
                  int ksize, int stride, int pad, uint8_t* data_col);

  void mm_generate(uint8_t* matA, uint8_t* matB, uint8_t* matC, const int M,
                   const int N, const int K, const int strideA,
                   const int strideB, const int strideC);
  // BN
  void normalize_cpu(uint8_t* x, float* mean, float* variance, int batch,
                     int filters, int spatial);
  void scale_bias(uint8_t* output, float* scales, int batch, int n, int size);

  void add_bias(uint8_t* output, uint8_t* biases, int batch, int n, int size);

  void copy_cpu(int N, uint8_t* X, int INCX, uint8_t* Y, int INCY);
  /**
   * inference
   */
  void forward_compute() override;

 private:
  /// conv paramter
  ConvParam conv_param_;
  /// batch norm paramter
  BatchNormParam bn_param_;
  /// model data: weight
  FlashTensor::sptr weight_;
  /// model data: bias
  FlashTensor::sptr bias_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_FUSION_CB_HPP_
