/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */
#ifndef INCLUDE_OPS_ACCELERATIONCONV_HPP_
#define INCLUDE_OPS_ACCELERATIONCONV_HPP_

#include <memory>
#include <vector>
#include "include/core/operation.hpp"
#include "include/core/tensor.hpp"
#include "include/core/types.hpp"

namespace RVTensor {

class CPUAccelerationConvOp : public Operation {
 public:
  using sptr = std::shared_ptr<CPUAccelerationConvOp>;
  static sptr create();
  static sptr create(ConvParam conv_param, RamTensor::sptr input,
                     RamTensor::sptr output, FlashTensor::sptr weight,
                     FlashTensor::sptr bias = nullptr);

  /**
   * Constructor & Deconstructor
   */
  CPUAccelerationConvOp();
  CPUAccelerationConvOp(ConvParam conv_param, RamTensor::sptr input,
                        RamTensor::sptr output, FlashTensor::sptr weight,
                        FlashTensor::sptr bias = nullptr);
  ~CPUAccelerationConvOp();
  CPUAccelerationConvOp& operator=(const CPUAccelerationConvOp& conv_op);

  /**
   * check output dims
   */
  void checkOutputDims() override;
  /**
   * 普通矩阵相乘
   */
  void mm_generate(float* matA, float* matB, float* matC, const int M,
                   const int N, const int K, const int strideA,
                   const int strideB, const int strideC);
  /**
   * S1 = A21 + A22     T1 = B21 - B11    M1 = A11*B11    M5 = S1*T1
   * S2 = S1 + A11      T2 = B22 - T1     M2 = A12*B21    M6 = S2*T2
   * S3 = A11 - A21     T3 = B22 - B12    M3 = S4*B22     M7 = S3*T3
   * S4 = A12 - S2      T4 = T2 - B21     M4 =  A22*T4
   * U1 = M1 + M2       U5 = U4 + M3      C11 = U1
   * U2 = M1 + M6       U6 = U3 - M4      C12 = U5
   * U3 = U2 + M7       U7 = U3 + M5      C21 = U6
   * U4 = U2 + M5                         C22 = U7
   *
   */
  void coppersmith_winograd(float* matA, float* matB, float* matC, int M, int N,
                            int K, int strideA, int strideB, int strideC);

  float im2col_get_pixel(float* im, int height, int width, int channels,
                         int row, int col, int channel, int pad);

  void im2col_cpu(float* data_im, int channels, int height, int width,
                  int ksize, int stride, int pad, float* data_col);
  /**
   * inference
   */
  void forward_compute() override;

 private:
  /// conv paramter
  ConvParam param_;
  /// model data: weight
  FlashTensor::sptr weight_;
  /// model data: bias
  FlashTensor::sptr bias_;
  std::vector<float*> tofree_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_ACCELERATIONCONV_HPP_
