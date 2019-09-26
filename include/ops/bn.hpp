/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_BN_HPP_
#define INCLUDE_OPS_BN_HPP_

#include <memory>
#include <vector>
#include "include/core/operation.hpp"
#include "include/core/tensor.hpp"
#include "include/core/types.hpp"

namespace RVTensor {

class CPUBnOp : public Operation {
 public:
  using sptr = std::shared_ptr<CPUBnOp>;
  static sptr create();
  static sptr create(BatchNormParam bn_param, RamTensor::sptr input,
                     RamTensor::sptr output);

  /**
   * Constructor & Deconstructor
   */
  CPUBnOp();
  CPUBnOp(BatchNormParam bn_param, RamTensor::sptr input,
          RamTensor::sptr output);
  ~CPUBnOp();
  CPUBnOp &operator=(const CPUBnOp &bn_op);

  /**
   * inference
   */
  void forward_compute() override;

  void normalize_cpu(uint8_t *x, float *mean, float *variance, int batch,
                     int filters, int spatial);
  void scale_bias(uint8_t *output, float *scales, int batch, int n, int size);

  void add_bias(uint8_t *output, uint8_t *biases, int batch, int n, int size);

  void copy_cpu(int N, uint8_t *X, int INCX, uint8_t *Y, int INCY);

 private:
  /// bn paramter
  BatchNormParam param_;
  /// model data: weight
  FlashTensor::sptr weight_;
  /// model data: bias
  FlashTensor::sptr bias_;
  float *scales_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_BN_HPP_
