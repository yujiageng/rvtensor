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
  static sptr create(ConvParam conv_param, BatchNormParam bn_param,
                     ActiveType active_type, RamTensor::sptr input,
                     RamTensor::sptr output, FlashTensor::sptr weight,
                     FlashTensor::sptr bias = nullptr);

  /**
   * Constructor & Deconstructor
   */
  CPUFusionCBAOp();
  CPUFusionCBAOp(ConvParam conv_param, BatchNormParam bn_param,
                 ActiveType active_type, RamTensor::sptr input,
                 RamTensor::sptr output, FlashTensor::sptr weight,
                 FlashTensor::sptr bias = nullptr);
  ~CPUFusionCBAOp();
  CPUFusionCBAOp& operator=(const CPUFusionCBAOp& fusion_cba_op);

  /**
   * check output dims
   */
  // void checkOutputDims() override;

  /**
   * inference
   */
  void forward_compute() override;

 private:
  /// conv paramter
  ConvParam conv_param_;
  /// batch norm paramter
  BatchNormParam bn_param_;
  /// active type
  ActiveType active_type_;
  /// model data: weight
  FlashTensor::sptr weight_;
  /// model data: bias
  FlashTensor::sptr bias_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_FUSION_CBA_HPP_
