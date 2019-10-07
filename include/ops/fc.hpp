/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_FC_HPP_
#define INCLUDE_OPS_FC_HPP_

#include <memory>
#include <vector>
#include "include/core/operation.hpp"
#include "include/core/tensor.hpp"
#include "include/ops/bn.hpp"

namespace RVTensor {

class CPUFCOp : public Operation {
 public:
  using sptr = std::shared_ptr<CPUFCOp>;
  static sptr create();
  static sptr create(RamTensor::sptr input, RamTensor::sptr output,
                     FlashTensor::sptr weight,
                     FlashTensor::sptr bias = nullptr);

  /**
   * Constructor & Deconstructor
   */
  CPUFCOp();
  CPUFCOp(RamTensor::sptr input, RamTensor::sptr output,
          FlashTensor::sptr weight, FlashTensor::sptr bias = nullptr);
  ~CPUFCOp();
  CPUFCOp& operator=(const CPUFCOp& fc_op);

  /**
   * inference
   */
  void forward_compute() override;

 private:
  /// model data: weight
  FlashTensor::sptr weight_;
  /// model data: bias
  FlashTensor::sptr bias_;
  ActiveType param_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_FC_HPP_
