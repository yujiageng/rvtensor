/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_FLATTEN_HPP_
#define INCLUDE_OPS_FLATTEN_HPP_

#include "include/core/operation.hpp"
#include "include/core/tensor.hpp"
#include "include/core/types.hpp"

namespace RVTensor {

class CPUFlattenOp : public Operation {
 public:
  using sptr = std::shared_ptr<CPUFlattenOp>;
  static sptr create();
  static sptr create(RamTensor::sptr input, RamTensor::sptr output);

  /**
   * Constructor & Deconstructor
   */
  CPUFlattenOp();
  CPUFlattenOp(RamTensor::sptr input, RamTensor::sptr output);
  ~CPUFlattenOp();
  CPUFlattenOp& operator=(const CPUFlattenOp& flatten_op);

  /**
   * check output dims
   */
  // void checkOutputDims() override;

  /**
   * inference
   */
  void forward_compute() override;

 private:
};

}  // namespace RVTensor

#endif
