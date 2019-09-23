/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_BN_HPP_
#define INCLUDE_OPS_BN_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"
#include "include/core/types.hpp"

namespace RVTensor {

class CPUBnOp: public Operation {
 public:
    using sptr = std::shared_ptr<CPUBnOp>;
    static sptr create();
    static sptr create(BatchNormParam bn_param,
        RamTensor::sptr input,
        RamTensor::sptr output);

    /**
     * Constructor & Deconstructor
     */
    CPUBnOp();
    CPUBnOp(BatchNormParam bn_param,
        RamTensor::sptr input,
        RamTensor::sptr output);
    ~CPUBnOp();
    CPUBnOp& operator=(const CPUBnOp& bn_op);

    /**
     * check output dims
     */
    // void checkOutputDims() override;

    /**
     * inference
     */
    void forward_compute() override;

 private:
    /// bn paramter
    BatchNormParam param_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_BN_HPP_
