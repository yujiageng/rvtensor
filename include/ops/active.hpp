/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_ACTIVE_HPP_
#define INCLUDE_OPS_ACTIVE_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"
#include "include/core/types.hpp"

namespace RVTensor {

class CPUActiveOp: public Operation {
 public:
    using sptr = std::shared_ptr<CPUActiveOp>;
    static sptr create();
    static sptr create(ActiveType active_type,
        RamTensor::sptr input,
        RamTensor::sptr output);

    /**
     * Constructor & Deconstructor
     */
    CPUActiveOp();
    CPUActiveOp(ActiveType active_type,
        RamTensor::sptr input,
        RamTensor::sptr output);
    ~CPUActiveOp();
    CPUActiveOp& operator=(const CPUActiveOp& active_op);

    /**
     * check output dims
     */
    // void checkOutputDims() override;

    /**
     * inference
     */
    void forward_compute() override;

 private:
    /// active type
    ActiveType param_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_ACTIVE_HPP_
