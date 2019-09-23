/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_ADD_HPP_
#define INCLUDE_OPS_ADD_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"

namespace RVTensor {

class CPUAddOp: public Operation {
 public:
    using sptr = std::shared_ptr<CPUAddOp>;
    static sptr create();
    static sptr create(RamTensor::sptr input1, RamTensor::sptr input2,
                       RamTensor::sptr output);

    /**
     * Constructor & Deconstructor
     */
    CPUAddOp();
    CPUAddOp(RamTensor::sptr input1,
        RamTensor::sptr input2,
        RamTensor::sptr output);
    ~CPUAddOp();
    CPUAddOp& operator=(const CPUAddOp& add_op);

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

#endif  // INCLUDE_OPS_ADD_HPP_
