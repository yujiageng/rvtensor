/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_AVPOOLING_HPP_
#define INCLUDE_OPS_AVPOOLING_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"

namespace RVTensor {

class CPUAVPoolingOp: public Operation {
 public:
    using sptr = std::shared_ptr<CPUAVPoolingOp>;
    static sptr create();
    static sptr create(RamTensor::sptr input,
                       RamTensor::sptr output);

    /**
     * Constructor & Deconstructor
     */
    CPUAVPoolingOp();
    CPUAVPoolingOp(RamTensor::sptr input,
                   RamTensor::sptr output);
    ~CPUAVPoolingOp();
    CPUAVPoolingOp& operator=(const CPUAVPoolingOp& avpooling_op);

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

#endif  // INCLUDE_OPS_AVPOOLING_HPP_
