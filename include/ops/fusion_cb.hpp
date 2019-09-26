/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_FUSION_CB_HPP_
#define INCLUDE_OPS_FUSION_CB_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"
#include "include/core/types.hpp"
#include "include/ops/conv.hpp"
#include "include/ops/bn.hpp"

namespace RVTensor {

class CPUFusionCBOp: public Operation {
 public:
    using sptr = std::shared_ptr<CPUFusionCBOp>;
    static sptr create();
    static sptr create(ConvParam conv_param, BatchNormParam bn_param,
        RamTensor::sptr input,
        RamTensor::sptr output,
        FlashTensor::sptr weight,
        FlashTensor::sptr bias = nullptr);

    /**
     * Constructor & Deconstructor
     */
    CPUFusionCBOp();
    CPUFusionCBOp(ConvParam conv_param, BatchNormParam bn_param,
        RamTensor::sptr input,
        RamTensor::sptr output,
        FlashTensor::sptr weight,
        FlashTensor::sptr bias = nullptr);
    ~CPUFusionCBOp();
    CPUFusionCBOp& operator=(const CPUFusionCBOp& fusion_cb_op);

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
    /// model data: weight
    FlashTensor::sptr weight_;
    /// model data: bias
    FlashTensor::sptr bias_;
};

}  // namespace RVTensor

#endif  // INCLUDE_OPS_FUSION_CB_HPP_
