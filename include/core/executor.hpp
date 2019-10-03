/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_EXECUTOR_HPP_
#define INCLUDE_CORE_EXECUTOR_HPP_

#include <string>
#include <vector>
#include "include/core/tensor.hpp"
#include "include/core/types.hpp"
#include "include/model/resnet_model.hpp"
#include "include/ops/accelerationconv.hpp"
#include "include/ops/active.hpp"
#include "include/ops/add.hpp"
#include "include/ops/avpooling.hpp"
#include "include/ops/bn.hpp"
#include "include/ops/conv.hpp"
#include "include/ops/fc.hpp"
#include "include/ops/flatten.hpp"
#include "include/ops/fusion_cb.hpp"
#include "include/ops/fusion_cba.hpp"


namespace RVTensor {

/**
 * Executor describes the context of a individual inference task.
 */
class Executor {
 public:
    using sptr = std::shared_ptr<Executor>;
    static sptr create();
    static sptr create(std::string model_name, int batch, int thread_num = 1);

    /**
     * Constructor
     */
    Executor();
    Executor(std::string model_name, int batch, int thread_num = 1);

    // for resnet20
    void parseModel();

    /**
     * load image to Mat struct
     *
     * @param ai_buf: camera input image
     * @param height: height of input image
     * @param width:  width of imput image
     */
    void loadImage(std::string image_name,
                   int batch, int height, int width, int channel);

    /**
     * Start to inference
     */
    int compute(int batch_round);

    /**
     * Copy Output data to application
     */
     // void copyOutputData(void* data_ptr, size_t size);

    /**
     * analysis inference result
     *
     * @param result_buf: result of inference
     * @param size: result size
     */
    //int inferenceResult(void* result_buf, uint64_t size);
    int inferenceResult();

    /**
     * Deconstructor
     */
    ~Executor();

    Executor& operator=(const Executor& exe);

 private:
    /// thread num
    int thread_num_;
    /// batch
    int n_batch;
    /// image struct
    RamTensor::sptr image_ptr;
    RamTensor::sptr operation_ptr;
    RamTensor::sptr label_ptr;
    /// model_name
    std::string model_name;
    /// resnet model data
    ResnetModelData::sptr resnet_model_data_ptr;
    /// for resnet20
    RamTensor::sptr temp_0;
    RamTensor::sptr temp_1;
    RamTensor::sptr temp_2;

    RamTensor::sptr stemp_0;
    RamTensor::sptr stemp_1;
    RamTensor::sptr stemp_2;

    RamTensor::sptr sstemp_0;
    RamTensor::sptr sstemp_1;
    RamTensor::sptr sstemp_2;

    RamTensor::sptr pool_temp;
    RamTensor::sptr dense_temp;

    CPUFusionCBAOp::sptr cba1_1;
    CPUFusionCBAOp::sptr cba2_2;
    CPUFusionCBOp::sptr cb3_3;
    CPUAddOp::sptr add1_4;
    CPUActiveOp::sptr ac3_5;

    CPUFusionCBAOp::sptr cba4_6;
    CPUFusionCBOp::sptr cb5_7;
    CPUAddOp::sptr add2_8;
    CPUActiveOp::sptr ac5_9;

    CPUFusionCBAOp::sptr cba6_10;
    CPUFusionCBOp::sptr cb7_11;
    CPUAddOp::sptr add3_12;
    CPUActiveOp::sptr ac7_13;

    CPUFusionCBAOp::sptr cba8_14;
    CPUFusionCBOp::sptr cb9_15;
    CPUConvOp::sptr c10_16;
    CPUAddOp::sptr add4_17;
    CPUActiveOp::sptr ac9_18;

    CPUFusionCBAOp::sptr cba11_19;
    CPUFusionCBOp::sptr cb12_20;
    CPUAddOp::sptr add5_21;
    CPUActiveOp::sptr ac11_22;

    CPUFusionCBAOp::sptr cba13_23;
    CPUFusionCBOp::sptr cb14_24;
    CPUAddOp::sptr add6_25;
    CPUActiveOp::sptr ac13_26;

    CPUFusionCBAOp::sptr cba15_27;
    CPUFusionCBOp::sptr cb16_28;
    CPUConvOp::sptr c17_29;
    CPUAddOp::sptr add7_30;
    CPUActiveOp::sptr ac15_31;

    CPUFusionCBAOp::sptr cba18_32;
    CPUFusionCBOp::sptr cb19_33;
    CPUAddOp::sptr add8_34;
    CPUActiveOp::sptr ac17_35;

    CPUFusionCBAOp::sptr cba20_36;
    CPUFusionCBOp::sptr cb21_37;
    CPUAddOp::sptr add9_38;
    CPUActiveOp::sptr ac19_39;

    CPUAVPoolingOp::sptr avpool_40;
    CPUFCOp::sptr dense_42;
    std::vector<Operation::sptr> ops_vec;
};

}  // namespace RVTensor

#endif  // INCLUDE_CORE_EXECUTOR_HPP_
