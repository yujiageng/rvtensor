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
#include "include/core/tensor.hpp"
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
    static sptr create(std::string model_name, int thread_num = 1);

    /**
     * Constructor
     */
    Executor();
    Executor(std::string model_name, int thread_num = 1);

    // for resnet20
    void parseModel();

    /**
     * load image to Mat struct
     *
     * @param ai_buf: camera input image
     * @param height: height of input image
     * @param width:  width of imput image
     */
    void loadImage(std::string image_name, uint8_t* ai_buf,
                   int channel, int height, int width);

    /**
     * Start to inference
     */
    int compute();

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
    /// image struct
    RamTensor::sptr image_ptr;
    RamTensor::sptr output_ptr;
    /// model_name
    std::string model_name;
    /// for resnet20
    RamTensor::sptr temp_0;
    RamTensor::sptr temp_1;
    RamTensor::sptr temp_2;
    CPUFusionCBAOp::sptr cba1_1;
    FlashTensor::sptr conv1_weight_ptr;
    FlashTensor::sptr conv1_bias_ptr;
    CPUFusionCBAOp::sptr cba2_2;
    FlashTensor::sptr conv2_weight_ptr;
    FlashTensor::sptr conv2_bias_ptr;
    CPUFusionCBOp::sptr cb3_3;
    FlashTensor::sptr conv3_weight_ptr;
    FlashTensor::sptr conv3_bias_ptr;
    CPUAddOp::sptr add1_4;
    CPUActiveOp::sptr ac3_5;

};

}  // namespace RVTensor

#endif  // INCLUDE_CORE_EXECUTOR_HPP_
