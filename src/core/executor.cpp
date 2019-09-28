/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/executor.hpp"

namespace RVTensor {

Executor::sptr Executor::create() {
  return std::make_shared<Executor>();
}

Executor::sptr Executor::create(std::string model_name,
                                int thread_num) {
  return std::make_shared<Executor>(model_name, thread_num);
}

Executor::Executor() {}

Executor::Executor(std::string model_name, int thread_num)
                  : thread_num_(thread_num), model_name(model_name),
                  image_ptr(nullptr), output_ptr(nullptr) {
    resnet_model_data_ptr = ResnetModelData::create();
    resnet_model_data_ptr->openModelFile(model_name.c_str());
}

void Executor::parseModel() {

    int n_batch = 500;
    ConvModelData conv_data;
    BnModelData bn_data;

    temp_0 = RamTensor::create(n_batch, 32, 32, 16, 4u);
    temp_1 = RamTensor::create(n_batch, 32, 32, 16, 4u);
    temp_2 = RamTensor::create(n_batch, 32, 32, 16, 4u);

    // conv1 + bn1 + at1
    ConvParam conv1_param({1, 1, 1, 1, 0, 0, 0});
    conv_data = resnet_model_data_ptr->getConvModelData(1);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(1);
    cba1_1 = CPUFusionCBAOp::create(conv1_param, bn_data, ACTIVE_RELU,
                                    image_ptr, temp_0,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);

    // conv2 + bn2 + at2
    ConvParam conv2_param({1, 1, 1, 1, 0, 0, 0});
    conv_data = resnet_model_data_ptr->getConvModelData(2);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(2);
    cba2_2 = CPUFusionCBAOp::create(conv2_param, bn_data, ACTIVE_RELU,
                                    temp_0, temp_1,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);

    // conv3 + bn3
    ConvParam conv3_param({1, 1, 1, 1, 0, 0, 0});
    conv_data = resnet_model_data_ptr->getConvModelData(3);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(3);
    cb3_3 = CPUFusionCBOp::create(conv3_param, bn_data,
                                  temp_1, temp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    // add1
    add1_4 = CPUAddOp::create(temp_0, temp_2, temp_1);

    // ac3
    ac3_5 = CPUActiveOp::create(ACTIVE_RELU, temp_1, temp_0);

}

void Executor::loadImage(std::string image_name, uint8_t* ai_buf,
                         int channel, int height, int width) {
  image_ptr = RamTensor::create(1, channel, height, width,
                                reinterpret_cast<void*>(ai_buf), 1u);
  // TODO: load image content

}

int Executor::compute() {
    return 0;
}

//int Executor::inferenceResult(void* result_buf, uint64_t size) {
int Executor::inferenceResult() {
    // TODO: image classfication to resnet

    return 0;
}

Executor::~Executor() {
    resnet_model_data_ptr->closeModelFile();
}

}  // namespace RVTensor
