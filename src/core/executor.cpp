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
                                int batch,
                                int thread_num) {
  return std::make_shared<Executor>(model_name, batch, thread_num);
}

Executor::Executor() {}

Executor::Executor(std::string model_name, int batch, int thread_num)
                  : thread_num_(thread_num), model_name(model_name),
                  image_ptr(nullptr), operation_ptr(nullptr),
                  n_batch(batch) {
    resnet_model_data_ptr = ResnetModelData::create();
    resnet_model_data_ptr->openModelFile(model_name.c_str());
}

void Executor::parseModel() {

    ConvModelData conv_data;
    BnModelData bn_data;

    temp_0 = RamTensor::create(n_batch, 32, 32, 16, 4u);
    temp_1 = RamTensor::create(n_batch, 32, 32, 16, 4u);
    temp_2 = RamTensor::create(n_batch, 32, 32, 16, 4u);

    stemp_0 = RamTensor::create(n_batch, 16, 16, 32, temp_0->data_ptr, 4u);
    stemp_1 = RamTensor::create(n_batch, 16, 16, 32, temp_1->data_ptr, 4u);
    stemp_2 = RamTensor::create(n_batch, 16, 16, 32, temp_2->data_ptr, 4u);

    sstemp_0 = RamTensor::create(n_batch, 8, 8, 64, temp_0->data_ptr, 4u);
    sstemp_1 = RamTensor::create(n_batch, 8, 8, 64, temp_1->data_ptr, 4u);
    sstemp_2 = RamTensor::create(n_batch, 8, 8, 64, temp_2->data_ptr, 4u);

    pool_temp = RamTensor::create(n_batch, 1, 1, 64, temp_0->data_ptr, 4u);
    dense_temp = RamTensor::create(n_batch, 1, 1, 10, temp_2->data_ptr, 4u);

    // conv1 + bn1 + at1
    ConvParam conv_param = {1, 1, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(1);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(1);
    cba1_1 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    operation_ptr, temp_0,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba1_1);
    // conv2 + bn2 + at2
    conv_data = resnet_model_data_ptr->getConvModelData(2);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(2);
    cba2_2 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    temp_0, temp_1,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba2_2);
    // conv3 + bn3
    conv_data = resnet_model_data_ptr->getConvModelData(3);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(3);
    cb3_3 = CPUFusionCBOp::create(conv_param, bn_data,
                                  temp_1, temp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb3_3);
    // add1
    add1_4 = CPUAddOp::create(temp_0, temp_2, temp_1);
    ops_vec.push_back(add1_4);
    // ac3
    ac3_5 = CPUActiveOp::create(ACTIVE_RELU, temp_1, temp_0);
    ops_vec.push_back(ac3_5);
    // conv4 + bn4 + at4
    conv_data = resnet_model_data_ptr->getConvModelData(4);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(4);
    cba4_6 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    temp_0, temp_1,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba4_6);
    // conv5 + bn5
    conv_data = resnet_model_data_ptr->getConvModelData(5);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(5);
    cb5_7 = CPUFusionCBOp::create(conv_param, bn_data,
                                  temp_1, temp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb5_7);
    // add2
    add2_8 = CPUAddOp::create(temp_0, temp_2, temp_1);
    ops_vec.push_back(add2_8);
    // ac5
    ac5_9 = CPUActiveOp::create(ACTIVE_RELU, temp_1, temp_0);
    ops_vec.push_back(ac5_9);
    // conv6 + bn6 + at6
    conv_data = resnet_model_data_ptr->getConvModelData(6);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(6);
    cba6_10 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    temp_0, temp_1,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba6_10);
    // conv7 + bn7
    conv_data = resnet_model_data_ptr->getConvModelData(7);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(7);
    cb7_11 = CPUFusionCBOp::create(conv_param, bn_data,
                                  temp_1, temp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb7_11);
    // add3
    add3_12 = CPUAddOp::create(temp_0, temp_2, temp_1);
    ops_vec.push_back(add3_12);
    // ac7
    ac7_13 = CPUActiveOp::create(ACTIVE_RELU, temp_1, temp_0);
    ops_vec.push_back(ac7_13);
    // conv8 + bn8 + at8
    conv_param = {2, 2, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(8);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(8);
    cba8_14 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    temp_0, stemp_1,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba8_14);
    // conv9 + bn9
    conv_param = {1, 1, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(9);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(9);
    cb9_15 = CPUFusionCBOp::create(conv_param, bn_data,
                                  stemp_1, stemp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb9_15);
    // conv10
    conv_param = {2, 2, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(10);
    c10_16 = CPUConvOp::create(conv_param,
                                  temp_0, stemp_1,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(c10_16);
    // add4
    add4_17 = CPUAddOp::create(stemp_1, stemp_2, stemp_0);
    ops_vec.push_back(add4_17);
    // ac9
    ac9_18 = CPUActiveOp::create(ACTIVE_RELU, stemp_0, stemp_1);
    ops_vec.push_back(ac9_18);
    // conv11 + bn10 + at10
    conv_param = {1, 1, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(11);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(10);
    cba11_19 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    stemp_1, stemp_2,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba11_19);
    // conv12 + bn11
    conv_data = resnet_model_data_ptr->getConvModelData(12);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(11);
    cb12_20 = CPUFusionCBOp::create(conv_param, bn_data,
                                  stemp_2, stemp_0,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb12_20);
    // add5
    add5_21 = CPUAddOp::create(stemp_0, stemp_1, stemp_2);
    ops_vec.push_back(add5_21);
    // ac11
    ac11_22 = CPUActiveOp::create(ACTIVE_RELU, stemp_2, stemp_0);
    ops_vec.push_back(ac11_22);
    // conv13 + bn12 + at12
    conv_data = resnet_model_data_ptr->getConvModelData(13);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(12);
    cba13_23 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    stemp_0, stemp_1,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba13_23);
    // conv14 + bn13
    conv_data = resnet_model_data_ptr->getConvModelData(14);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(13);
    cb14_24 = CPUFusionCBOp::create(conv_param, bn_data,
                                  stemp_1, stemp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb14_24);
    // add6
    add6_25 = CPUAddOp::create(stemp_0, stemp_2, stemp_1);
    ops_vec.push_back(add6_25);
    // ac13
    ac13_26 = CPUActiveOp::create(ACTIVE_RELU, stemp_1, stemp_0);
    ops_vec.push_back(ac13_26);
    // conv15 + bn14 + at14
    conv_param = {2, 2, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(15);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(14);
    cba15_27 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    stemp_0, sstemp_1,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba15_27);
    // conv16 + bn15
    conv_param = {1, 1, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(16);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(15);
    cb16_28 = CPUFusionCBOp::create(conv_param, bn_data,
                                  sstemp_1, sstemp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb16_28);
    // conv17
    conv_param = {2, 2, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(17);
    c17_29 = CPUConvOp::create(conv_param,
                                  stemp_0, sstemp_1,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(c17_29);
    // add7
    add7_30 = CPUAddOp::create(sstemp_1, sstemp_2, sstemp_0);
    ops_vec.push_back(add7_30);
    // ac15
    ac15_31 = CPUActiveOp::create(ACTIVE_RELU, sstemp_0, sstemp_1);
    ops_vec.push_back(ac15_31);
    // conv18 + bn16 + at16
    conv_param = {1, 1, 1, 1, 0, 0, 0};
    conv_data = resnet_model_data_ptr->getConvModelData(18);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(16);
    cba18_32 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    sstemp_1, sstemp_0,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba18_32);
    // conv19 + bn17
    conv_data = resnet_model_data_ptr->getConvModelData(19);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(17);
    cb19_33 = CPUFusionCBOp::create(conv_param, bn_data,
                                  sstemp_0, sstemp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb19_33);
    // add8
    add8_34 = CPUAddOp::create(sstemp_1, sstemp_2, sstemp_0);
    ops_vec.push_back(add8_34);
    // ac17
    ac17_35 = CPUActiveOp::create(ACTIVE_RELU, sstemp_0, sstemp_1);
    ops_vec.push_back(ac17_35);
    // conv20 + bn18 + at18
    conv_data = resnet_model_data_ptr->getConvModelData(20);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(18);
    cba20_36 = CPUFusionCBAOp::create(conv_param, bn_data, ACTIVE_RELU,
                                    sstemp_1, sstemp_0,
                                    conv_data.conv_kernel_ptr,
                                    conv_data.conv_bias_ptr);
    ops_vec.push_back(cba20_36);
    // conv21 + bn19
    conv_data = resnet_model_data_ptr->getConvModelData(21);
    bn_data = resnet_model_data_ptr->getBatchNormModelData(19);
    cb21_37 = CPUFusionCBOp::create(conv_param, bn_data,
                                  sstemp_0, sstemp_2,
                                  conv_data.conv_kernel_ptr,
                                  conv_data.conv_bias_ptr);
    ops_vec.push_back(cb21_37);
    // add9
    add9_38 = CPUAddOp::create(sstemp_1, sstemp_2, sstemp_0);
    ops_vec.push_back(add9_38);
    // ac19
    ac19_39 = CPUActiveOp::create(ACTIVE_RELU, sstemp_0, sstemp_1);
    ops_vec.push_back(ac19_39);
    // av_pool
    avpool_40 = CPUAVPoolingOp::create(sstemp_1, pool_temp);
    ops_vec.push_back(avpool_40);
    // dense
    conv_data = resnet_model_data_ptr->getDenseModelData();
    dense_42 = CPUFCOp::create(pool_temp, dense_temp,
                               conv_data.conv_kernel_ptr,
                               conv_data.conv_bias_ptr);
    ops_vec.push_back(dense_42);
}

void Executor::loadImage(std::string image_name,
                         int batch, int height, int width, int channel) {
  image_ptr = RamTensor::create(batch, height, width, channel, 1u);
  label_ptr = RamTensor::create(1, 1, 1, 10000, 1u);
  operation_ptr = RamTensor::create(batch, height, width, channel,
                                    image_ptr->data_ptr, 1u);
  // TODO: load image content
  FILE* fpr = fopen(image_name.c_str(), "rb");
  if (!fpr) {
       printf("Open error!");
       fclose(fpr);
  }

  for (int i = 0; i < 10000; i++) {
    fread((uint8_t*)(label_ptr->data_ptr) + i, sizeof(uint8_t), 1, fpr);
    fseek(fpr, 1, SEEK_CUR);
    fread((uint8_t*)(image_ptr->data_ptr) +
                     i * image_ptr->height * image_ptr->width * image_ptr->channel,
                     sizeof(uint8_t), 3072, fpr);
    fseek(fpr, 3072, SEEK_CUR);
	printf("Batch %d : %c\n", i+1, (uint8_t*)(label_ptr->data_ptr) + i);
  }

  fclose(fpr);
}

int Executor::compute(int batch_round) {

    int shift = batch_round * n_batch * image_ptr->height * image_ptr->width *
                image_ptr->channel;
    operation_ptr->reConfigTensor(n_batch, image_ptr->height,
                                  image_ptr->width, image_ptr->channel,
                                  (uint8_t*)(image_ptr->data_ptr) + shift,
                                  image_ptr->element_size);
    for(size_t i = 0; i < ops_vec.size(); i++)
        ops_vec[i]->forward_compute();
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
