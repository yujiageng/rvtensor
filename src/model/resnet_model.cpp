/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/model/resnet_model.hpp"
#include "include/core/tensor.hpp"

namespace RVTensor {

ResnetModelData::sptr ResnetModelData::create() {
  return std::make_shared<ResnetModelData>();
}

inline ResnetModelData::ResnetModelData() {
    bn_model_datas.reserve(19);
    conv_model_datas.reserve(21);

    for (size_t i = 0; i < 7; i++) {
        bn_model_datas[i].bn_beta_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        bn_model_datas[i].bn_gamma_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        bn_model_datas[i].bn_mean_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        bn_model_datas[i].bn_variance_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        if (i == 0) {
            conv_model_datas[i].conv_kernel_ptr = FlashTensor::create(3, 3, 3, 16, 4u);
        } else {
            conv_model_datas[i].conv_kernel_ptr = FlashTensor::create(3, 3, 16, 16, 4u);
        }
            conv_model_datas[i].conv_bias_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
    }
    for (size_t i = 7; i < 13; i++) {
        bn_model_datas[i].bn_beta_ptr = FlashTensor::create(1, 1, 1, 32, 4u);
        bn_model_datas[i].bn_gamma_ptr = FlashTensor::create(1, 1, 1, 32, 4u);
        bn_model_datas[i].bn_mean_ptr = FlashTensor::create(1, 1, 1, 32, 4u);
        bn_model_datas[i].bn_variance_ptr = FlashTensor::create(1, 1, 1, 32, 4u);
    }
    for (size_t i = 13; i < 19; i++) {
        bn_model_datas[i].bn_beta_ptr = FlashTensor::create(1, 1, 1, 64, 4u);
        bn_model_datas[i].bn_gamma_ptr = FlashTensor::create(1, 1, 1, 64, 4u);
        bn_model_datas[i].bn_mean_ptr = FlashTensor::create(1, 1, 1, 64, 4u);
        bn_model_datas[i].bn_variance_ptr = FlashTensor::create(1, 1, 1, 64, 4u);
    }
    for (size_t i = 7; i < 14; i++) {
        conv_model_datas[i].conv_bias_ptr = FlashTensor::create(1, 1, 1, 32, 4u);
    }
    for (size_t i = 14; i < 21; i++) {
        conv_model_datas[i].conv_bias_ptr = FlashTensor::create(1, 1, 1, 64, 4u);
    }
    conv_model_datas[7].conv_kernel_ptr = FlashTensor::create(3, 3, 16, 32, 4u);
    conv_model_datas[8].conv_kernel_ptr = FlashTensor::create(3, 3, 32, 32, 4u);
    conv_model_datas[9].conv_kernel_ptr = FlashTensor::create(1, 1, 16, 32, 4u);
    conv_model_datas[10].conv_kernel_ptr = FlashTensor::create(3, 3, 32, 32, 4u);
    conv_model_datas[11].conv_kernel_ptr = FlashTensor::create(3, 3, 32, 32, 4u);
    conv_model_datas[12].conv_kernel_ptr = FlashTensor::create(3, 3, 32, 32, 4u);
    conv_model_datas[13].conv_kernel_ptr = FlashTensor::create(3, 3, 32, 32, 4u);
    conv_model_datas[14].conv_kernel_ptr = FlashTensor::create(3, 3, 32, 64, 4u);
    conv_model_datas[15].conv_kernel_ptr = FlashTensor::create(3, 3, 64, 64, 4u);
    conv_model_datas[16].conv_kernel_ptr = FlashTensor::create(1, 1, 32, 64, 4u);
    conv_model_datas[17].conv_kernel_ptr = FlashTensor::create(3, 3, 64, 64, 4u);
    conv_model_datas[18].conv_kernel_ptr = FlashTensor::create(3, 3, 64, 64, 4u);
    conv_model_datas[19].conv_kernel_ptr = FlashTensor::create(3, 3, 64, 64, 4u);
    conv_model_datas[20].conv_kernel_ptr = FlashTensor::create(3, 3, 64, 64, 4u);

    dense_model_data.conv_kernel_ptr = FlashTensor::create(1, 1, 64, 10, 4u);
    dense_model_data.conv_bias_ptr = FlashTensor::create(1, 1, 1, 10, 4u);
}

inline ResnetModelData::~ResnetModelData() {

    for (size_t i = 0; i < 19; i++) {
        free(bn_model_datas[i].bn_beta_ptr->data_ptr);
        free(bn_model_datas[i].bn_gamma_ptr->data_ptr);
        free(bn_model_datas[i].bn_mean_ptr->data_ptr);
        free(bn_model_datas[i].bn_variance_ptr->data_ptr);
        free(conv_model_datas[i].conv_kernel_ptr->data_ptr);
        free(conv_model_datas[i].conv_bias_ptr->data_ptr);
    }
}

void ResnetModelData::openModelFile(const char* filename) {
   file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
}

void ResnetModelData::closeModelFile() {
   H5Fclose(file);
}

float* ResnetModelData::getWeightByID(const char* weightID, int* count) {
   int ndims, n, i;
   hid_t space, dset;
   herr_t status;
   hsize_t dims[4];
   dset = H5Dopen(file, weightID, H5P_DEFAULT);
   space = H5Dget_space(dset);
   ndims = H5Sget_simple_extent_dims(space, dims, NULL);
   n = 1;
   for (i = 0; i < ndims; i++) {
       n = n * (int)dims[i];
   }
   *count = n;
   float* weight = (float *)malloc(n * sizeof (float));
   status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, weight);
   status = H5Dclose(dset);
   status = H5Sclose(space);
   return weight;
}

BnModelData ResnetModelData::getBatchNormModelData(int index) {
   char buf[90];
   int count;
   sprintf(buf,
           "model_weights/batch_normalization_%d/batch_normalization_%d/beta:0",
           index, index);
   float* beta = getWeightByID(buf, &count);
   (bn_model_datas[index - 1].bn_beta_ptr)->bindModelData(
                                   (void*)beta, count * sizeof(float));

   sprintf(buf,
           "model_weights/batch_normalization_%d/batch_normalization_%d/gamma:0",
           index, index);
   float* gamma = getWeightByID(buf, &count);
   (bn_model_datas[index - 1].bn_gamma_ptr)->bindModelData(
                                   (void*)gamma, count * sizeof(float));

   sprintf(buf,
           "model_weights/batch_normalization_%d/batch_normalization_%d/moving_mean:0",
           index, index);
   float* mean = getWeightByID(buf, &count);
   (bn_model_datas[index - 1].bn_mean_ptr)->bindModelData(
                                   (void*)mean, count * sizeof(float));

   sprintf(buf,
           "model_weights/batch_normalization_%d/batch_normalization_%d/moving_variance:0",
           index, index);
   float* variance = getWeightByID(buf, &count);
   (bn_model_datas[index - 1].bn_variance_ptr)->bindModelData(
                                   (void*)variance, count * sizeof(float));

   return bn_model_datas[index - 1];
}

ConvModelData ResnetModelData::getConvModelData(int index) {
   char buf[90];
   int count;
   sprintf(buf, "model_weights/conv2d_%d/conv2d_%d/kernel:0", index, index);
   float* kernel = getWeightByID(buf, &count);
   conv_model_datas[index - 1].conv_kernel_ptr->bindModelData(
                                    (void*)kernel, count * sizeof(float));

   sprintf(buf, "model_weights/conv2d_%d/conv2d_%d/bias:0", index, index);
   float* bias = getWeightByID(buf, &count);
   conv_model_datas[index - 1].conv_bias_ptr->bindModelData(
                                    (void*)bias, count * sizeof(float));

   return conv_model_datas[index - 1];
}

ConvModelData ResnetModelData::getDenseModelData() {
   char buf[90];
   int count;
   sprintf(buf, "model_weights/dense_1/dense_1/kernel:0");
   float* kernel = getWeightByID(buf, &count);
   dense_model_data.conv_kernel_ptr->bindModelData(
                                    (void*)kernel, count * sizeof(float));

   sprintf(buf, "model_weights/dense_1/dense_1/bias:0");
   float* bias = getWeightByID(buf, &count);
   dense_model_data.conv_bias_ptr->bindModelData(
                                    (void*)bias, count * sizeof(float));

   return dense_model_data;
}

}
