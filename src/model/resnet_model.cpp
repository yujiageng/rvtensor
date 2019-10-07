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

ResnetModelData::ResnetModelData() {
    bn_model_datas.reserve(19);
    conv_model_datas.reserve(21);

    for (size_t i = 0; i < 7; i++) {
        bn_model_datas[i].bn_beta_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        bn_model_datas[i].bn_gamma_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        bn_model_datas[i].bn_mean_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        bn_model_datas[i].bn_variance_ptr = FlashTensor::create(1, 1, 1, 16, 4u);
        if (i == 0) {
            conv_model_datas[i].conv_kernel_ptr = FlashTensor::create(16, 3, 3, 3, 4u);
        } else {
            conv_model_datas[i].conv_kernel_ptr = FlashTensor::create(16, 16, 3, 3, 4u);
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

    conv_model_datas[7].conv_kernel_ptr = FlashTensor::create(32, 16, 3, 3, 4u);
    conv_model_datas[8].conv_kernel_ptr = FlashTensor::create(32, 32, 3, 3, 4u);
    conv_model_datas[9].conv_kernel_ptr = FlashTensor::create(32, 16, 1, 1, 4u);

    conv_model_datas[10].conv_kernel_ptr = FlashTensor::create(32, 32, 3, 3, 4u);
    conv_model_datas[11].conv_kernel_ptr = FlashTensor::create(32, 32, 3, 3, 4u);
    conv_model_datas[12].conv_kernel_ptr = FlashTensor::create(32, 32, 3, 3, 4u);
    conv_model_datas[13].conv_kernel_ptr = FlashTensor::create(32, 32, 3, 3, 4u);

    conv_model_datas[14].conv_kernel_ptr = FlashTensor::create(64, 32, 3, 3, 4u);
    conv_model_datas[15].conv_kernel_ptr = FlashTensor::create(64, 64, 3, 3, 4u);
    conv_model_datas[16].conv_kernel_ptr = FlashTensor::create(64, 32, 1, 1, 4u);
    conv_model_datas[17].conv_kernel_ptr = FlashTensor::create(64, 64, 3, 3, 4u);
    conv_model_datas[18].conv_kernel_ptr = FlashTensor::create(64, 64, 3, 3, 4u);
    conv_model_datas[19].conv_kernel_ptr = FlashTensor::create(64, 64, 3, 3, 4u);
    conv_model_datas[20].conv_kernel_ptr = FlashTensor::create(64, 64, 3, 3, 4u);

    dense_model_data.conv_kernel_ptr = FlashTensor::create(1, 1, 10, 64, 4u);
    dense_model_data.conv_bias_ptr = FlashTensor::create(1, 1, 1, 10, 4u);
}

ResnetModelData::~ResnetModelData() {

    for (size_t i = 0; i < 19; i++) {
        free(bn_model_datas[i].bn_beta_ptr->data_ptr);
        free(bn_model_datas[i].bn_gamma_ptr->data_ptr);
        free(bn_model_datas[i].bn_mean_ptr->data_ptr);
        free(bn_model_datas[i].bn_variance_ptr->data_ptr);
    }
    for (size_t i = 0; i < 21; i++) {
        free(conv_model_datas[i].conv_kernel_ptr->data_ptr);
        free(conv_model_datas[i].conv_bias_ptr->data_ptr);
    }
    free(dense_model_data.conv_kernel_ptr->data_ptr);
    free(dense_model_data.conv_bias_ptr->data_ptr);
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

void ResnetModelData::transpose(float* input, int n, int c, int h, int w) {
    float* temp = (float*)calloc(n * c * h * w, sizeof(float));
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            for (int hi = 0; hi < h; hi++) {
                for (int wi = 0; wi < w; wi++) {
                    int index_t = ni * c * h * w + ci * h * w + hi * w + wi;
                    int index_i = hi * w * c * n + wi * c * n + ci * n + ni;
                    temp[index_t] = input[index_i];
                }
            }
        }
    }
    memcpy(input, temp, n * c * h * w * sizeof(float));
    free(temp);
}

void ResnetModelData::transpose(float* input, int h, int w) {
    float* temp = (float*)calloc(h * w, sizeof(float));
    for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
            int index_t = hi * w + wi;
            int index_i = wi * h + hi;
            temp[index_t] = input[index_i];
        }
    }
    memcpy(input, temp, h * w * sizeof(float));
    free(temp);
}

ConvModelData ResnetModelData::getConvModelData(int index) {
   char buf[90];
   int count;
   sprintf(buf, "model_weights/conv2d_%d/conv2d_%d/kernel:0", index, index);
   float* kernel = getWeightByID(buf, &count);
   // transpose(kernel, conv_model_datas[index - 1].conv_kernel_ptr->n_batch,
   //         conv_model_datas[index - 1].conv_kernel_ptr->channel,
   //         conv_model_datas[index - 1].conv_kernel_ptr->height,
   //         conv_model_datas[index - 1].conv_kernel_ptr->width);
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
   // transpose(kernel, dense_model_data.conv_kernel_ptr->height,
   //           dense_model_data.conv_kernel_ptr->width);
   dense_model_data.conv_kernel_ptr->bindModelData(
                                    (void*)kernel, count * sizeof(float));

   sprintf(buf, "model_weights/dense_1/dense_1/bias:0");
   float* bias = getWeightByID(buf, &count);
   dense_model_data.conv_bias_ptr->bindModelData(
                                    (void*)bias, count * sizeof(float));

   return dense_model_data;
}

}
