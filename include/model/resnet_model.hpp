/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_Model_RESNET_MODEL_HPP_
#define INCLUDE_Model_RESNET_MODEL_HPP_

#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "include/core/tensor.hpp"
#include "include/core/types.hpp"

// #define RESNET_NUM 19

namespace RVTensor {

class ResnetModelData
{
 public:
    using sptr = std::shared_ptr<ResnetModelData>;
    static sptr create();
    ResnetModelData();
    ~ResnetModelData();
    ResnetModelData& operator=(const ResnetModelData& m);

    void openModelFile(const char* filename);
    void close_ModelFile();
    bn_model_data getBatchNormModelData(int index);
    conv_model_data getConvModelData(int index);
    conv_model_data getDenseModelData();

 private:
    hid_t file;
    float* getWeightByID(const char* weightID, int* count);
    std::vector<bn_model_data> bn_model_datas;
    std::vector<conv_model_data> conv_model_datas;
    conv_model_data dense_model_data;
};

}  // namespace RVTensor

#endif  // INCLUDE_Model_RESNET_MODEL_HPP_
