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
    void closeModelFile();
    BnModelData getBatchNormModelData(int index);
    ConvModelData getConvModelData(int index);
    ConvModelData getDenseModelData();

 private:
    hid_t file;
    float* getWeightByID(const char* weightID, int* count);
    std::vector<BnModelData> bn_model_datas;
    std::vector<ConvModelData> conv_model_datas;
    ConvModelData dense_model_data;
};

}  // namespace RVTensor

#endif  // INCLUDE_Model_RESNET_MODEL_HPP_
