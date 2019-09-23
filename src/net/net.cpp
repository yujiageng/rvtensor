/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/net/net.hpp"

namespace RVTensor {

Net::sptr Net::create() {
    return std::make_shared<Net>();
}

Net::sptr Net::create(std::string model_name) {
    return std::make_shared<Net>(model_name);
}

Net::Net() : inputs_({}), outputs_({}) {}

Net::Net(std::string model_name) {}

Net::~Net() {}

std::vector<RamTensor::sptr> Net::getInputs() {
    return inputs_;
}

std::vector<RamTensor::sptr> Net::getOutputs() {
    return outputs_;
}

}  // namespace RVTensor
