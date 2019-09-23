/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_NET_NET_HPP_
#define INCLUDE_NET_NET_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"

namespace RVTensor {

class Net {
 public:
    using sptr = std::shared_ptr<Net>;
    static sptr create();
    static sptr create(std::string model_name);

    /**
     * Constructor & Deconstructor
     */
    Net();
    Net(std::string model_name);
    ~Net();
    Net& operator=(const Net& op);

    /**
     * get inputs/outputs
     */
    std::vector<RamTensor::sptr> getInputs();
    std::vector<RamTensor::sptr> getOutputs();

    /**
     * inference
     */
    virtual void compute(std::vector<RamTensor::sptr> inputs) {}

 private:
    std::vector<RamTensor::sptr> inputs_;
    std::vector<RamTensor::sptr> outputs_;

};

}  // namespace RVTensor

#endif  // INCLUDE_NET_NET_HPP_
