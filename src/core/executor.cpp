/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/executor.hpp"
#include "include/core/tensor.hpp"

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
    network_ptr = Net::create(model_name);
}

void Executor::loadImage(std::string image_name, uint8_t* ai_buf,
                         int channel, int height, int width) {
  image_ptr = RamTensor::create(1, channel, height, width,
                                reinterpret_cast<void*>(ai_buf), 1u);
  // TODO: load image content

}

int Executor::compute() {
    network_ptr->compute({image_ptr});
    return 0;
}

//int Executor::inferenceResult(void* result_buf, uint64_t size) {
int Executor::inferenceResult() {
    // TODO: image classfication to resnet

    return 0;
}

Executor::~Executor() {}

}  // namespace RVTensor
