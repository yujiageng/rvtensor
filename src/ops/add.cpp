/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/add.hpp"
#include "assert.h"

namespace RVTensor {

CPUAddOp::sptr CPUAddOp::create() { return std::make_shared<CPUAddOp>(); }

CPUAddOp::sptr CPUAddOp::create(RamTensor::sptr input1, RamTensor::sptr input2,
                                RamTensor::sptr output) {
  CPUAddOp::sptr ptr = std::make_shared<CPUAddOp>(input1, input2, output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUAddOp::CPUAddOp() : Operation({}, {}) {}

inline CPUAddOp::CPUAddOp(RamTensor::sptr input1, RamTensor::sptr input2,
                          RamTensor::sptr output)
    : Operation({input1, input2}, {output}) {}

inline CPUAddOp::~CPUAddOp() {}

inline void CPUAddOp::forward_compute() {
  auto input_tensor1 = getInputs()[0];
  auto input_tensor2 = getInputs()[1];
  auto output_tensor = getOutputs()[0];

  uint8_t* input1 = reinterpret_cast<uint8_t*>(input_tensor1->data_ptr);
  uint8_t* input2 = reinterpret_cast<uint8_t*>(input_tensor2->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t*>(output_tensor->data_ptr);
  // 两个feature map x相加
  // TODO: complete it
  int batch1 = input_tensor1->n_batch;
  int channel1 = input_tensor1->channel;
  int height1 = input_tensor1->height;
  int width1 = input_tensor1->width;

  int batch2 = input_tensor2->n_batch;
  int channel2 = input_tensor2->channel;
  int height2 = input_tensor2->height;
  int width2 = input_tensor2->width;
  assert(batch1 == batch2 && height1 == height2 && width1 == width2 &&
         channel1 == channel2);

  int i, j, k, b;
  for (b = 0; b < batch2; ++b) {
    for (k = 0; k < channel1; ++k) {
      for (j = 0; j < height1; ++j) {
        for (i = 0; i < width1; ++i) {
          int index = i + width1 * (j + height1 * (k + channel1 * b));
          output[index] = input2[index] + input1[index];
        }
      }
    }
  }
}

}  // namespace RVTensor
