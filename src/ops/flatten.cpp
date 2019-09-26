/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/flatten.hpp"
#include <stdlib.h>
namespace RVTensor {
CPUFlattenOp::sptr CPUFlattenOp::create() {
  return std::make_shared<CPUFlattenOp>();
}

CPUFlattenOp::sptr CPUFlattenOp::create(RamTensor::sptr input,
                                        RamTensor::sptr output) {
  CPUFlattenOp::sptr ptr = std::make_shared<CPUFlattenOp>(input, output);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUFlattenOp::CPUFlattenOp() : Operation({}, {}) {}

inline CPUFlattenOp::CPUFlattenOp(RamTensor::sptr input, RamTensor::sptr output)
    : Operation({input}, {output}) {}

inline CPUFlattenOp::~CPUFlattenOp() {}

inline void CPUFlattenOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t*>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t*>(output_tensor->data_ptr);

  int batch = input_tensor->n_batch;
  int channel = input_tensor->channel;
  int height = input_tensor->height;
  int width = input_tensor->width;
  int count = input_tensor->count();

  float* swap = (float*)calloc(count, sizeof(float));
  int i, c, b;
  for (b = 0; b < batch; ++b) {
    for (c = 0; c < channel; ++c) {
      for (i = 0; i < width * height; ++i) {
        int i1 = b * width * height * channel + c * width * height + i;
        int i2 = b * width * height * channel + i * channel + c;

        swap[i1] = input[i2];
      }
    }
  }
  memcpy(output, swap, count * sizeof(float));
  free(swap);
}

}  // namespace RVTensor
