/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <stdio.h>
#include <memory>
#include "include/core/executor.hpp"

uint8_t g_ai_buf[500 * 32 * 32 * 3] __attribute__((aligned(128)));

int main(void)
{
    std::string model_name("resnet20");
    std::string input_name("test_batch.bin");

    RVTensor::Executor::sptr sp = RVTensor::Executor::create(model_name);

    sp->parseModel();

    sp->loadImage(input_name, g_ai_buf, 500, 32, 32, 3);

    sp->compute();

    sp->inferenceResult();

    return 0;
}
