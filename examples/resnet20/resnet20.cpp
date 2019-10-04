/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <stdio.h>
#include <memory>
#include "include/core/executor.hpp"

int main(void)
{
    std::string model_name("resnet20.h5");
    std::string input_name("test_batch.bin");
    int n_batch = 500;

    RVTensor::Executor::sptr sp =
            RVTensor::Executor::create(model_name, n_batch);

    sp->parseModel();

    sp->loadImage(input_name, 10000, 32, 32, 3);
    for (int i = 0; i < 10000/n_batch; i++) {
        printf("i:%d\n", i);
        sp->compute(i);
        sp->inferenceResult();
    }

    return 0;
}
