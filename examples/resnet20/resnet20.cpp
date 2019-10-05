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
    int top_5 = 5;
    RVTensor::Executor::sptr sp =
            RVTensor::Executor::create(model_name, n_batch);

    sp->parseModel();

    sp->loadImage(input_name, 10000, 32, 32, 3);
    for (int i = 0; i < 10000/n_batch; i++) {
        sp->compute(i);
    }

    int acc = sp->inferenceResult(top_5);
    printf("top5 acc:%d%%\n", acc/100);

    return 0;
}
