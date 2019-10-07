/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include "include/core/executor.hpp"

int main(void)
{
    std::string model_name("resnet20.h5");
    std::string input_name("test_batch.bin");
    int n_batch = 500;
    int top_5 = 5;
    int top_1 = 1;
    RVTensor::Executor::sptr sp =
            RVTensor::Executor::create(model_name, n_batch);

    sp->parseModel();

    struct timeval start, end;
    int timeuse;
    gettimeofday(&start, NULL);

    sp->loadImage(input_name, 10000, 32, 32, 3);
    for (int i = 0; i < 10000/n_batch; i++) {
        sp->compute(i);
    }

    gettimeofday(&end, NULL);
    timeuse = 1000000 * ( end.tv_sec - start.tv_sec  ) + end.tv_usec - start.tv_usec;
    printf("inference time is %.3fs\n", (double)timeuse/1000000);

    int acc = sp->inferenceResult(top_5);
    printf("top5 acc:%d%%\n", acc/100);
    acc = sp->inferenceResult(top_1);
    printf("top1 acc:%d%%\n", acc/100);

    return 0;
}
