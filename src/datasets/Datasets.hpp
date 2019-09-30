/*  The MIT License                                                                                        
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_Datasets_Datasets_HPP_
#define INCLUDE_Datasets_Datasets_HPP_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

class Datasets
{
 public:
    void openDatasets(const char* filename);
    void readBatchDatasets();
    void closeDatasets();

    char* batch_datasets;
    FILE *fpr;
};
#endif // INCLUDE_Datasets_Datasets_HPP_
