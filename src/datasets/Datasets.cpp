/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)                         
 *  All rights reserved.
 *
 */
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include "Datasets.hpp"
#define IMAGE_SIZE 3073
#define N_Batch 500

void Datasets::openDatasets(const char* filename) {
    fpr = fopen(filename,"rb");
    if ( fpr == NULL ){
       printf("Open error!");
       fclose(fpr);

    }
    int size = int(N_Batch) * int(IMAGE_SIZE);
    batch_datasets = (char *)malloc(size * sizeof (char)); 
}

void Datasets::readBatchDatasets() {
    fread(batch_datasets, sizeof(char), int(N_Batch) * int(IMAGE_SIZE), fpr);
}

void Datasets::closeDatasets(){
    free(batch_datasets);
    fclose(fpr);
}


