#include <iostream>                                                             
#include <stdio.h>
#include <stdlib.h>
#include "Datasets.hpp"

using namespace std;

int main(void)
{
    Datasets datasets;
    datasets.openDatasets("../../../cifar-10-batches-bin/test_batch.bin");
    for(int i=0; i<20; i++){
	datasets.readBatchDatasets();
        for(int j=0; j<500; j++){
	    printf("Batch %d - label %d : %d\n", i+1, j+1, datasets.batch_datasets[j*3073]);
	}
    }
    datasets.closeDatasets();
    return 0;
}

