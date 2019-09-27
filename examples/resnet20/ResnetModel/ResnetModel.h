#ifndef _ResnetModel_H_
#define _ResnetModel_H_

#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class ResnetModel
{
    public:
        hid_t file;
        float* weight;
        int* weight_size;

        void openModelFile(const char* filename);
        void close_ModelFile();
        void getWeightByID(const char* weightID);
};

#endif
