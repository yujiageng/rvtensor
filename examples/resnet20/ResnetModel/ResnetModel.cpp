#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using namespace std;
using std::cout;
using std::endl;
#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ResnetModel.h"

void ResnetModel::openModelFile(const char* filename)
{  
   file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);

}
void ResnetModel::close_ModelFile()
{
   H5Fclose(file);
}
float* ResnetModel::getWeightByID(const char* weightID, int* size)
{
   float *rdata;
   int ndims,n,i;
   hid_t space, dset;
   herr_t status;
   hsize_t dims[4];
   dset = H5Dopen (file, weightID, H5P_DEFAULT);
   space = H5Dget_space (dset);
   ndims = H5Sget_simple_extent_dims (space, dims, NULL);
   n=1;
   for (i=0; i<ndims; i++ )
   {
         n = n*(int)dims[i];
   }
   size[0] = n;
   rdata = (float *) malloc (n * sizeof (float));
   status = H5Dread (dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
   status = H5Dclose (dset);
   status = H5Sclose (space);
   return rdata;
}

