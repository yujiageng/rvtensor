#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#endif
#include <iostream>

#include<stdio.h>
using namespace std;
using std::cout;
using std::endl;
#include <ResnetModel.h>
int main (void)
{
    float *rdata;                    /* Read buffer */
    int i;
    int *size;
    size =(int *) malloc (sizeof (int));
    ResnetModel model;
    model.openModelFile("resnet20.h5");
    rdata = model.getWeightByID("/model_weights/conv2d_1/conv2d_1/kernel:0", size);
    model.close_ModelFile();
    for (i=0; i<432; i++) {
            printf ("%20.18f ",rdata[i]);
        }
    cout << "size: "<<size[0] << endl;
    free (rdata);
    free (size);
    return 0;
}

