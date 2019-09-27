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
    float* weight_value;
    ResnetModel model;
    model.openModelFile("resnet20.h5");
    model.getWeightByID("/model_weights/conv2d_1/conv2d_1/kernel:0");
    cout << "size: "<<model.weight_size[0] << endl;
    weight_value = model.weight;
    for (int i=0; i< model.weight_size[0]; i++)
    {
        printf ("%20.18f ", weight_value[i]);
    }
    model.close_ModelFile();//close model file
    return 0;
}

