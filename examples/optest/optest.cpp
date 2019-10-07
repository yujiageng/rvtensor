/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <stdio.h>
#include <memory>
#include "include/core/executor.hpp"
#include "include/ops/add.hpp"
#include "include/ops/avpooling.hpp"
#include "include/ops/conv.hpp"
#include "include/core/types.hpp"
#include "include/ops/active.hpp"


void printData(char* log, float* data, int len) {
    printf("debug: %s\n",log);
    for (int i = 0; i < len; i++) {
        printf(" %f \n", data[i]);
    }
}

void printData(float* data, int len) {
    for (int i = 0; i < len; i++) {
        printf(" %f \n", data[i]);
    }
}
void printData(float* data, int n, int c, int h, int w) {
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
    	    for (int hi = 0; hi < h; hi++) {
    		for (int wi = 0; wi < w; wi++) {
        	    printf(" %f \n", data[ni*c*h*w + ci*h*w + hi*w + wi]);
    		}
    	    }
    	}
    }
}
void printData(char* log,float* data, int n, int c, int h, int w) {
    printf("hpdebug :%s\n",log);
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
    	    for (int hi = 0; hi < h; hi++) {
    		for (int wi = 0; wi < w; wi++) {
        	    printf(" %f ", data[ni*c*h*w + ci*h*w + hi*w + wi]);
    		}
		printf("\n");
    	    }
	    printf("\n");
    	}
	printf("\n");
    }
}
void initData(float* data, int len, float v) {
    for (int i = 0; i < len; i++) {
        ((float*)data)[i] = v;
    }
}
void initData(float* data, int len) {
    for (int i = 0; i < len; i++) {
        ((float*)data)[i] = i%4;
    }
}
void initData(float* data, int n, int c, int h, int w,float v) {
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
    	    for (int hi = 0; hi < h; hi++) {
    		for (int wi = 0; wi < w; wi++) {
        	    data[ni*c*h*w + ci*h*w + hi*w + wi] = v;
    		}
    	    }
    	}
    }
}
void test_addop(){
   auto input1 = RVTensor::RamTensor::create(2, 3, 3, 3);
   auto input2 = RVTensor::RamTensor::create(2, 3, 3, 3);
   auto output = RVTensor::RamTensor::create(2, 3, 3, 3);
   float* data1 = reinterpret_cast<float*>(input1->data_ptr);
   initData(data1,54,1);
   float* data2 = reinterpret_cast<float*>(input2->data_ptr);
   initData(data2,54,2);
   float* data3 = reinterpret_cast<float*>(output->data_ptr);
   memset(data3, 0, 54 * sizeof(float));
   
   printData("data1",data1,2,3,3,3);
   printData("data2",data2,2,3,3,3);

   auto addop = RVTensor::CPUAddOp::create(input1, input2, output);
   addop->forward_compute();
   printData("data3",data3,2,3,3,3);

}


void test_avpoolingop(){
   auto input1 = RVTensor::RamTensor::create(2, 3, 4, 4);
   auto output = RVTensor::RamTensor::create(2, 3, 1, 1);
   float* data1 = reinterpret_cast<float*>(input1->data_ptr);
   initData(data1,2*3*4*4);
   float* data3 = reinterpret_cast<float*>(output->data_ptr);
   memset(data3, 0, 2*3 * sizeof(float));
   
   printData("data1",data1,2,3,4,4);

   auto addop = RVTensor::CPUAVPoolingOp::create(input1, output);
   addop->forward_compute();
   printData("data3",data3,2,3,1,1);

}

void test_convop(){
  
   RVTensor::ConvParam conv_param = {1, 1, 1, 1, 0, 0, 0};
   auto input1 = RVTensor::RamTensor::create(2, 3, 4, 4);
   auto kernel = RVTensor::FlashTensor::create(1, 3, 2, 2);
   auto output = RVTensor::RamTensor::create(2, 1, 3, 3);

   float* data1 = reinterpret_cast<float*>(input1->data_ptr);
   initData(data1,2*3*4*4,2.0);

   float* ker_data = (float*)malloc(3*2*2 * sizeof(float));
   initData(ker_data,1*3*2*2,1.0);
   kernel->bindModelData(ker_data, 3*2*2* sizeof(float));

   float* data3 = reinterpret_cast<float*>(output->data_ptr);
   memset(data3, 0, 2*1*3*3 * sizeof(float));
   
   printData("data1",data1,2,3,4,4);
   printData("kernel",ker_data,1,3,2,2);

   auto convop = RVTensor::CPUConvOp::create(conv_param,input1, output, kernel);
   convop->forward_compute();
   printData("data3",data3,2,1,3,3);

}


void test_activeop(){
   RVTensor::ActiveType at = RVTensor::ActiveType::ACTIVE_RELU;
   auto input1 = RVTensor::RamTensor::create(2, 3, 4, 4);
   auto output = RVTensor::RamTensor::create(2, 3, 4, 4);
   float* data1 = reinterpret_cast<float*>(input1->data_ptr);
   initData(data1,2*3*4*4);
   float* data3 = reinterpret_cast<float*>(output->data_ptr);
   memset(data3, 0, 2*3*4*4* sizeof(float));
   
   printData("data1",data1,2,3,4,4);

   auto activeop = RVTensor::CPUActiveOp::create(at,input1, output);
   activeop->forward_compute();
   printData("data3",data3,2,3,4,4);

}

int main(void)
{
    std::string model_name("resnet20.h5");
    std::string input_name("test_batch.bin");
    int n_batch = 500;
    int top_5 = 5;
    printf("hpdebug ----\n\n\n");
    //test_addop();
    //test_avpoolingop();
    //test_convop();
    test_activeop();
    
    return 0;
}
