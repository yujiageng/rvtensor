/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/fusion_cb.hpp"
#include "math.h"
namespace RVTensor {

CPUFusionCBOp::sptr CPUFusionCBOp::create() {
  return std::make_shared<CPUFusionCBOp>();
}

CPUFusionCBOp::sptr CPUFusionCBOp::create(
    ConvParam conv_param, BnModelData bn_param, RamTensor::sptr input,
    RamTensor::sptr output, FlashTensor::sptr weight, FlashTensor::sptr bias) {
  CPUFusionCBOp::sptr ptr = std::make_shared<CPUFusionCBOp>(
      conv_param, bn_param, input, output, weight, bias);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUFusionCBOp::CPUFusionCBOp()
    : Operation({}, {}),
      conv_param_({0, 0, 1, 1, 0, 0, false}),
      bn_param_({nullptr, nullptr, nullptr, nullptr}),
      weight_(nullptr),
      bias_(nullptr) {}

inline CPUFusionCBOp::CPUFusionCBOp(
    ConvParam conv_param, BnModelData bn_param, RamTensor::sptr input,
    RamTensor::sptr output, FlashTensor::sptr weight, FlashTensor::sptr bias)
    : Operation({input}, {output}),
      conv_param_(conv_param),
      bn_param_(bn_param),
      weight_(weight),
      bias_(bias) {}

inline CPUFusionCBOp::~CPUFusionCBOp() {}

inline void CPUFusionCBOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  float* input = reinterpret_cast<float*>(input_tensor->data_ptr);
  float* output = reinterpret_cast<float*>(output_tensor->data_ptr);
  float* weight = reinterpret_cast<float*>(weight_->data_ptr);
  float* bias = bias_ ? reinterpret_cast<float*>(bias_->data_ptr) : nullptr;
  // auto tmp_output = output_tensor;
  // //[TODO] create a tmp_output
  // auto conv =
  //     CPUConvOp::create(conv_param_, input_tensor, tmp_output, weight_,
  //     bias_);
  // conv->forward_compute();
  // //[TODO] use tmp_output as input
  // auto bn = CPUBnOp::create(bn_param_, tmp_output, output_tensor);
  // bn->forward_compute();

  int ni = input_tensor->n_batch;
  int ci = input_tensor->channel;
  int hi = input_tensor->height;
  int wi = input_tensor->width;

  int co = output_tensor->channel;
  int ho = output_tensor->height;
  int wo = output_tensor->width;
  int sh = conv_param_.sh;
  int sw = conv_param_.sw;
  int kh = weight_->height;
  int kw = weight_->width;
  int dh = conv_param_.dh;
  int dw = conv_param_.dw;
  int ph = conv_param_.ph;
  int pw = conv_param_.pw;

  // bn_param;
  float* mean = (float*)(bn_param_.bn_mean_ptr->data_ptr);
  float* variance = (float*)(bn_param_.bn_variance_ptr->data_ptr);
  float* scales = (float*)calloc(co, sizeof(float));
  // 卷积核的个数 = 输出的通道数
  int m = output_tensor->channel;
  /*卷积核 元素的个数,l.size=卷积核的尺寸，l.c= 卷积核的通道*/
  int k = kh * kw * ci;
  /*该层输出单通道的特征图的尺寸*/
  int n = ho * wo;

  /*循环batch中的每个输入*/
  for (int i = 0; i < ni; ++i) {
    /*用于存储经im2col转换后的输入特征矩阵*/
    float* a = (float*)calloc(1024, sizeof(float));

    /*a是指向当前层所有卷积核的*/
    float* b = weight;
    /*输出特征图个数*/
    float* c = output + i * n * m;
    float* im = input + i * ci * hi * wi;
    /*如果是1*1的卷积，那么不用对输入特征进行转化*/
    if (kh * kw == 1) {
      a = im;
    } else {
      /*对输入特征进行转化*/
      im2col_cpu(im, ci, hi, wi, kh, sh, ph, a);
    }
    coppersmith_winograd(a, b, c, m, n, k, k, n, n);
  }
  // BN input
  auto temp_tensor = output_tensor;
  float* temp_input = reinterpret_cast<float*>(output_tensor->data_ptr);
  // BN
  copy_cpu(input_tensor->count(), temp_input, 1, output, 1);
  // 归一化
  normalize_cpu(output, &mean[0], &variance[0], ni, co, ho * wo);
  // scales大小为out_c的数组，值全是1
  scale_bias(output, scales, ni, co, ho * wo);
  add_bias(output, bias, ni, co, ho * wo);
}
inline void CPUFusionCBOp::mm_generate(float* matA, float* matB,
                                       float* matC, const int M, const int N,
                                       const int K, const int strideA,
                                       const int strideB, const int strideC) {
  //    printf("into mm_generate\n");

  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        matC[i * strideC + j] += matA[i * strideA + k] * matB[k * strideB + j];
      }
    }
  }
}

inline void CPUFusionCBOp::coppersmith_winograd(float* matA, float* matB,
                                                float* matC, int M, int N,
                                                int K, int strideA, int strideB,
                                                int strideC) {
  // step 1:使用普通的矩阵
  if ((M <= 64) || (M % 2 != 0 || N % 2 != 0 || K % 2 != 0)) {
    return mm_generate(matA, matB, matC, M, N, K, strideA, strideB, strideC);
  }
  // matC = calloc(M * strideC, sizeof(float));
  int offset = 0;

  float* S1 = (float*)calloc((M / 2) * (K / 2), sizeof(float));
  float* S2 = (float*)calloc((M / 2) * (K / 2), sizeof(float));
  float* S3 = (float*)calloc((M / 2) * (K / 2), sizeof(float));
  float* S4 = (float*)calloc((M / 2) * (K / 2), sizeof(float));
  for (int i = 0; i < M / 2; i++) {
    for (int j = 0; j < K / 2; j++) {
      const int idx = i * K / 2 + j;
      // S1 = A21 + A22
      S1[idx] = matA[(i + M / 2) * strideA + j] +
                matA[(i + M / 2) * strideA + j + K / 2];
      // S2 = S1 - A11
      S2[idx] = S1[idx] - matA[i * strideA + j];
      // S3 = A11 - A21
      S3[idx] = matA[i * strideA + j] - matA[(i + M / 2) * strideA + j];
      // S4 = A12 - S2
      S4[idx] = matA[i * strideA + j + K / 2] - S2[idx];
    }
  }
  float* T1 = (float*)calloc((K / 2) * (N / 2), sizeof(float));
  float* T2 = (float*)calloc((K / 2) * (N / 2), sizeof(float));
  float* T3 = (float*)calloc((K / 2) * (N / 2), sizeof(float));
  float* T4 = (float*)calloc((K / 2) * (N / 2), sizeof(float));
  for (int i = 0; i < K / 2; i++) {
    for (int j = 0; j < N / 2; j++) {
      const int idx = i * N / 2 + j;
      // T1 = B21 - B11
      T1[idx] = matB[(i + K / 2) * strideB + j] - matB[i * strideB + j];
      // T2 = B22 - T1
      T2[idx] = matB[(i + K / 2) * strideB + j + N / 2] - T1[idx];
      // T3 = B22 - B12
      T3[idx] = matB[(i + K / 2) * strideB + j + N / 2] -
                matB[i * strideB + j + N / 2];
      // T4 = T2 - B21
      T4[idx] = T2[idx] - matB[(i + K / 2) * strideB + j];
    }
  }

  // M1 = A11*B11
  float* M1 = (float*)calloc((M / 2) * (N / 2), sizeof(float));
  {
    coppersmith_winograd(matA, matB, M1, M / 2, N / 2, K / 2, strideA, strideB,
                         N / 2);
  }

  // M2 = A12*B21
  float* M2 = (float*)calloc((M / 2) * (N / 2), sizeof(float));
  {
    coppersmith_winograd(matA + K / 2, matB + K * strideB / 2, M2, M / 2, N / 2,
                         K / 2, strideA, strideB, N / 2);
  }

  // M3 = S4*B22
  float* M3 = (float*)calloc((M / 2) * (N / 2), sizeof(float));
  {
    coppersmith_winograd(S4, matB + K * strideB / 2 + N / 2, M3, M / 2, N / 2,
                         K / 2, K / 2, strideB, N / 2);
  }

  // M4 = A22*T4
  float* M4 = (float*)calloc((M / 2) * (N / 2), sizeof(float));
  {
    coppersmith_winograd(matA + M * strideA / 2 + K / 2, T4, M4, M / 2, N / 2,
                         K / 2, strideA, N / 2, N / 2);
  }

  // M5 = S1*T1
  float* M5 = (float*)calloc((M / 2) * (N / 2), sizeof(float));
  {
    coppersmith_winograd(S1, T1, M5, M / 2, N / 2, K / 2, K / 2, N / 2, N / 2);
  }

  // M6 = S2*T2
  float* M6 = (float*)calloc((M / 2) * (N / 2), sizeof(float));
  {
    coppersmith_winograd(S2, T2, M6, M / 2, N / 2, K / 2, K / 2, N / 2, N / 2);
  }

  // M7 = S3*T3
  float* M7 = (float*)calloc((M / 2) * (N / 2), sizeof(float));
  {
    coppersmith_winograd(S3, T3, M7, M / 2, N / 2, K / 2, K / 2, N / 2, N / 2);
  }
  for (int i = 0; i < M / 2; i++) {
    for (int j = 0; j < N / 2; j++) {
      const int idx = i * N / 2 + j;
      // U1 = M1 + M2
      const auto U1 = M1[idx] + M2[idx];
      // U2 = M1 + M6
      const auto U2 = M1[idx] + M6[idx];
      // U3 = U2 + M7
      const auto U3 = U2 + M7[idx];
      // U4 = U2 + M5
      const auto U4 = U2 + M5[idx];
      // U5 = U4 + M3
      const auto U5 = U4 + M3[idx];
      // U6 = U3 - M4
      const auto U6 = U3 - M4[idx];
      // U7 = U3 + M5
      const auto U7 = U3 + M5[idx];

      // C11 = U1
      matC[i * strideC + j] = U1;
      // C12 = U5
      matC[i * strideC + j + N / 2] = U5;
      // C21 = U6
      matC[(i + M / 2) * strideC + j] = U6;
      // C22 = U7
      matC[(i + M / 2) * strideC + j + N / 2] = U7;
    }
  }
}

inline float CPUFusionCBOp::im2col_get_pixel(float* im, int height, int width,
                                             int channels, int row, int col,
                                             int channel, int pad) {
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width) return 0;
  return im[col + width * (row + height * channel)];
}

inline void CPUFusionCBOp::im2col_cpu(float* data_im, int channels,
                                      int height, int width, int ksize,
                                      int stride, int pad, float* data_col) {
  int c, h, w;
  // 计算卷基层输出图像的高 和宽
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  //计算卷积层输入单通道图像的数据容量
  int channels_col = channels * ksize * ksize;
  //循环每个卷积核的参数个数
  for (c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    // 用卷积核把图像遍历一遍
    for (h = 0; h < height_col; ++h) {
      for (w = 0; w < width_col; ++w) {
        int im_row = h_offset + h * stride;
        int im_col = w_offset + w * stride;
        int col_index = (c * height_col + h) * width_col + w;
        data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                               im_row, im_col, c_im, pad);
      }
    }
  }
}

inline void CPUFusionCBOp::copy_cpu(int N, float* X, int INCX, float* Y,
                                    int INCY) {
  int i;
  for (i = 0; i < N; ++i) Y[i * INCY] = X[i * INCX];
}
//归一化
inline void CPUFusionCBOp::normalize_cpu(float* x, float* mean,
                                         float* variance, int batch,
                                         int filters, int spatial) {
  int b, f, i;
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < spatial; ++i) {
      for (f = 0; f < filters; ++f) {
        int index = b * filters * spatial + f * spatial + i;
        //公式中的ε=.000001f
        x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
      }
    }
  }
}
inline void CPUFusionCBOp::scale_bias(float* output, float* scales, int batch,
                                      int n, int size) {
  // scales 全是1
  int i, j, b;
  for (b = 0; b < batch; ++b) {
    for (j = 0; j < size; ++j) {
      for (i = 0; i < n; ++i) {
        output[(b * n + i) * size + j] *= scales[i];
      }
    }
  }
}
inline void CPUFusionCBOp::add_bias(float* output, float* biases, int batch,
                                    int n, int size) {
  int i, j, b;
  for (b = 0; b < batch; ++b) {
    for (j = 0; j < size; ++j) {
      for (i = 0; i < n; ++i) {
        output[(b * n + i) * size + j] += biases[i];
      }
    }
  }
}
}  // namespace RVTensor
