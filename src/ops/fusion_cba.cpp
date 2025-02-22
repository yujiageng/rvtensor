/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/fusion_cba.hpp"
#include "math.h"

namespace RVTensor {

CPUFusionCBAOp::sptr CPUFusionCBAOp::create() {
  return std::make_shared<CPUFusionCBAOp>();
}

CPUFusionCBAOp::sptr CPUFusionCBAOp::create(
    ConvParam conv_param, BnModelData bn_param, ActiveType active_type,
    RamTensor::sptr input, RamTensor::sptr output, FlashTensor::sptr weight,
    FlashTensor::sptr bias) {
  CPUFusionCBAOp::sptr ptr = std::make_shared<CPUFusionCBAOp>(
      conv_param, bn_param, active_type, input, output, weight, bias);
  ptr->checkOutputDims();
  return ptr;
}

inline CPUFusionCBAOp::CPUFusionCBAOp()
    : Operation({}, {}),
      conv_param_({0, 0, 1, 1, 0, 0, false}),
      bn_param_({nullptr, nullptr, nullptr, nullptr}),
      active_type_(ACTIVE_SIGMOID),
      weight_(nullptr),
      bias_(nullptr) {}

inline CPUFusionCBAOp::CPUFusionCBAOp(
    ConvParam conv_param, BnModelData bn_param, ActiveType active_type,
    RamTensor::sptr input, RamTensor::sptr output, FlashTensor::sptr weight,
    FlashTensor::sptr bias)
    : Operation({input}, {output}),
      conv_param_(conv_param),
      bn_param_(bn_param),
      active_type_(active_type),
      weight_(weight),
      bias_(bias) {}

inline CPUFusionCBAOp::~CPUFusionCBAOp() {}

void CPUFusionCBAOp::checkOutputDims() {
  auto input = getInputs()[0];
  auto output = getOutputs()[0];
  if (input->channel != weight_->channel) {
    throw std::runtime_error("CPUFusionCBAOp channel of input is wrong!");
  }

  int input_h = input->height + conv_param_.ph;
  int input_w = input->width + conv_param_.pw;
  int kh =
      conv_param_.dh > 1 ? (weight_->height - 1) * conv_param_.dh + 1 : weight_->height;
  int kw =
      conv_param_.dw > 1 ? (weight_->width - 1) * conv_param_.dw + 1 : weight_->width;
  int output_h = (input_h - kh) / conv_param_.sh + 1;
  int output_w = (input_w - kw) / conv_param_.sw + 1;
  int output_c = weight_->n_batch;
  int output_n = input->n_batch;
  if (output->n_batch != output_n || output->channel != output_c ||
      output->height != output_h || output->width != output_w) {
    printf("infer: n(%d) c(%d) h(%d) w(%d)\n", output_n, output_c, output_h, output_w);
    printf("actual: n(%d) c(%d) h(%d) w(%d)\n", output->n_batch, output->channel, output->height, output->width);
    throw std::runtime_error("CPUFusionCBAOp output shape is wrong!");
  }

  if (input_h < kh) {
    throw std::runtime_error("CPUFusionCBAOp kernel_h is wrong");
  }

  if (input_w < kw) {
    throw std::runtime_error("CPUFusionCBAOp kernel_w is wrong!");
  }
}

void CPUFusionCBAOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  float* output = reinterpret_cast<float*>(output_tensor->data_ptr);
  float* weight = reinterpret_cast<float*>(weight_->data_ptr);
  float* bias = bias_ ? reinterpret_cast<float*>(bias_->data_ptr) : nullptr;

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
  float* gamma = (float*)(bn_param_.bn_gamma_ptr->data_ptr);
  float* beta = (float*)(bn_param_.bn_beta_ptr->data_ptr);

#ifdef CONV
  // float* temp_weight = nullptr;
  // int x = 0, y = 0;
  // if (dh > 1 || dw > 1) {
  //   kh = (kh - 1) * dh + 1;
  //   kw = (kw - 1) * dw + 1;
  //   temp_weight =
  //       reinterpret_cast<float*>(malloc(sizeof(float) * kw * kh * ci * co));
  //   x = -1;
  //   y = -1;
  //   // padding
  //   for (int coi = 0; coi < co; coi++) { // 输出channel
  //     for (int cii = 0; cii < ci; cii++) { // 输入channel
  //       for (int khi = 0; khi < kh; khi++) {
  //         for (int kwi = 0; kwi < kw; kwi++) {
  //           x++;
  //           if (khi % dh != 0 || kwi % dw != 0) {
  //             temp_weight[x] = 0;
  //           } else {
  //             y++;
  //             temp_weight[x] = weight[y];
  //           }
  //         }
  //       }
  //     }
  //   }
  // } else {
  //   temp_weight = weight;
  // }
  assert((dh == 1) && (dw == 1));
  void* input = input_tensor->data_ptr;
  float* temp_weight = weight;

  // 卷积
  for (int n = 0; n < ni; n++) {
    for (int coo = 0; coo < co; coo++) {
      for (int hoo = 0; hoo < ho; hoo++) {
        for (int woo = 0; woo < wo; woo++) {
          // 卷积开始和结束的index
          int start_w = sw * woo - pw / 2;
          int start_h = sh * hoo - ph / 2;
          int end_w = (std::min)(start_w + kw, wi);
          int end_h = (std::min)(start_h + kh, hi);
          // kernel滑动的 index
          int kernel_shift_w = (start_w < 0) ? -start_w : 0;
          int kernel_shift_h = (start_h < 0) ? -start_h : 0;
          //
          int rem_dw = kernel_shift_w % dw;
          int rem_dh = kernel_shift_h % dh;
          int kernel_shift_dw = (rem_dw > 0) ? dw - rem_dw : 0;
          int kernel_shift_dh = (rem_dh > 0) ? dh - rem_dh : 0;
          start_w = (std::max)(start_w, kernel_shift_dw);
          start_h = (std::max)(start_h, kernel_shift_dh);
          output[n * co * ho * wo + coo * ho * wo + hoo * wo + woo] = 0;
          for (int cii = 0; cii < ci; cii++) {
            for (int h = start_h; h < end_h; h += dh) {
              for (int w = start_w; w < end_w; w += dw) {
                if (input_tensor->element_size == 4u) {
                    output[n * co * ho * wo + coo * ho * wo + hoo * wo + woo] +=
                        ((float*)input)[n * ci * hi * wi + cii * hi * wi + h * wi + w] *
                         temp_weight[coo * ci * kh * kw + cii * kh * kw +
                         (kernel_shift_h + kernel_shift_dh + h - start_h) * kw +
                         (kernel_shift_w + kernel_shift_dw + w - start_w)];
                } else if (input_tensor->element_size == 1u) {
                    output[n * co * ho * wo + coo * ho * wo + hoo * wo + woo] +=
                        ((uint8_t*)input)[n * ci * hi * wi + cii * hi * wi + h * wi + w] * 1./255 *
                         temp_weight[coo * ci * kh * kw + cii * kh * kw +
                         (kernel_shift_h + kernel_shift_dh + h - start_h) * kw +
                         (kernel_shift_w + kernel_shift_dw + w - start_w)];
                }
              }
            }
          }
          if (bias != nullptr) {
            output[n * co * ho * wo + coo * ho * wo + hoo * wo + woo] += bias[coo];
          }
        }
      }
    }
  }
  // if (dh > 1 || dw > 1) {
  //   free(temp_weight);
  // }

#else

  // 卷积核的个数 = 输出的通道数
  int m = output_tensor->channel;
  /*卷积核 元素的个数,l.size=卷积核的尺寸，l.c= 卷积核的通道*/
  int k = kh * kh * ci;
  /*该层输出单通道的特征图的尺寸*/
  int n = ho * wo;

  int height_col = (hi + 2 * ph - kh) / sh + 1;
  int width_col = (wi + 2 * pw - kh) / sw + 1;
  /*循环batch中的每个输入*/

  float* input = NULL;

  if (input_tensor->element_size == 1u) {
    input = (float*)calloc(input_tensor->count(), sizeof(float));
    uint8_t* input_orig = reinterpret_cast<uint8_t*>(input_tensor->data_ptr);
    for (int i = 0; i < input_tensor->count(); i++)
        input[i] = input_orig[i] * 1./255;
  } else {
    input = reinterpret_cast<float*>(input_tensor->data_ptr);
  }

  for (int i = 0; i < ni; ++i) {
  //  printf("i:%d",i);
    /*用于存储经im2col转换后的输入特征矩阵*/
    float* a = (float*)calloc(((k+1)*height_col+1)*width_col, sizeof(float));

    /*a是指向当前层所有卷积核的*/
    float* b = weight;
    /*输出特征图个数*/
    float* c = output + i * n * m;
    float* im = input + i * ci * hi * wi;
    /*如果是1*1的卷积，那么不用对输入特征进行转化*/
    if (kh * kh == 1) {
      a = im;
    } else {
      /*对输入特征进行转化*/
      im2col_cpu(im, ci, hi, wi, kh, sh, ph, a);
    }
    coppersmith_winograd(a, b, c, m, n, k, k, n, n);
    free(a);
  }
  if (input_tensor->element_size == 1u) {
      free(input);
  }
#endif

  /*BN input
  conv_layer output = bn_layer input
  bn_layer output 放入 output_tensor，并将output_tensor
  作为activate_layer的输入， activate_layer的输出放入output_tensor中
  */

  // 归一化
  normalize_cpu(output, mean, variance, gamma, beta, ni, co, ho * wo);

  /*激活  */
  relu(output, ni * co * ho * wo);
}

void CPUFusionCBAOp::mm_generate(float* matA, float* matB,
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


void CPUFusionCBAOp::coppersmith_winograd(float* matA, float* matB,
                                                float* matC, int M, int N,
                                                int K, int strideA,
                                                int strideB, int strideC) {
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

float CPUFusionCBAOp::im2col_get_pixel(float* im, int height,
                                              int width, int channels, int row,
                                              int col, int channel, int pad) {
  row -= pad;
  col -= pad;

  if (row < 0 || col < 0 || row >= height || col >= width) return 0;
  return im[col + width * (row + height * channel)];
}

void CPUFusionCBAOp::im2col_cpu(float* data_im, int channels,
                                       int height, int width, int ksize,
                                       int stride, int pad, float* data_col) {
  int c, h, w;
  // 计算卷基层输出图像的高 和宽
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  //计算卷积层输入单通道图像的数据容量
  int channels_col = channels * ksize * ksize;
  //循环每个卷积核的参数个数
  // printf("col_index:%d\n", (channels_col * height_col + height_col) * width_col + width_col);
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
//        printf("col_index:%d\n", col_index);
        data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                               im_row, im_col, c_im, pad);
      }
    }
  }
}

void CPUFusionCBAOp::copy_cpu(int N, float* X, int INCX, float* Y,
                                     int INCY) {
  int i;
  for (i = 0; i < N; ++i) Y[i * INCY] = X[i * INCX];
}

//归一化
void CPUFusionCBAOp::normalize_cpu(float* x, float* mean,
                                          float* variance, float*gamma,
                                          float* beta, int batch,
                                          int filters, int spatial) {
  int b, f, i;
  for (b = 0; b < batch; ++b) {
    for (f = 0; f < filters; ++f) {
      float offset = beta[f] - gamma[f] * mean[f] / sqrt(variance[f] + 0.001f);
      float slope = gamma[f] / sqrt(variance[f] + 0.001f);
      for (i = 0; i < spatial; ++i) {
        int index = b * filters * spatial + f * spatial + i;
        x[index] = slope * x[index] + offset;
      }
    }
  }
}
void CPUFusionCBAOp::scale_bias(float* output, float* scales,
                                       int batch, int n, int size) {
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
void CPUFusionCBAOp::add_bias(float* output, float* biases,
                                     int batch, int n, int size) {
  int i, j, b;
  for (b = 0; b < batch; ++b) {
    for (j = 0; j < size; ++j) {
      for (i = 0; i < n; ++i) {
        output[(b * n + i) * size + j] += biases[i];
      }
    }
  }
}
void CPUFusionCBAOp::relu(float* inputs, int n) {
  for (int i = 0; i < n; ++i) {
    if (inputs[i] <= 0) {
      inputs[i] = 0;
    }
  }
}

}  // namespace RVTensor
