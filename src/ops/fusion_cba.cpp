/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/fusion_cba.hpp"

namespace RVTensor {

CPUFusionCBAOp::sptr CPUFusionCBAOp::create() {
  return std::make_shared<CPUFusionCBAOp>();
}

CPUFusionCBAOp::sptr CPUFusionCBAOp::create(ConvParam conv_param,
                             BatchNormParam bn_param,
                             ActiveType active_type,
                             RamTensor::sptr input,
                             RamTensor::sptr output,
                             FlashTensor::sptr weight,
                             FlashTensor::sptr bias) {
  CPUFusionCBAOp::sptr ptr = std::make_shared<CPUFusionCBAOp>(conv_param,
                                                    bn_param, active_type,
                                                    input, output,
                                                    weight, bias);
  // ptr->checkOutputDims();
  return ptr;
}

inline CPUFusionCBAOp::CPUFusionCBAOp() : Operation({}, {}),
                                conv_param_({0, 0, 1, 1, 0, 0, false}),
                                bn_param_({{}, {}, 0.001}),
                                active_type_(ACTIVE_SIGMOID),
                                weight_(nullptr), bias_(nullptr) {}

inline CPUFusionCBAOp::CPUFusionCBAOp(ConvParam conv_param,
                            BatchNormParam bn_param,
                            ActiveType active_type,
                            RamTensor::sptr input,
                            RamTensor::sptr output,
                            FlashTensor::sptr weight,
                            FlashTensor::sptr bias)
                          : Operation({input}, {output}),
                            conv_param_(conv_param),
                            bn_param_(bn_param),
                            active_type_(active_type),
                            weight_(weight), bias_(bias) {}

inline CPUFusionCBAOp::~CPUFusionCBAOp() {}

inline void CPUFusionCBAOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  uint8_t* input = reinterpret_cast<uint8_t *>(input_tensor->data_ptr);
  uint8_t* output = reinterpret_cast<uint8_t *>(output_tensor->data_ptr);
  uint8_t* weight = reinterpret_cast<uint8_t *>(weight_->data_ptr);
  uint8_t* bias = bias_ ?
                    reinterpret_cast<uint8_t *>(bias_->data_ptr) : nullptr;

  int ni = input_tensor->n_batch;
  int ci = input_tensor->channel;
  int hi = input_tensor->height;
  int wi = input_tensor->width;
  int stepi = input_tensor->cstep;
  int co = output_tensor->channel;
  int ho = output_tensor->height;
  int wo = output_tensor->width;
  int stepo = output_tensor->cstep;
  int sh = conv_param_.sh;
  int sw = conv_param_.sw;
  int kh = weight_->height;
  int kw = weight_->width;
  int dh = conv_param_.dh;
  int dw = conv_param_.dw;
  int ph = conv_param_.ph;
  int pw = conv_param_.pw;

  uint8_t* temp_weight = nullptr;
  int x = 0, y = 0;
  if (dh > 1 || dw > 1) {
    kh = (kh - 1) * dh + 1;
    kw = (kw - 1) * dw + 1;
    temp_weight = reinterpret_cast<uint8_t *>(
                   malloc(sizeof(uint8_t) * kw * kh * ci * co));
    x = -1;
    y = -1;
    for (int coi = 0; coi < co; coi++) {
      for (int cii = 0; cii < ci; cii++) {
        for (int khi = 0; khi < kh; khi++) {
          for (int kwi = 0; kwi < kw; kwi++) {
            x++;
            if (khi % dh != 0 || kwi % dw != 0) {
              temp_weight[x] = 0;
            } else {
              y++;
              temp_weight[x] = weight[y];
            }
          }
        }
      }
    }
  } else {
    temp_weight = weight;
  }

  for (int n = 0; n < ni; n++) {
    for (int coo = 0; coo < co; coo++) {
      for (int hoo = 0; hoo < ho; hoo++) {
        for (int woo = 0; woo < wo; woo++) {
          int start_w = sw * woo - pw / 2;
          int start_h = sh * hoo - ph / 2;
          int end_w = (std::min)(start_w + kw, wi);
          int end_h = (std::min)(start_h + kh, hi);
          int kernel_shift_w = (start_w < 0) ? -start_w : 0;
          int kernel_shift_h = (start_h < 0) ? -start_h : 0;
          int rem_dw = kernel_shift_w % dw;
          int rem_dh = kernel_shift_h % dh;
          int kernel_shift_dw = (rem_dw > 0) ? dw - rem_dw : 0;
          int kernel_shift_dh = (rem_dh > 0) ? dh - rem_dh : 0;
          start_w = (std::max)(start_w, kernel_shift_dw);
          start_h = (std::max)(start_h, kernel_shift_dh);
          output[n * co * stepo + coo * stepo + hoo * wo + woo] = 0;
          for (int cii = 0; cii < ci; cii++) {
            for (int h = start_h; h < end_h; h += dh) {
              for (int w = start_w; w < end_w; w += dw) {
                output[n * co * stepo + coo * stepo + hoo * wo + woo] +=
                  input[n * ci *  stepi + cii *  stepi + h * wi + w] *
                  temp_weight[coo * ci * kh * kw + cii * kh * kw +
                  (kernel_shift_h + kernel_shift_dh + h - start_h) * kw +
                  (kernel_shift_w + kernel_shift_dw + w - start_w)];
              }
            }
          }
          if (bias != nullptr) {
            output[n * co * stepo + coo * stepo + hoo * wo + woo] += bias[coo];
          }
        }
      }
    }
  }
  if (dh > 1 || dw > 1) {
    free(temp_weight);
  }

  // TODO: complete it

}

}  // namespace RVTensor
