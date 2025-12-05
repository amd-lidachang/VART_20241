/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "AIE2Cfg.hpp"
#include "Util.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

inline int up_align(int total, int line) {
  int ref = total / line;
  if (total % line != 0) {
    ref += 1;
  }
  return ref;
}

template <typename T>
T DRound(double data) {
  T rlt = 0;
  T data_max = std::numeric_limits<T>::max();
  T data_min = std::numeric_limits<T>::min();
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if (data < 0 && (data - floor(data)) >= 0.5) {
    rlt = static_cast<T>(ceil(data));
  } else {
    rlt = static_cast<T>(round(data));
  }
  return rlt;
}

template <typename T>
T DFloor(double data) {
  T rlt = 0;
  T data_max = std::numeric_limits<T>::max();
  T data_min = std::numeric_limits<T>::min();
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if (data < 0) {
    rlt = static_cast<T>(floor(data));
  } else {
    rlt = static_cast<T>(floor(data));
  }
  return rlt;
}

double dr(double x);

double activation(double tmp, const AIE2Cfg& A);

template <typename Dtype>
void macc_op(const AIE2Cfg A, const Dtype* img, const Dtype* weights,
             const Dtype* bias, Dtype* rlt) {
  // const int dst_h = A.dst_h;
  const int dst_w = A.dst_w;
  const int pad_left = A.pad_left;
  const int pad_right = A.pad_right;
  const int pad_top = A.pad_top;
  const int pad_bottom = A.pad_bottom;
  const int ic = A.ic;
  const int oc = A.oc;
  const int src_h = A.src_h;
  const int src_w = A.src_w;
  const int kernel_h = A.kernel_h;
  const int kernel_w = A.kernel_w;
  const int stride_h = A.stride_h;
  const int stride_w = A.stride_w;
  // const int SHIFT_CUT = A.SHIFT_CUT;
  const int SHIFT_BIAS = A.SHIFT_BIAS;
  // const int HSIGMOID_IN = A.HSIGMOID_IN;
  // const int HSIGMOID_OUT = A.HSIGMOID_OUT;
  // const int HSWISH_OUT = A.HSWISH_OUT;
  // const int ACT_TYPE = A.ACT_TYPE;
  int bias_factor = SHIFT_BIAS;

  for (int idx_oc = 0; idx_oc < oc; idx_oc++) {
    int idx_oh = 0;
    for (int idx_src_h = -pad_top; idx_src_h + kernel_h <= src_h + pad_bottom;
         idx_src_h += stride_h) {
      int idx_ow = 0;
      for (int idx_src_w = -pad_left; idx_src_w + kernel_w <= src_w + pad_right;
           idx_src_w += stride_w) {
        int64_t result = 0;

        for (int i = 0; i < kernel_h; i++) {
          for (int j = 0; j < kernel_w; j++) {
            int idh = idx_src_h + i;
            int idw = idx_src_w + j;

            if (idh >= 0 && idh < src_h && idw >= 0 && idw < src_w) {
              int img_addr = idh * src_w * ic + idw * ic + idx_oc;
              int wgt_addr = (i * kernel_w + j) * ic + idx_oc;
              result += img[img_addr] * weights[wgt_addr];
            }
          }
        }
        int rlt_addr = idx_oh * dst_w * oc + idx_ow * oc + idx_oc;
        result *= 2;
        double tmp = result;
        tmp += std::floor(double(bias[idx_oc]) * pow(2, bias_factor));

        tmp = activation(tmp, A);
        rlt[rlt_addr] = DRound<Dtype>(tmp);

        idx_ow++;
      }
      idx_oh++;
    }
  }
}

template <typename Dtype>
void avgpool_op(const AIE2Cfg A, const Dtype* img, const Dtype* weights,
                const Dtype* bias, Dtype* rlt) {
  // const int dst_h = A.dst_h;
  const int dst_w = A.dst_w;
  const int pad_left = A.pad_left;
  const int pad_right = A.pad_right;
  const int pad_top = A.pad_top;
  const int pad_bottom = A.pad_bottom;
  const int ic = A.ic;
  // const int icp = A.ICp;
  const int oc = A.oc;
  const int ocp = A.OCp;
  const int src_h = A.src_h;
  const int src_w = A.src_w;
  const int kernel_h = A.kernel_h;
  const int kernel_w = A.kernel_w;
  const int stride_h = A.stride_h;
  const int stride_w = A.stride_w;
  // const int SHIFT_CUT = A.SHIFT_CUT;
  // const int SHIFT_BIAS = A.SHIFT_BIAS;
  // const int HSIGMOID_IN = A.HSIGMOID_IN;
  // const int HSIGMOID_OUT = A.HSIGMOID_OUT;
  // const int HSWISH_OUT = A.HSWISH_OUT;
  // const int ACT_TYPE = A.ACT_TYPE;
  // int bias_factor = SHIFT_BIAS;

  for (int idx_oc = 0; idx_oc < oc; idx_oc++) {
    int idx_oh = 0;
    for (int idx_src_h = -pad_top; idx_src_h + kernel_h <= src_h + pad_bottom;
         idx_src_h += stride_h) {
      int idx_ow = 0;
      for (int idx_src_w = -pad_left; idx_src_w + kernel_w <= src_w + pad_right;
           idx_src_w += stride_w) {
        int64_t result = 0;

        for (int i = 0; i < kernel_h; i++) {
          for (int j = 0; j < kernel_w; j++) {
            int idh = idx_src_h + i;
            int idw = idx_src_w + j;

            if (idh >= 0 && idh < src_h && idw >= 0 && idw < src_w) {
              int img_addr = idh * src_w * ic + idw * ic + idx_oc;
              int wgt_addr = idx_oc % ocp;
              result += img[img_addr] * weights[wgt_addr];
            }
          }
        }
        int rlt_addr = idx_oh * dst_w * oc + idx_ow * oc + idx_oc;
        result *= 2;
        double tmp = result;
        // tmp += std::floor(double(bias[idx_oc%ocp]) * pow(2, bias_factor));
        // bias is 0
        tmp += 0;
        tmp = activation(tmp, A);
        rlt[rlt_addr] = DRound<Dtype>(tmp);

        idx_ow++;
      }
      idx_oh++;
    }
  }
}

template <typename Dtype>
void maxpool_op(const AIE2Cfg A, const Dtype* img, Dtype* rlt) {
  // const int dst_h = A.dst_h;
  const int dst_w = A.dst_w;
  const int pad_left = A.pad_left;
  const int pad_right = A.pad_right;
  const int pad_top = A.pad_top;
  const int pad_bottom = A.pad_bottom;
  const int ic = A.ic;
  // const int icp = A.ICp;
  const int oc = A.oc;
  // const int ocp = A.OCp;
  const int src_h = A.src_h;
  const int src_w = A.src_w;
  const int kernel_h = A.kernel_h;
  const int kernel_w = A.kernel_w;
  const int stride_h = A.stride_h;
  const int stride_w = A.stride_w;
  // const int SHIFT_CUT = A.SHIFT_CUT;
  // const int SHIFT_BIAS = A.SHIFT_BIAS;
  // const int HSIGMOID_IN = A.HSIGMOID_IN;
  // const int HSIGMOID_OUT = A.HSIGMOID_OUT;
  // const int HSWISH_OUT = A.HSWISH_OUT;
  // const int ACT_TYPE = A.ACT_TYPE;
  // int bias_factor = SHIFT_BIAS;

  for (int idx_oc = 0; idx_oc < oc; idx_oc++) {
    int idx_oh = 0;
    for (int idx_src_h = -pad_top; idx_src_h + kernel_h <= src_h + pad_bottom;
         idx_src_h += stride_h) {
      int idx_ow = 0;
      for (int idx_src_w = -pad_left; idx_src_w + kernel_w <= src_w + pad_right;
           idx_src_w += stride_w) {
        int64_t result = -128;

        for (int i = 0; i < kernel_h; i++) {
          for (int j = 0; j < kernel_w; j++) {
            int idh = idx_src_h + i;
            int idw = idx_src_w + j;

            if (idh >= 0 && idh < src_h && idw >= 0 && idw < src_w) {
              int img_addr = idh * src_w * ic + idw * ic + idx_oc;
              // int wgt_addr = (i * kernel_w + j) * ic + idx_oc;
              int64_t pixel = img[img_addr];
              result = max(result, pixel);
            }
          }
        }
        int rlt_addr = idx_oh * dst_w * oc + idx_ow * oc + idx_oc;

        rlt[rlt_addr] = DRound<Dtype>(result);
        idx_ow++;
      }
      idx_oh++;
    }
  }
}

template <typename Dtype>
void elew_shift_op(const AIE2Cfg A, const Dtype* img, Dtype* rlt) {
  const int SHIFT_CUT = A.SHIFT_CUT;
  const int SHIFT_BIAS = A.SHIFT_BIAS;
  const int act_type = A.ACT_TYPE;
  const int src_h = A.src_h;
  const int src_w = A.src_w;
  const int ic = A.ic;
  int len = ic * src_h * src_w;

  int shift_left_bits = 0;
  int shift_right_bits = 0;
  int shift_elew_a = SHIFT_BIAS & 0xffff;
  // int shift_elew_b = SHIFT_BIAS >> 16;
  if (SHIFT_CUT > 0) {
    shift_left_bits = SHIFT_CUT & 0x7f;
    shift_right_bits = 0;
  } else {
    shift_right_bits = SHIFT_CUT & 0x7f;
    shift_left_bits = 0;
  }
  shift_right_bits += 2;
  int shift_a = pow(2, shift_elew_a);
  // int shift_b = pow(2, shift_elew_b);
  int shift_l = pow(2, shift_left_bits);
  int shift_r = pow(2, shift_right_bits);

  if (act_type == 1) {  // RELU
    for (int idx = 0; idx < len; idx++) {
      double res = 0;
      int a = img[idx];
      a = a << 2;
      res = a;
      res = res / shift_a;
      a = DFloor<int16_t>(res);
      a = max(a, 0);
      res = a;
      res = res * shift_l / shift_r;
      rlt[idx] = DRound<Dtype>(res);
    }
  } else if (act_type == 5) {  // H-Sigmoid

    const int HSIGMOID_IN = A.HSIGMOID_IN;
    const int HSIGMOID_OUT = A.HSIGMOID_OUT;
    printf("=================== \n");
    printf("shift_cut:%d, sig_in:%d, sig_out:%d\n", A.SHIFT_CUT, A.HSIGMOID_IN,
           A.HSIGMOID_OUT);
    for (int idx = 0; idx < len; idx++) {
      double res = 0;
      int a = img[idx];
      // -----------no elew shift-------
      res = a;
      res = std::min(
                pow(2, 32),
                std::max(0.0, (res * 2731 + 3 * 2731 * pow(2, HSIGMOID_IN)))) *
            pow(2, HSIGMOID_OUT - HSIGMOID_IN - 14);
      // -------- with elew shift
      /*
      a = a << 2;
      res = a;
      res = res / shift_a;
      a = DRound<int16_t>(res);
      res = a;
      res = res * shift_l / shift_r;
      res = DRound<int16_t>(res);
      res = a;
      res = dr(res);
      res = std::min(pow(2,32),
          std::max(0.0, (res * 2731 + 3*2731 * pow(2,HSIGMOID_IN)))) *
      pow(2,HSIGMOID_OUT - HSIGMOID_IN -14);
      */
      rlt[idx] = DRound<Dtype>(res);
    }
  } else {
    printf("====== elew shift =============\n");
    for (int idx = 0; idx < len; idx++) {
      double res = 0;
      int a = img[idx];
      a = a << 2;
      res = a;
      res = res / shift_a;
      a = DFloor<int16_t>(res);
      res = a;
      res = res * shift_l / shift_r;
      rlt[idx] = DRound<Dtype>(res);
    }
  }
}

template <typename Dtype>
void elew_add_op(const AIE2Cfg A, const Dtype* img, Dtype* rlt) {
  // use 2 imgs to add
  // which means in DDR 2 imgs are arranged img1:ic->iw->ih-> img2

  const int SHIFT_CUT = A.SHIFT_CUT;
  const int SHIFT_BIAS = A.SHIFT_BIAS;
  const int act_type = A.ACT_TYPE;
  const int src_h = A.src_h;
  const int src_w = A.src_w;
  const int ic = A.ic;

  int len = ic * src_h * src_w;

  int shift_elew_a = SHIFT_BIAS & 0xffff;
  int shift_elew_b = SHIFT_BIAS >> 16;
  int shift_a = pow(2, shift_elew_a);
  int shift_b = pow(2, shift_elew_b);

  int shift_left_bits = 0;
  int shift_right_bits = 0;
  // if(SHIFT_CUT  & 0x80000000 == 0x80000000){
  if (SHIFT_CUT > 0) {
    shift_left_bits = SHIFT_CUT & 0x7f;
    shift_right_bits = 0;
    printf("========= shift cuut < 0 \n");
  } else {
    printf("========= shift cuut > 0 \n");
    shift_right_bits = SHIFT_CUT & 0x7f;
    shift_left_bits = 0;
  }
  shift_right_bits += 2;

  int shift_l = pow(2, shift_left_bits);
  int shift_r = pow(2, shift_right_bits);

  for (int idx = 0; idx < len; idx++) {
    int a = img[idx];
    int b = img[idx + len];
    double res_a = a << 2;
    double res_b = b << 2;
    res_a = res_a / shift_a;
    res_b = res_b / shift_b;
    a = DFloor<int16_t>(res_a);
    b = DFloor<int16_t>(res_b);
    res_a = a * shift_l;
    res_b = b * shift_l;
    double res = res_a + res_b;
    res = res / shift_r;
    rlt[idx] = DRound<Dtype>(res);
    if (act_type == 1) {  // RELU
      rlt[idx] = rlt[idx] < 0 ? 0 : rlt[idx];
    }
  }
}

template <typename Dtype>
void elew_mul_op(const AIE2Cfg A, const Dtype* img, Dtype* rlt) {
  const int SHIFT_CUT = A.SHIFT_CUT;
  const int SHIFT_BIAS = A.SHIFT_BIAS;
  const int act_type = A.ACT_TYPE;
  const int src_h = A.src_h;
  const int src_w = A.src_w;
  const int ic = A.ic;

  int len = ic * src_h * src_w;

  int shift_elew_a = SHIFT_BIAS & 0xffff;
  int shift_elew_b = SHIFT_BIAS >> 16;
  int shift_a = pow(2, shift_elew_a);
  int shift_b = pow(2, shift_elew_b);

  int shift_left_bits = 0;
  int shift_right_bits = 0;
  // if(SHIFT_CUT  & 0x80000000 == 0x80000000){
  if (SHIFT_CUT > 0) {
    shift_left_bits = SHIFT_CUT & 0x7f;
    shift_right_bits = 0;
  } else {
    shift_right_bits = SHIFT_CUT & 0x7f;
    shift_left_bits = 0;
  }

  shift_right_bits += 4;

  int shift_l = pow(2, shift_left_bits);
  int shift_r = pow(2, shift_right_bits);

  for (int idx = 0; idx < len; idx++) {
    int a = img[idx];
    int b = img[idx + len];
    double res_a = a << 2;
    double res_b = b << 2;
    res_a = res_a / shift_a;
    res_b = res_b / shift_b;
    a = DFloor<int16_t>(res_a);
    b = DFloor<int16_t>(res_b);
    double res = a * b;
    res = res * shift_l / shift_r;
    rlt[idx] = DRound<Dtype>(res);

    if (act_type == 1) {  // RELU
      rlt[idx] = rlt[idx] < 0 ? 0 : rlt[idx];
    }
  }
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
