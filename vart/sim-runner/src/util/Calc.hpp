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
#ifndef __CALC_HPP__
#define __CALC_HPP__

#include "FMapTypes.hpp"
#include "common.hpp"

using namespace std;

class Calc {
 public:
  Calc() = delete;
  ~Calc() = delete;

 public:
  enum act_type_t {
    RELU_TYPE_MIN = 0,
    RELU_TYPE_NONE = RELU_TYPE_MIN,  // 0
    RELU_TYPE_RELU,                  // 1
    RELU_TYPE_PRELU,                 // 2
    RELU_TYPE_LEAKY_RELU,            // 3
    RELU_TYPE_RELUSIX,               // 4
    RELU_TYPE_HSIGMOID,              // 5
    RELU_TYPE_HSWISH,                // 6
    RELU_TYPE_BN,                    // 7
    RELU_TYPE_BN_RELU,               // 8
    RELU_TYPE_MAX,                   // 9
  };
  enum pool_type_t {
    POOL_TYPE_MIN = 0,
    POOL_TYPE_MAX_POOL = POOL_TYPE_MIN,  // 0
    POOL_TYPE_AVG_POOL,                  // 1
    POOL_TYPE_MAX_REDUCE,                // 2
    POOL_TYPE_MAX                        // 3
  };
  enum elew_type_t {
    ELEW_TYPE_MIN = 0,
    ELEW_TYPE_ADD = ELEW_TYPE_MIN,  // 0
    ELEW_TYPE_MULT,                 // 1
    ELEW_TYPE_MAX                   // 2
  };

  enum pad_type_t {
    PAD_TYPE_ZERO = 0,
    PAD_TYPE_MIN = 1,
  };

  enum alu_exec_mode_t : int {
    ALU_TYPE_MIN = 0,
    ALU_TYPE_DWCV = ALU_TYPE_MIN,  // 0
    ALU_TYPE_PRELU,                // 1
    ALU_TYPE_AVEPOOL,              // 2
    ALU_TYPE_MAXPOOL,              // 3
    ALU_TYPE_LEAKYRELU,            // 4
    ALU_TYPE_MAXREDUCE,            // 5
    ALU_TYPE_DWCVB0,               // 6
    ALU_TYPE_HSIGMOID,             // 7
    ALU_TYPE_DWCVW16B0,            // 8
    ALU_TYPE_MACC,                 // 9
    ALU_TYPE_COMP,                 // 10
    ALU_TYPE_ELEW_ADD,             // 11
    ALU_TYPE_ELEW_MUL,             // 12
    ALU_TYPE_MAX,                  // 13
  };

  enum conv_exec_mode_t : int {
    CONV_MODE_MIN = 0,
    CONV_MODE_CONV = CONV_MODE_MIN,  // 0
    CONV_MODE_FC,                    // 1
    CONV_MODE_MATMUL,                // 2
    CONV_MODE_MATMUL_T,              // 3
    CONV_MODE_SOFTMAX,               // 4
    CONV_MODE_SIGMOID,               // 5
    CONV_MODE_TAN_H,                 // 6
    CONV_MODE_EADD,                  // 7
    CONV_MODE_EMUL,                  // 8
    CONV_MODE_MACC,                  // 9
    CONV_MODE_LAYERNORM,             // 10
    CONV_MODE_GLU,                   // 11
    CONV_MODE_GELU,                  // 12
    CONV_MODE_CONV_ELEWADD_FUSION,   // 13
  };

  const static array<string, RELU_TYPE_MAX> ReluTypeStr;

 public:
  template <typename T>
  static void Softmax(const T* src, int num, T* rlt);

  // change data access order
 public:
  template <typename T>
  static T DPURound(double data);
  template <typename T>
  static void HWC2CHW(int h, int w, int c, const T* src, T* dst);
  template <typename T0, typename T1>
  static void HWC2CHW(int h, int w, int c, const T0* src, T1* dst);
  template <typename T>
  static void CHW2HWC(int c, int h, int w, const T* src, T* dst);

 public:
  enum TransDimMethod {
    TDM_MIN = 0,
    NSTRIDE_2_NSTRIDE = TDM_MIN,
    NSTRIDE_2_YSTRIDE,
    YSTRIDE_2_NSTRIDE,
    YSTRIDE_2_YSTRIDE,
    TDM_MAX,
  };
  static string GetTDMName(int method);
  template <typename T>
  static void TransDim(T* dst, size_t dst_size, const T* src, size_t src_size,
                       const FMapAttr& fmap, int method);
};

template <typename T>
void Calc::Softmax(const T* src, int num, T* rlt) {
  T sum = 0;

  for (auto i = 0; i < num; i++) {
    rlt[i] = exp(src[i]);
    sum += rlt[i];
  }

  for (auto i = 0; i < num; i++) {
    rlt[i] = rlt[i] / sum;
  }
}

template <typename T>
T Calc::DPURound(double data) {
  T rlt = 0;
  T data_max = std::numeric_limits<T>::max();
  T data_min = std::numeric_limits<T>::min();

  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if (data < 0 && (data - floor(data)) == 0.5) {
    rlt = static_cast<T>(ceil(data));
  } else {
    rlt = static_cast<T>(round(data));
  }

  return rlt;
}

template <typename T>
void Calc::HWC2CHW(int h, int w, int c, const T* src, T* dst) {
  for (int idx_c = 0; idx_c < c; idx_c++) {
    for (int idx_h = 0; idx_h < h; idx_h++) {
      for (int idx_w = 0; idx_w < w; idx_w++) {
        dst[idx_c * h * w + idx_h * w + idx_w] =
            src[idx_h * w * c + idx_w * c + idx_c];
      }
    }
  }
}

template <typename T0, typename T1>
void Calc::HWC2CHW(int h, int w, int c, const T0* src, T1* dst) {
  for (int idx_c = 0; idx_c < c; idx_c++) {
    for (int idx_h = 0; idx_h < h; idx_h++) {
      for (int idx_w = 0; idx_w < w; idx_w++) {
        dst[idx_c * h * w + idx_h * w + idx_w] =
            static_cast<T1>(src[idx_h * w * c + idx_w * c + idx_c]);
      }
    }
  }
}

template <typename T>
void Calc::CHW2HWC(int c, int h, int w, const T* src, T* dst) {
  for (int idx_h = 0; idx_h < h; idx_h++) {
    for (int idx_w = 0; idx_w < w; idx_w++) {
      for (int idx_c = 0; idx_c < c; idx_c++) {
        dst[idx_h * w * c + idx_w * c + idx_c] =
            src[idx_c * h * w + idx_h * w + idx_w];
      }
    }
  }
}

template <typename T>
void Calc::TransDim(T* dst, size_t dst_size, const T* src, size_t src_size,
                    const FMapAttr& fmap, int method) {
  UNI_LOG_CHECK(method >= TDM_MIN && method < TDM_MAX, SIM_OUT_OF_RANGE)
      << ": TransDimMethod value " << method << " error!" << endl;
  UNI_LOG_CHECK(dst != nullptr, SIM_NULL_PTR);
  UNI_LOG_CHECK(src != nullptr, SIM_NULL_PTR);

  if (method == NSTRIDE_2_NSTRIDE) {
    UNI_LOG_CHECK(dst_size >= static_cast<size_t>(fmap.h * fmap.w * fmap.c),
                  SIM_PARAMETER_FAILED)
        << ": dst_size " << dst_size << " error!" << endl;
    UNI_LOG_CHECK(src_size >= static_cast<size_t>(fmap.h * fmap.w * fmap.c),
                  SIM_PARAMETER_FAILED)
        << ": src_size " << src_size << " error!" << endl;
  } else if (method == NSTRIDE_2_YSTRIDE) {
    UNI_LOG_CHECK(dst_size >= static_cast<size_t>(fmap.hs * fmap.ws * fmap.cs),
                  SIM_PARAMETER_FAILED)
        << ": dst_size " << dst_size << " error!" << endl;
    UNI_LOG_CHECK(src_size >= static_cast<size_t>(fmap.h * fmap.w * fmap.c),
                  SIM_PARAMETER_FAILED)
        << ": src_size " << src_size << " error!" << endl;
  } else if (method == YSTRIDE_2_NSTRIDE) {
    UNI_LOG_CHECK(dst_size >= static_cast<size_t>(fmap.h * fmap.w * fmap.c),
                  SIM_PARAMETER_FAILED)
        << ": dst_size " << dst_size << " error!" << endl;
    UNI_LOG_CHECK(src_size >= static_cast<size_t>(fmap.hs * fmap.ws * fmap.cs),
                  SIM_PARAMETER_FAILED)
        << ": src_size " << src_size << " error!" << endl;
  } else if (method == YSTRIDE_2_YSTRIDE) {
    UNI_LOG_CHECK(dst_size >= static_cast<size_t>(fmap.hs * fmap.ws * fmap.cs),
                  SIM_PARAMETER_FAILED)
        << ": dst_size " << dst_size << " error!" << endl;
    UNI_LOG_CHECK(src_size >= static_cast<size_t>(fmap.hs * fmap.ws * fmap.cs),
                  SIM_PARAMETER_FAILED)
        << ": src_size " << src_size << " error!" << endl;
  }

  // normal general assign algorithm
  for (auto i = 0; i < fmap.h; i++) {
    for (auto j = 0; j < fmap.w; j++) {
      for (auto k = 0; k < fmap.c; k++) {
        auto src_pos = 0U;
        auto dst_pos = 0U;
        if (method == NSTRIDE_2_NSTRIDE) {
          src_pos = i * fmap.w * fmap.c + j * fmap.c + k;
          dst_pos = i * fmap.w * fmap.c + j * fmap.c + k;
        } else if (method == NSTRIDE_2_YSTRIDE) {
          src_pos = i * fmap.w * fmap.c + j * fmap.c + k;
          dst_pos = i * fmap.ws * fmap.cs + j * fmap.cs + k;
        } else if (method == YSTRIDE_2_NSTRIDE) {
          src_pos = i * fmap.ws * fmap.cs + j * fmap.cs + k;
          dst_pos = i * fmap.w * fmap.c + j * fmap.c + k;
        } else if (method == YSTRIDE_2_YSTRIDE) {
          src_pos = i * fmap.ws * fmap.cs + j * fmap.cs + k;
          dst_pos = i * fmap.ws * fmap.cs + j * fmap.cs + k;
        }

        dst[dst_pos] = src[src_pos];
      }
    }
  }
}

template <typename T1, typename T2, typename Enable = void>
struct IntegralTools;

template <typename T1, typename T2>
struct IntegralTools<T1, T2,
                     std::enable_if_t<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value>> {
  using return_type = typename std::common_type<T1, T2>::type;
  static return_type div_floor(T1 a, T2 b) {
    UNI_LOG_CHECK(b > 0, XCOM_DATA_OUTRANGE);
    auto a_ = static_cast<return_type>(a);
    auto b_ = static_cast<return_type>(b);
    return (a_ - (a_ % b_)) / b_;
  }
  static return_type div_ceil(T1 a, T2 b) {
    UNI_LOG_CHECK(b > 0, XCOM_DATA_OUTRANGE);
    auto a_ = static_cast<return_type>(a);
    auto b_ = static_cast<return_type>(b);
    return a_ % b_ == 0 ? (a_ - (a_ % b_)) / b_ : (a_ - (a_ % b_)) / b_ + 1;
  }
  static return_type align_lower(T1 a, T2 b) {
    UNI_LOG_CHECK(b > 0, XCOM_DATA_OUTRANGE);
    auto a_ = static_cast<return_type>(a);
    auto b_ = static_cast<return_type>(b);
    return (a_ - (a_ % b_));
  }
  static return_type align_upper(T1 a, T2 b) {
    UNI_LOG_CHECK(b > 0, XCOM_DATA_OUTRANGE);
    auto a_ = static_cast<return_type>(a);
    auto b_ = static_cast<return_type>(b);
    return a_ % b_ == 0 ? a_ : a_ - (a_ % b_) + b_;
  }
};

template <typename T1, typename T2>
typename IntegralTools<T1, T2>::return_type div_floor(T1 a, T2 b) {
  return IntegralTools<T1, T2>::div_floor(a, b);
}
template <typename T1, typename T2>
typename IntegralTools<T1, T2>::return_type div_ceil(T1 a, T2 b) {
  return IntegralTools<T1, T2>::div_ceil(a, b);
}
template <typename T1, typename T2>
typename IntegralTools<T1, T2>::return_type align_lower(T1 a, T2 b) {
  return IntegralTools<T1, T2>::align_lower(a, b);
}
template <typename T1, typename T2>
typename IntegralTools<T1, T2>::return_type align_upper(T1 a, T2 b) {
  return IntegralTools<T1, T2>::align_upper(a, b);
}

#endif /* __CALC_HPP__ */
