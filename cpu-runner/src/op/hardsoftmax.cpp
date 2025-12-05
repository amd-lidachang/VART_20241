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

#include "hardsoftmax.hpp"

namespace vart {
namespace cpu {

template <typename DType>
const vector<string> HardSoftmax<DType>::ITName = {
    "input",
};

// constructor and deconstructor
template <typename DType>
HardSoftmax<DType>::HardSoftmax(const xir::Subgraph* subg, const xir::Op* op,
                                IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 6);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 6);

  // get pad info
  axis_ = xir_op_->get_attr<int>("axis");
  axis_ = axis_ == -1
              ? (xir_op_->get_input_tensor("input")->get_shape().size() - 1)
              : axis_;
  input_fix_point_ =
      xir_op_->get_input_tensor("input")->get_attr<int>("fix_point");
  is_bf16_in_ = (xir_op_->get_input_tensor("input")->get_data_type().type ==
                 xir::DataType::BFLOAT);
  is_bf16_out_ = (xir_op_->get_output_tensor()->get_data_type().type ==
                  xir::DataType::BFLOAT);
  if (is_bf16_out_ == false) {
    output_fix_point_ =
        xir_op_->get_output_tensor()->get_attr<int>("fix_point");
  }
}

template <typename DType>
void HardSoftmax<DType>::run() {
  // // do necessary check
  print_param();
  check_param();

  // // read data
  read();

  // do softmax
  softmax();

  // // do save, debug...
  // save();
}

template <typename DType>
void HardSoftmax<DType>::print_param() {
  UNI_LOG_DEBUG_INFO << "fmap_i_ = " << Vec2Str(fmap_i_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "fmap_o_ = " << Vec2Str(fmap_o_, ", ") << endl;

  UNI_LOG_DEBUG_INFO << "axis = " << axis_ << endl;
  UNI_LOG_DEBUG_INFO << "input_fix_point_ = " << input_fix_point_ << endl;
}

template <typename DType>
void HardSoftmax<DType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() == 1, VART_SIZE_ERROR);

  // check fmap_out height
  UNI_LOG_CHECK(fmap_i_.size() == fmap_o_.size(), VART_SIZE_ERROR)
      << fmap_i_.size() << " != " << fmap_o_.size();
  for (auto i = 0U; i < fmap_i_.size(); i++) {
    UNI_LOG_CHECK(fmap_i_[i] == fmap_o_[i], VART_INVALID_VALUE)
        << ", " << fmap_i_[i] << " != " << fmap_o_[i];
  }
}

// read data from DDR
template <typename DType>
void HardSoftmax<DType>::read() {
  // read img data
  auto* cputb = inputs_.at(ITName[INPUT]).at(0);
  img_ = GET_CPUTB_DType_PTR(DType, cputb);

  // handle output buffer
  rlt_ = GET_CPUTB_DType_PTR(DType, output_);
  // rlt_ = GET_CPUTB_FLOAT_PTR(output_);
}

template <typename DType>
uint64_t HardSoftmax<DType>::get_workload() {
  // TODO
  return 0;
}

typedef union value_convert {
  std::uint32_t u;
  float f;
} value_convert_t;

std::uint32_t f_to_u(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

float u_to_f(std::uint32_t data) {
  value_convert_t vc{};
  vc.u = data;
  return vc.f;
}

float f_to_bf(float data) {
  std::uint32_t u = f_to_u(data);
  std::uint32_t flag = (u & 0x00010000) >> 16;
  u = (u + 0x7fff + flag) & 0xFFFF0000;
  return u_to_f(u);
}

float mult_add(float a_bf16, float b_bf16, float c_fp32) {
  float out = f_to_bf(a_bf16) * f_to_bf(b_bf16) + f_to_bf(c_fp32);
  return f_to_bf(out);
}

float ploy_HORNER_SCHEME(float r, std::vector<float> cs, int degree) {
  float out = mult_add(r, cs[degree], cs[degree - 1]);
  auto i = degree - 1;
  while (i > 0) {
    out = mult_add(r, out, cs[i - 1]);
    i--;
  }
  return f_to_bf(out);
}

float exp_approx_poly(float data) {
  // floor 1 + x * (1 + x * (0.486328125 + x * 0.21875))
  std::vector<float> cs = {1, 1, 0.486328125, 0.21875};
  float LN2 = f_to_bf(0.6931471805599453);
  std::uint32_t EXP_TABLE_SIZE = 1;
  std::uint32_t degree = 3;
  float _MAX_BFLOAT16 = u_to_f(0x7f7f0000);
  float _MIN_BFLOAT16 = u_to_f(0xff7f0000);
  float x = f_to_bf(data);
  float scaling_factor = f_to_bf(x) * f_to_bf(EXP_TABLE_SIZE / LN2);
  float n = f_to_bf(floor(f_to_bf(scaling_factor)));
  float m = f_to_bf(floor(n / EXP_TABLE_SIZE));
  float t = f_to_bf(n * f_to_bf(LN2 / EXP_TABLE_SIZE));
  float r = f_to_bf(x - t);
  float exp_x = ploy_HORNER_SCHEME(r, cs, degree);
  exp_x = pow(2, m) * exp_x;
  return f_to_bf(std::max(std::min(exp_x, _MAX_BFLOAT16), _MIN_BFLOAT16));
}

float reciprocal_approx_moroz(float z, bool output_subnormals = false) {
  std::uint32_t magic_n = 0x7eb53567;
  float k1 = f_to_bf(1.9395974);
  float k2 = f_to_bf(1.436142);
  float ones = f_to_bf(1.0f);
  float x = f_to_bf(z);
  float y = u_to_f(magic_n - f_to_u(x));
  float acc = f_to_bf(-x * y + k2);
  acc = f_to_bf(acc * y);
  acc = f_to_bf(acc * k1);
  y = acc;
  acc = f_to_bf(acc * -x + ones);
  acc = f_to_bf(acc * y + y);
  y = acc;
  return (y);
}

float parallel_sum(std::vector<float> data) {
  std::vector<float> psum;
  float sum = 0;
  for (std::uint32_t i = 0; i < 16; i++) {
    psum.push_back(0);
  }
  for (std::uint32_t i = 0; i < data.size(); i++) {
    psum[i % 16] += data[i];
  }
  for (std::uint32_t i = 0; i < 16; i++) {
    sum += f_to_bf(psum[i]);
  }
  return f_to_bf(sum);
}

template <typename DType>
void poly_softmax(DType* src, int num, DType* rlt, int fix_point,
                  bool is_bf16_in) {
  // R sum = 0;
  std::vector<float> exp_res;
  std::vector<float> softmax_res;
  float sum = 0;
  float input_fix_point_scale = std::pow(2, -fix_point);
  for (auto i = 0; i < num; i++) {
    float input_data = is_bf16_in
                           ? u_to_f(static_cast<std::uint32_t>(src[i]) << 16)
                           : ((float)((char)src[i]) * input_fix_point_scale);
    float exp_x = exp_approx_poly(input_data);
    exp_res.push_back(exp_x);
  }
  sum = parallel_sum(exp_res);
  float div = reciprocal_approx_moroz(sum);
  for (auto i : exp_res) {
    softmax_res.push_back(f_to_bf(i * div));
  }
  for (auto i = 0; i < num; i++) {
    rlt[i] = (f_to_u(softmax_res[i])) >> 16;
    if (rlt[i] < 128 && rlt[i] >= 0) rlt[i] = 0;
  }
}

template <typename DType>
void HardSoftmax<DType>::bf16_to_int8(int num, DType* rlt) {
  float output_fix_point_scale = std::pow(2, output_fix_point_);
  for (auto i = 0; i < num; i++) {
    if (is_bf16_out_ == false) {
      float tmp = (float)(u_to_f((*((uint32_t*)(&rlt[i]))) << 16));
      rlt[i] = DPURound<DType>(tmp * output_fix_point_scale,
                               CPUOPBase::data_min_, CPUOPBase::data_max_);
    }
  }
}
template <typename DType>
void HardSoftmax<DType>::softmax() {
  if (fmap_i_.size() == 1) {
    UNI_LOG_CHECK(axis_ == 0, VART_INVALID_VALUE) << ", " << axis_ << " != 0";
    poly_softmax(img_, (int)(fmap_i_[0]), rlt_, input_fix_point_, is_bf16_in_);
    bf16_to_int8((int)(fmap_i_[0]), rlt_);
  } else if (fmap_i_.size() == 2) {
    UNI_LOG_CHECK(axis_ == 1, VART_INVALID_VALUE) << ", " << axis_ << " != 1";
    for (auto i = 0; i < fmap_i_[0]; i++) {
      poly_softmax(img_ + i * fmap_i_[1], (int)fmap_i_[1],
                   rlt_ + i * fmap_i_[1], input_fix_point_, is_bf16_in_);
      bf16_to_int8((int)fmap_i_[1], rlt_ + i * fmap_i_[1]);
    }
  } else if (fmap_i_.size() == 3) {
    UNI_LOG_CHECK(axis_ == 2, VART_INVALID_VALUE) << ", " << axis_ << " != 2";
    for (auto i = 0; i < fmap_i_[0]; i++) {
      for (auto j = 0; j < fmap_i_[1]; j++) {
        poly_softmax(img_ + i * fmap_i_[1] * fmap_i_[2] + j * fmap_i_[2],
                     (int)fmap_i_[2],
                     rlt_ + i * fmap_i_[1] * fmap_i_[2] + j * fmap_i_[2],
                     input_fix_point_, is_bf16_in_);
        bf16_to_int8((int)fmap_i_[2],
                     rlt_ + i * fmap_i_[1] * fmap_i_[2] + j * fmap_i_[2]);
      }
    }
  } else if (fmap_i_.size() == 4) {
    UNI_LOG_CHECK(axis_ == 3, VART_INVALID_VALUE) << ", " << axis_ << " != 3";

    for (auto i = 0; i < fmap_i_[0]; i++) {
      for (auto j = 0; j < fmap_i_[1]; j++) {
        for (auto k = 0; k < fmap_i_[2]; k++) {
          poly_softmax(img_ + i * fmap_i_[1] * fmap_i_[2] * fmap_i_[3] +
                           j * fmap_i_[2] * fmap_i_[3] + k * fmap_i_[3],
                       (int)fmap_i_[3],
                       rlt_ + i * fmap_i_[1] * fmap_i_[2] * fmap_i_[3] +
                           j * fmap_i_[2] * fmap_i_[3] + k * fmap_i_[3],
                       input_fix_point_, is_bf16_in_);
          bf16_to_int8((int)fmap_i_[3],
                       rlt_ + i * fmap_i_[1] * fmap_i_[2] * fmap_i_[3] +
                           j * fmap_i_[2] * fmap_i_[3] + k * fmap_i_[3]);
        }
      }
    }
  }
}

INSTANTIATE_TPCLASS(HardSoftmax);
REG_OP_INSTANCE_FUNC("hard-softmax", HardSoftmax);
REG_OP_INSTANCE_FUNC("hard-softmax-fix", HardSoftmax);

}  // namespace cpu
}  // namespace vart
