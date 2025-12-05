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

#include "eltwise_fix.hpp"

namespace vart {
namespace cpu {

namespace {
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

uint32_t u32_to_u16(uint32_t u) {
  std::uint32_t flag = (u & 0x00010000) >> 16;
  u = (u + 0x7fff + flag) & 0xFFFF0000;
  return u;
}

float reciprocal_approx_moroz(float z) {
  std::uint32_t magic_n = u32_to_u16(0x7eb53567);
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

float fast_sqrt(float z) {
  std::uint32_t magic_n = u32_to_u16(0x5f3759df);
  z = std::max(z, 0.0f); // return sqrt of negative number as 0
  float x = f_to_bf(z);
  float x2 = f_to_bf(x * f_to_bf(0.5f));
  float y = u_to_f(magic_n - (f_to_u(x) >> 1));
  x2 = f_to_bf(x2 * y);
  x2 = f_to_bf(x2 * y);
  float t = f_to_bf(f_to_bf(1.5f) - x2);
  x2 = f_to_bf(t * y);
  y = f_to_bf(x2 * x);
  return (y);
}

float fast_elu(float z, float alpha) {
  float static_z = z;
  z = std::min(std::max(z, -88.f), 88.f);
  float rcp_ln2 = f_to_bf(1.4453125);
  float x_bf16 = f_to_bf(z);
  float q = x_bf16 * rcp_ln2;
  int q_int = floor(q);
  float ln2 = 0.69140625;
  float approx = f_to_bf(float(q_int) * ln2);
  float m = x_bf16 - approx;

  float alpha_0 = f_to_bf(1.0);
  float alpha_1 = f_to_bf(1.0);
  float alpha_2 = f_to_bf(0.486328125);
  float alpha_3 = f_to_bf(0.21875);

  float exp_m = alpha_3 * m + alpha_2;
  exp_m = f_to_bf(exp_m) * m + alpha_1;
  exp_m = f_to_bf(f_to_bf(exp_m) * m + alpha_0);

  float pow2_q = pow(2.0, q_int);
  exp_m = f_to_bf(exp_m * pow2_q);

  float result = (static_z > 0.f ) ? f_to_bf(static_z) : f_to_bf(f_to_bf(alpha)*(exp_m -1.f));

  return result;
}

}  // namespace
double dr(double x);

template <typename DType>
EltwiseFix<DType>::EltwiseFix(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : Eltwise<DType>(subg, op, inputs, output) {
  auto nonlinear_type_str = op->get_attr<string>("nonlinear");
  if (nonlinear_type_str == "NONE" || nonlinear_type_str == "")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_NONE;
  else if (nonlinear_type_str == "RELU")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_RELU;
  else if (nonlinear_type_str == "PRELU")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_PRELU;
  else if (nonlinear_type_str == "LEAKYRELU")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_LEAKY_RELU;
  else if (nonlinear_type_str == "RELU6")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_RELU6;
  else if (nonlinear_type_str == "HSIGMOID")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_HSIGMOID;
  else
    UNI_LOG_FATAL(VART_NOT_SUPPORT)
        << "Unsupported nonlinear type: " << nonlinear_type_str << ".";

  for (auto i = 0; i < input_num_; i++) {
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input", i);
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
    auto fp = xir_tensor_i->get_attr<int>("fix_point");
    fp_inputs_.push_back(fp);
  }

  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");


  if (elt_type_ == "CLAMP") {
    minval_ = op->get_attr<std::int32_t>("min");
    maxval_ = op->get_attr<std::int32_t>("max");
  }
   if (elt_type_ == "ELU") {
    alpha_val_ = op->get_attr<float>("alpha");
  }
  
  if (elt_type_ == "ADD" || elt_type_ == "SUB" || elt_type_ == "MAX" ||
      elt_type_ == "MIN" || elt_type_ == "RELU") {
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i] - fp_min);
    }
    shift_write_ = fp_min - fp_output_;
  } else if (elt_type_ == "ABS" || elt_type_ == "NEG") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    shift_write_ = fp_inputs_[0] - fp_output_;
  } else if (elt_type_ == "MUL") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    auto fp_sum = std::accumulate(fp_inputs_.begin(), fp_inputs_.end(), 0);
    shift_write_ = fp_sum - fp_output_;
  } else if (elt_type_ == "DIV") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    shift_write_ = fp_inputs_[0] - fp_inputs_[1] - fp_output_;
  } else if (elt_type_ == "EQUAL" || elt_type_ == "LESS" ||
             elt_type_ == "LESS-EQUAL" || elt_type_ == "GREATER" ||
             elt_type_ == "GREATER-EQUAL" || elt_type_ == "NOT" ||
             elt_type_ == "AND" || elt_type_ == "OR") {
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    fp_min = std::min(0, fp_min);
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i] - fp_min);
    }
    shift_write_ = 0;
  } else if (elt_type_ == "SQRT") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i]);
    }
    shift_write_ = - fp_output_;
  } else if (elt_type_ == "CLAMP" || elt_type_ == "ELU") {
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    fp_min = std::min(0, fp_min);
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i] - fp_min);
    }
    shift_write_ = - fp_output_;
  }


  if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_HSIGMOID) {
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<std::int32_t>("hsigmoid_in");
    shift_hsigmoid_ =
        CPUOPBase::xir_op_->get_attr<std::int32_t>("shift_hsigmoid");
    if (elt_type_ == "ADD" || elt_type_ == "SUB" || elt_type_ == "MAX" ||
        elt_type_ == "MIN") {
      auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
      shift_write_ = fp_min - hsigmoid_in_;
    }
  }
}

template <typename DType>
void EltwiseFix<DType>::print_param() {
  Eltwise<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "nonlinear_type = " << nonlinear_type_ << endl;
  if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_HSIGMOID) {
    UNI_LOG_DEBUG_INFO << "shift_hsigmoid = " << shift_hsigmoid_ << endl;
    UNI_LOG_DEBUG_INFO << "hsigmoid_in = " << hsigmoid_in_ << endl;
  }
  UNI_LOG_DEBUG_INFO << "broadcast = " << broadcast_ << endl;
  UNI_LOG_DEBUG_INFO << "bit_width = " << CPUOPBase::bit_width_ << endl;
  for (auto i = 0; i < input_num_; i++) {
    UNI_LOG_DEBUG_INFO << "fp_input" << i << " = " << fp_inputs_[i] << endl;
  }
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;

  for (auto i = 0; i < input_num_; i++) {
    UNI_LOG_DEBUG_INFO << "shift_read" << i << " = " << shift_read_[i] << endl;
  }
  UNI_LOG_DEBUG_INFO << "shift_write = " << shift_write_ << endl;
}

template <typename DType>
void EltwiseFix<DType>::eltwise(std::uint32_t start_index,
                                std::uint32_t end_index) {
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;
  for (auto pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    double tmp = (elt_type_ == "MUL" || elt_type_ == "DIV") ? 1 : 0;
    int tmp_int = 0;
    for (auto fp_iter = 0U; fp_iter < fmap_i_.size(); fp_iter++) {
      // dim_t pos;
      auto pos = pos_iter;
      if (broadcast_) {
        fmap_o_.pos2coord(pos_iter, dst_coord);
        for (auto dim_iter = 0U; dim_iter < dst_coord.size(); dim_iter++) {
          src_coord[dim_iter] =
              dst_coord[dim_iter] % fmap_i_[fp_iter][dim_iter];
        }
        pos = fmap_i_[fp_iter].coord2pos(src_coord);
      }
      if (elt_type_ == "ADD") {
        tmp += floor((double)data_in_[fp_iter][pos] * 4 /
                     pow(2.0, shift_read_[fp_iter]));
      } else if (elt_type_ == "RELU") {
        tmp = std::max(floor((double)data_in_[fp_iter][pos] * 4 /
                             pow(2.0, (shift_read_[fp_iter]))),
                       double(0));
      } else if (elt_type_ == "SUB") {
        if (fp_iter == 0U) {
          tmp = floor((double)data_in_[fp_iter][pos] *
                      pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          tmp -= floor((double)data_in_[fp_iter][pos] *
                       pow(2.0, 7 - shift_read_[fp_iter]));
        }
      } else if (elt_type_ == "MUL") {
        tmp *= floor((double)data_in_[fp_iter][pos] * 4 /
                     pow(2.0, shift_read_[fp_iter]));
      } else if (elt_type_ == "DIV") {
        if (fp_iter == 0U) {
          tmp =
              f_to_bf(data_in_[fp_iter][pos] / pow(2.0, shift_read_[fp_iter]));
        } else {

          tmp *= reciprocal_approx_moroz(data_in_[fp_iter][pos] /
                                         pow(2.0, shift_read_[fp_iter]));
          tmp = f_to_bf(tmp);
        }

      } else if (elt_type_ == "CLAMP") {
        tmp = std::min(std::max(floor((double)data_in_[fp_iter][pos] * 4 /
                                      pow(2.0, shift_read_[fp_iter])),
                                static_cast<double>(minval_ * 4)),
                       static_cast<double>(maxval_ * 4));
      } else if (elt_type_ == "MAX") {
        if (fp_iter == 0U) {
          tmp = floor((double)data_in_[fp_iter][pos] *
                      pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 > tmp1 ? tmp0 : tmp1;
        }
      } else if (elt_type_ == "MIN") {
        if (fp_iter == 0U) {
          tmp = floor((double)data_in_[fp_iter][pos] *
                      pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 > tmp1 ? tmp1 : tmp0;
        }
      } else if (elt_type_ == "EQUAL") {
        if (fp_iter == 0U) {
          tmp_int = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        } else {
          int tmp0 = tmp_int;
          int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
          tmp = tmp0 == tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "LESS") {
        if (fp_iter == 0U) {
          tmp_int = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        } else {
          int tmp0 = tmp_int;
          int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
          tmp = tmp0 < tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "LESS-EQUAL") {
        if (fp_iter == 0U) {
          tmp_int = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        } else {
          int tmp0 = tmp_int;
          int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
          tmp = tmp0 <= tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "GREATER") {
        if (fp_iter == 0U) {
          tmp_int = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        } else {
          int tmp0 = tmp_int;
          int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
          tmp = tmp0 > tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "GREATER-EQUAL") {
        if (fp_iter == 0U) {
          tmp_int = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        } else {
          int tmp0 = tmp_int;
          int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
          tmp = tmp0 >= tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "NOT") {
        int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        tmp = !tmp1;
      } else if (elt_type_ == "AND") {
        if (fp_iter == 0U) {
          tmp_int = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        } else {
          int tmp0 = tmp_int;
          int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
          tmp = tmp0 && tmp1;
        }
      } else if (elt_type_ == "OR") {
        if (fp_iter == 0U) {
          tmp_int = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
        } else {
          int tmp0 = tmp_int;
          int tmp1 = (int)data_in_[fp_iter][pos] >> (int)shift_read_[fp_iter];
          tmp = tmp0 || tmp1 ;
        }
        } else if (elt_type_ == "ABS") {
        // float tmp1 = (float)data_in_[fp_iter][pos] / pow(2.0, shift_read_[fp_iter]);
        tmp = floor((double)data_in_[fp_iter][pos] * 4 /
                      pow(2.0, shift_read_[fp_iter]));
        tmp = tmp > 0 ? tmp : (-tmp);
      } else if (elt_type_ == "NEG") {
        tmp = floor((double)data_in_[fp_iter][pos] * 4 /
                      pow(2.0, shift_read_[fp_iter]));
        tmp = -tmp;
      } else if (elt_type_ == "SQRT") {
        tmp = fast_sqrt(data_in_[fp_iter][pos] / pow(2.0, shift_read_[fp_iter]));
      } else if (elt_type_ == "ELU") {  
        tmp = fast_elu(floor((double)data_in_[fp_iter][pos]  /
                      pow(2.0, shift_read_[fp_iter]) + 0.5), alpha_val_);
      }
    }

    // tmp /= pow(2.0, shift_write_);
    // float tmp0;
    if (elt_type_ == "ABS" || elt_type_ == "NEG") {
      // tmp0 = tmp;
      tmp = tmp / pow(2.0, shift_write_);
      tmp /=4;
      data_out_[pos_iter] =
          DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }

    if (elt_type_ == "EQUAL" || elt_type_ == "LESS" ||
        elt_type_ == "LESS-EQUAL" || elt_type_ == "GREATER" ||
        elt_type_ == "GREATER-EQUAL" || elt_type_ == "AND" ||
        elt_type_ == "OR" || elt_type_ == "NOT" ) {
      data_out_[pos_iter] =
          DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    if(elt_type_ == "ELU"){
      data_out_[pos_iter] = DPURound<DType>(floor(floor(f_to_bf(tmp) + 0.5) /pow(2.0, shift_write_) + 0.5), CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    tmp /= pow(2.0, shift_write_);
    
    if (elt_type_ == "DIV" || elt_type_ == "SQRT") {
      tmp = floor(f_to_bf(tmp));
    } else if (elt_type_ == "SUB" || elt_type_ == "MAX" || elt_type_ == "MIN") {
      tmp /= pow(2, 7);
    } else {
      tmp /= (elt_type_ == "ADD")
                 ? 4
                 : pow(4, fmap_i_.size());
    }
    if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_RELU) {
      if (tmp < 0) tmp = 0;
    } else if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_LEAKY_RELU) {
      if (tmp < 0) tmp *= 26.f / 256.f;
    } else if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_RELU6) {
      if (tmp < 0) tmp = 0;
      if (fp_output_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_HSIGMOID) {
      tmp = dr(tmp);
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    }
    data_out_[pos_iter] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS(EltwiseFix);
REG_OP_INSTANCE_FUNC("eltwise-fix", EltwiseFix);

}  // namespace cpu
}  // namespace vart
