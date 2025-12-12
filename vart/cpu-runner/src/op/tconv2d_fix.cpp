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

#include "tconv2d_fix.hpp"

namespace vart {
namespace cpu {

double dr(double x);

// constructor and deconstructor
template <typename DType, typename WType>
TConv2dFix<DType, WType>::TConv2dFix(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : TConv2d<DType, WType>(subg, op, inputs, output) {
  auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  fp_input_ = xir_tensor_i->get_attr<int>("fix_point");

  auto* xir_tensor_w = CPUOPBase::xir_op_->get_input_tensor("weights");
  fp_weights_ = xir_tensor_w->get_attr<int>("fix_point");

  if (has_bias_) {
    auto* xir_tensor_b = CPUOPBase::xir_op_->get_input_tensor("bias");
    fp_bias_ = xir_tensor_b->get_attr<int>("fix_point");
    shift_bias_ = fp_input_ + fp_weights_ - fp_bias_;
  }

  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
  shift_cut_ = fp_input_ + fp_weights_ - fp_output_;

  auto nonlinear_type_str = CPUOPBase::xir_op_->get_attr<string>("nonlinear");
  if (nonlinear_type_str == "NONE" || nonlinear_type_str == "") {
    nonlinear_type_ = 0;
  } else if (nonlinear_type_str == "RELU") {
    nonlinear_type_ = 1;
  } else if (nonlinear_type_str == "PRELU") {
    nonlinear_type_ = 2;
  } else if (nonlinear_type_str == "LEAKYRELU") {
    nonlinear_type_ = 3;
  } else if (nonlinear_type_str == "RELU6") {
    nonlinear_type_ = 4;
  } else if (nonlinear_type_str == "HSIGMOID") {
    nonlinear_type_ = 5;
    shift_bias_ = CPUOPBase::xir_op_->get_attr<int>("shift_bias");
    shift_cut_ = CPUOPBase::xir_op_->get_attr<int>("shift_cut");
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = CPUOPBase::xir_op_->get_attr<int>("shift_hsigmoid");
  } else if (nonlinear_type_str == "HSWISH") {
    nonlinear_type_ = 6;
    shift_bias_ = CPUOPBase::xir_op_->get_attr<int>("shift_bias");
    shift_cut_ = CPUOPBase::xir_op_->get_attr<int>("shift_cut");
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = CPUOPBase::xir_op_->get_attr<int>("shift_hsigmoid");
    shift_hswish_ = CPUOPBase::xir_op_->get_attr<int>("shift_hswish");
  } else {
    UNI_LOG_FATAL(VART_NOT_SUPPORT)
        << "Unsupported nonlinear type: " << nonlinear_type_str << ".";
  }
}

template <typename DType, typename WType>
void TConv2dFix<DType, WType>::print_param() {
  ConvBase<DType, WType>::print_param();

  UNI_LOG_DEBUG_INFO << "fp_input: " << fp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_weights: " << fp_weights_ << endl;
  if (has_bias_) {
    UNI_LOG_DEBUG_INFO << "fp_bias: " << fp_bias_ << endl;
  }
  UNI_LOG_DEBUG_INFO << "fp_output: " << fp_output_ << endl;

  if (has_bias_) {
    UNI_LOG_DEBUG_INFO << "shift_bias: " << shift_bias_ << endl;
  }
  UNI_LOG_DEBUG_INFO << "shift_cut: " << shift_cut_ << endl;
  UNI_LOG_DEBUG_INFO << "nonlinear_type: " << nonlinear_type_ << endl;
}

template <typename DType, typename WType>
void TConv2dFix<DType, WType>::check_param() {
  ConvBase<DType, WType>::check_param();

  UNI_LOG_CHECK(nonlinear_type_ == 0 || nonlinear_type_ == 1 ||
                    nonlinear_type_ == 3 || nonlinear_type_ == 4,
                VART_INVALID_VALUE)
      << " not supported. ";
}

template <typename DType, typename WType>
void TConv2dFix<DType, WType>::run() {
  // // do necessary check
  // print_param();
  // check_param();

  // // read data
  // this->read();

  // calc conv
  this->conv();

  // do fix
  this->fix();

  // // do save, debug...
  // this->save();
}

template <typename DType, typename WType>
void TConv2dFix<DType, WType>::fix() {
  for (auto i = 0; i < fmap_o_.num(); i++) {
    data_out_ptr_[i] *= 2;
    // do fix for bias
    if (has_bias_) {
      auto pos = i % fmap_o_.c;
      data_out_ptr_[i] += floor(2.0 * bias_ptr_[pos] * pow(2.0, shift_bias_));
    }
  }

  // do fix for output
  auto factor = pow(2, shift_cut_ + 1);
  for (auto i = 0; i < fmap_o_.num(); i++) {
    double tmp = data_out_ptr_[i];
    tmp /= factor;

    if (nonlinear_type_ == 1) {
      if (tmp < 0) tmp = 0;
    } else if (nonlinear_type_ == 3) {
      if (tmp < 0) tmp = tmp * 26. / 256;
    } else if (nonlinear_type_ == 4) {
      if (tmp < 0) tmp = 0;
      if (fp_output_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (nonlinear_type_ == 4) {
      if (tmp < 0) tmp = 0;
      if (fp_output_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (nonlinear_type_ == 5) {
      tmp = dr(tmp);
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    } else if (nonlinear_type_ == 6) {
      auto x = dr(tmp);
      auto hsigmoid_x =
          dr(std::min(
                 pow(2, 32),
                 std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
             pow(2, -shift_hsigmoid_));
      tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
    }
    data_out_ptr_[i] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(TConv2dFix);
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("transposed-conv2d-fix", TConv2dFix, "weights");

}  // namespace cpu
}  // namespace vart
