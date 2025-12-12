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

#include "reduce_mean_fix.hpp"

#include "pair_hash.hpp"

namespace vart {
namespace cpu {

template <typename DType>
ReduceMeanFix<DType>::ReduceMeanFix(const xir::Subgraph* subg, const xir::Op* op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : ReduceMean<DType>(subg, op, inputs, output) {
  auto xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  auto xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  fp_input_ = xir_tensor_i->get_attr<int>("fix_point");
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
  fp_shift_ = fp_output_ - fp_input_;
  pow_shift_ = pow(2.0, fp_shift_);
}

std::pair<int32_t, int32_t> get_mean_dpu_fix_factors(
    const std::int32_t & input_div) {
  auto rec = input_div;
  auto multi_factor = 0;
  auto shift_factor = 0;
  auto diff = 1.f;
    auto max_factor = std::ceil(std::log2(rec * 128));
    for (auto shift_factor_ = 0; shift_factor_ < max_factor; shift_factor_++) {
      auto factor = std::round(std::exp2(shift_factor_) / rec);
      auto diff_ = std::abs(factor / std::exp2(shift_factor_) - 1.f / rec);
      if (diff_ < diff) {
        multi_factor = factor;
        diff = diff_;
        shift_factor = shift_factor_;
      }
    }
  return {multi_factor, shift_factor};
}

/// return the avgpool coefficient according to the kernel size
//float get_mean_dpu_coefficient(const std::int32_t & input_div) {
//  auto factors = get_mean_dpu_factors(input_div);
//  return float(factors.first) / std::exp2(factors.second);
//}

template <typename DType>
void ReduceMeanFix<DType>::print_param() {
  ReduceBase<DType>::print_param();
  UNI_LOG_DEBUG_INFO << "fp_input = " << fp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_shift_ = " << fp_shift_ << endl;
  UNI_LOG_DEBUG_INFO << "pow_shift = " << pow_shift_ << endl;
}

template <typename DType>
void ReduceMeanFix<DType>::calculate() {
//  ReduceMean<DType>::calculate();
  // caculate
  double factor = 1.0;
  float pow_shift = 1.0;
  for (auto idx : reduce_dims_) {
    auto cur_dim = fmap_i_.dim(idx);
   // factor = factor / cur_dim;
    auto factors = get_mean_dpu_fix_factors(cur_dim);
    factor *= (double)(factors.first);
	pow_shift *= std::exp2(factors.second);
    auto cur_cod = fmap_i_.cod(idx);
    auto n_cur_chunk = fmap_i_.num() / (cur_dim * cur_cod);

    for (auto i = 0; i < n_cur_chunk; i++) {
      DType* prlt = data_in_ptr_ + i * cur_dim * cur_cod;

      for (auto j = 1; j < cur_dim; j++) {
        DType* pcur = prlt + j * cur_cod;

        // accumulate
        for (auto k = 0; k < cur_cod; k++) {
          prlt[k] += pcur[k];
        }
      }
    }
  }
  for (auto i = 0; i < fmap_i_.num(); i++) {
	double tmp = 0.f;
	tmp = (double)data_in_ptr_[i] * factor / pow_shift;
    data_in_ptr_[i] =
        DPURound<DType>(tmp * pow_shift_, CPUOPBase::data_min_,
                        CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS(ReduceMeanFix);
REG_OP_INSTANCE_FUNC("reduction_mean-fix", ReduceMeanFix);
}  // namespace cpu
}  // namespace vart
