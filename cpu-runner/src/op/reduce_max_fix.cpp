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

#include "reduce_max_fix.hpp"

#include "pair_hash.hpp"

namespace vart {
namespace cpu {

template <typename DType>
ReduceMaxFix<DType>::ReduceMaxFix(const xir::Subgraph* subg, const xir::Op* op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : ReduceMax<DType>(subg, op, inputs, output) {
  auto xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  auto xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  fp_input_ = xir_tensor_i->get_attr<int>("fix_point");
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
  fp_shift_ = fp_output_ - fp_input_;
  pow_shift_ = pow(2.0, fp_shift_);
}

template <typename DType>
void ReduceMaxFix<DType>::print_param() {
  ReduceBase<DType>::print_param();
  UNI_LOG_DEBUG_INFO << "fp_input = " << fp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_shift_ = " << fp_shift_ << endl;
  UNI_LOG_DEBUG_INFO << "pow_shift = " << pow_shift_ << endl;
}

template <typename DType>
void ReduceMaxFix<DType>::calculate() {
  ReduceMax<DType>::calculate();
  for (auto i = 0; i < fmap_i_.num(); i++) {
    data_in_ptr_[i] =
        DPURound<DType>(data_in_ptr_[i] * pow_shift_, CPUOPBase::data_min_,
                        CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS(ReduceMaxFix);
REG_OP_INSTANCE_FUNC("reduction_max-fix", ReduceMaxFix);
}  // namespace cpu
}  // namespace vart