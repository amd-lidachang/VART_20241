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

#include "cost_volume_fix.hpp"

namespace vart {
namespace cpu {

template <typename DType>
CostVolumeFix<DType>::CostVolumeFix(const xir::Subgraph* subg,
                                    const xir::Op* op, IMapTBs_t inputs,
                                    CPUTBPtr_t output)
    : CostVolume<DType>(subg, op, inputs, output) {
  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  UNI_LOG_CHECK(xir_tensor_o->has_attr("fix_point"), VART_VALUE_ERROR);
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");

  for (auto i = 0; i < input_num_; ++i) {
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input", i);
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
    UNI_LOG_CHECK(xir_tensor_i->has_attr("fix_point"), VART_VALUE_ERROR);
    auto fp_input = xir_tensor_i->get_attr<int>("fix_point");
    UNI_LOG_CHECK(fp_input == fp_output_, VART_VALUE_ERROR);
    fp_inputs_.push_back(fp_input);
  }
}

template <typename DType>
void CostVolumeFix<DType>::print_param() {
  CostVolume<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "bit_width = " << CPUOPBase::bit_width_ << endl;
  for (auto i = 0; i < input_num_; i++) {
    UNI_LOG_DEBUG_INFO << "fp_input" << i << " = " << fp_inputs_[i] << endl;
  }
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
}

template <typename DType>
void CostVolumeFix<DType>::cost_volume(std::int32_t start_index,
                                       std::int32_t end_index) {
  for (auto idx = start_index; idx < end_index; ++idx) {
    auto fmap_tmp = fmap_o_.pos2coord(idx);
    if (fmap_tmp.w < fmap_tmp.d) {
      data_out_[idx] = 0;
    } else {
      // input_0
      if (fmap_tmp.c < fmap_i_[0].c) {
        auto addr = fmap_tmp.n * fmap_i_[0].ncod() +
                    fmap_tmp.h * fmap_i_[0].hcod() +
                    fmap_tmp.w * fmap_i_[0].wcod() + fmap_tmp.c;
        auto data = DPURound<DType>((double)data_in_[0][addr],
                                    CPUOPBase::data_min_, CPUOPBase::data_max_);
        data_out_[idx] = data;
      } else {  // input_1
        auto addr = fmap_tmp.n * fmap_i_[1].ncod() +
                    fmap_tmp.h * fmap_i_[1].hcod() +
                    (fmap_tmp.w - fmap_tmp.d) * fmap_i_[1].wcod() +
                    (fmap_tmp.c - fmap_i_[1].c);
        auto data = DPURound<DType>((double)data_in_[1][addr],
                                    CPUOPBase::data_min_, CPUOPBase::data_max_);
        data_out_[idx] = data;
      }
    }
  }
}

INSTANTIATE_TPCLASS(CostVolumeFix);
REG_OP_INSTANCE_FUNC("cost_volume-fix", CostVolumeFix);

}  // namespace cpu
}  // namespace vart
