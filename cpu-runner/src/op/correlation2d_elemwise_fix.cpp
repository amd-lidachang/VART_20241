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

#include "correlation2d_elemwise_fix.hpp"
#include "align_buf_mgr.hpp"

namespace vart {
namespace cpu {

template <typename DType>
Correlation2dElemwiseFix<DType>::Correlation2dElemwiseFix(
    const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
    CPUTBPtr_t output)
    : Correlation2dElemwise<DType>(subg, op, inputs, output) {
  auto input_num = op->get_input_num();
  int fp_inputs_sum = 0;
  for (auto i = 0; i < input_num; ++i) {
    auto* xir_tensor_i = op->get_input_tensor("input", i);
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
    auto fp = xir_tensor_i->get_attr<int>("fix_point");
    fp_inputs_.push_back(fp);
    fp_inputs_sum += fp;
  }

  auto* xir_tensor_o = op->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
  shift_write_ = fp_output_ - fp_inputs_sum;
}

template <typename DType>
void Correlation2dElemwiseFix<DType>::print_param() {
  Correlation2dElemwise<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "bit_width = " << CPUOPBase::bit_width_ << endl;
  for (auto i = 0; i < 2; i++) {
    UNI_LOG_DEBUG_INFO << "fp_input" << i << " = " << fp_inputs_[i] << endl;
  }
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;

  UNI_LOG_DEBUG_INFO << "shift_write = " << shift_write_ << endl;
}

template <typename DType>
void Correlation2dElemwiseFix<DType>::eltwise(std::uint64_t start_index,
                                              std::uint64_t end_index) {
  auto stride = 2 * pad_size_ + 1;
  for (auto i = start_index; i < end_index; ++i) {
    auto fmap = fmap_o_.pos2coord(i);
    auto h_offset = fmap.d / stride;
    auto w_offset = fmap.d % stride;
    auto addr_offset = fmap.n * fmap_i_[1].ncod() +
                       h_offset * fmap_i_[1].hcod() +
                       w_offset * fmap_i_[1].wcod();
    auto* data_in_1 = data_in_[1] + addr_offset;

    auto addr_in_0 = fmap.n * fmap_i_[0].ncod() + fmap.h * fmap_i_[0].hcod() +
                     fmap.w * fmap_i_[0].wcod() + fmap.c;
    auto addr_in_1 = fmap.n * fmap_i_[1].ncod() + fmap.h * fmap_i_[1].hcod() +
                     fmap.w * fmap_i_[1].wcod() + fmap.c;

    auto tmp = (data_in_[0][addr_in_0] * data_in_1[addr_in_1]) *
               pow(2.0, shift_write_);
    data_out_[i] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS(Correlation2dElemwiseFix);
REG_OP_INSTANCE_FUNC("correlation2d_elemwise-fix", Correlation2dElemwiseFix);

}  // namespace cpu
}  // namespace vart

