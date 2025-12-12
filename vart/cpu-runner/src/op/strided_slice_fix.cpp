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

#include "strided_slice_fix.hpp"

namespace vart {
namespace cpu {
template <typename DType>
StridedSliceFix<DType>::StridedSliceFix(const xir::Subgraph* subg,
                                        const xir::Op* op, IMapTBs_t inputs,
                                        CPUTBPtr_t output)
    : StridedSlice<DType>(subg, op, inputs, output) {
  auto xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
  fp_input_ = xir_tensor_i->get_attr<int>("fix_point");

  auto xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
  shift_ = fp_output_ - fp_input_;
}

template <typename DType>
void StridedSliceFix<DType>::print_param() {
  StridedSlice<DType>::print_param();
  UNI_LOG_DEBUG_INFO << "fp_input = " << fp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
  UNI_LOG_DEBUG_INFO << "shift = " << shift_ << endl;
}

template <typename DType>
void StridedSliceFix<DType>::calculate(std::uint32_t start_index,
                                       std::uint32_t end_index) {
  for (uint32_t pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    int32_t dim3_iter = pos_iter % fmap_o_[3];
    int32_t dim2_iter = (pos_iter / (fmap_o_[3])) % fmap_o_[2];
    int32_t dim1_iter = (pos_iter / (fmap_o_[3] * fmap_o_[2])) % fmap_o_[1];
    int32_t dim0_iter = pos_iter / (fmap_o_[3] * fmap_o_[2] * fmap_o_[1]);

    int32_t src_pos =
        ((dim0_iter * strides_[0]) + begin_[0]) * fmap_i_[1] * fmap_i_[2] *
            fmap_i_[3] +
        ((dim1_iter * strides_[1]) + begin_[1]) * fmap_i_[2] * fmap_i_[3] +
        ((dim2_iter * strides_[2]) + begin_[2]) * fmap_i_[3] +
        ((dim3_iter * strides_[3]) + begin_[3]);
    data_out_[pos_iter] =
        DPURound<DType>(data_in_[src_pos] * pow(2, shift_),
                        CPUOPBase::data_min_, CPUOPBase::data_max_);
    ;
  }
}

INSTANTIATE_TPCLASS(StridedSliceFix);
REG_OP_INSTANCE_FUNC("strided_slice-fix", StridedSliceFix);
}  // namespace cpu
}  // namespace vart