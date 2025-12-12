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

#include "permute.hpp"
#include "cpu_types.hpp"

namespace vart {
namespace cpu {

template <typename DType>
Permute<DType>::Permute(const xir::Subgraph* subg, const xir::Op* op,
                        IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  o_shape_ = op->get_output_tensor()->get_shape();
  for (unsigned i = 0; i < o_shape_.size(); i++) {
    o_prod_.push_back(std::accumulate(
      begin(o_shape_) + i, end(o_shape_), 1, std::multiplies<int>()));
  }
  i_shape_ = op->get_input_ops("input")[0]->get_output_tensor()->get_shape();
  for (unsigned i = 0; i < i_shape_.size(); i++) {
    i_prod_.push_back(std::accumulate(
      begin(i_shape_) + i, end(i_shape_), 1, std::multiplies<int>()));
  }
  order_ = xir_op_->get_attr<std::vector<std::int32_t>>("order");
}

template <typename DType>
void Permute<DType>::run() {
  read();
  permute();
}

template <typename DType>
void Permute<DType>::read() {
  auto* data_in_tb = inputs_.at("input").at(0);
  data_in_ptr_ = GET_CPUTB_DType_PTR(DType, data_in_tb);
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
void Permute<DType>::permute() {
  if (i_shape_.size() == 1) {
    for (auto i = 0; i < o_prod_[0]; i++) {
      data_out_ptr_[i] = data_in_ptr_[i];
    }
    return;
  }
  for (auto i = 0; i < i_prod_[0]; i++) {
    std::vector<int> dims;
    dims.push_back(i / i_prod_[1]);
    int left = i;
    for (unsigned j = 1; j < i_shape_.size(); j++) {
      left = left - dims[dims.size() - 1] * i_prod_[dims.size()];
      if (j == i_shape_.size() - 1)
        dims.push_back(left % i_prod_[j]);
      else
        dims.push_back(left / i_prod_[j + 1]);
    }
    int out_addr = dims[order_[dims.size() - 1]];
    for (unsigned d = 0; d < dims.size() - 1; d++) {
      out_addr += dims[order_[d]] * o_prod_[d + 1];
    }
    data_out_ptr_[out_addr] = data_in_ptr_[i];
  }
}

INSTANTIATE_TPCLASS(Permute);
REG_OP_INSTANCE_FUNC("permute", Permute);
REG_OP_INSTANCE_FUNC("transpose", Permute);

}  // namespace cpu
}  // namespace vart
