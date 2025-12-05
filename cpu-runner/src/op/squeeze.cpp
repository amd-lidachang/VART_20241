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

#include "squeeze.hpp"

namespace vart {
namespace cpu {
template <typename DType>
Squeeze<DType>::Squeeze(const xir::Subgraph* subg, const xir::Op* op,
                        IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  o_shape_ = op->get_output_tensor()->get_shape();
  num_ =
    std::accumulate(begin(o_shape_), end(o_shape_), 1, std::multiplies<int>());
  i_shape_ = op->get_input_ops("input")[0]->get_output_tensor()->get_shape();
}

template <typename DType>
void Squeeze<DType>::run() {
  read();
  squeeze();
}

// read data from DDR
template <typename DType>
void Squeeze<DType>::read() {
  auto* data_in_tb = inputs_.at("input").at(0);
  data_in_ptr_ = GET_CPUTB_DType_PTR(DType, data_in_tb);
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
void Squeeze<DType>::squeeze() {
  for (auto i = 0; i < num_; i++) data_out_ptr_[i] = data_in_ptr_[i];
}

INSTANTIATE_TPCLASS(Squeeze);
REG_OP_INSTANCE_FUNC("squeeze", Squeeze);

}  // namespace cpu
}  // namespace vart
