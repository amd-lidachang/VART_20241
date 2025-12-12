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

#include "pad.hpp"

#include "fast_pad.hpp"

namespace vart {
namespace cpu {

template <typename DType>
Pad<DType>::Pad(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                CPUTBPtr_t output)
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
  mode_ = op->get_attr<std::string>("mode");
  val_ = op->get_attr<std::vector<char>>("constant_values");
  paddings_ = op->get_attr<std::vector<int> >("paddings");
  if (mode_ == "SYMMETRIC") {
    for (unsigned i = 0; i < i_shape_.size(); i++)
      UNI_LOG_CHECK(
        paddings_[2 * i] <= i_shape_[i] && paddings_[2 * i + 1] <= i_shape_[i],
        VART_INVALID_VALUE);
  } else if (mode_ == "REFLECT") {
    for (unsigned i = 0; i < i_shape_.size(); i++)
      UNI_LOG_CHECK(paddings_[2 * i] <= i_shape_[i] - 1 &&
                      paddings_[2 * i + 1] <= i_shape_[i] - 1,
                    VART_INVALID_VALUE);
  }
}

template <typename DType>
void Pad<DType>::run() {
  read();
  pad();
}

template <typename DType>
void Pad<DType>::read() {
  auto* cputb = inputs_.at("input").at(0);
  data_in_ptr_ = GET_CPUTB_DType_PTR(DType, cputb);
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
void Pad<DType>::pad() {
  if (i_shape_.size() == 1) {
    for (auto i = 0; i < o_prod_[0]; i++) {
      if (mode_ == "CONSTANT") {
        if (i >= paddings_[0] && i <= (paddings_[0] + i_shape_[0] - 1))
          data_out_ptr_[i] = data_in_ptr_[i - paddings_[0]];
      } else if (mode_ == "SYMMETRIC") {
        int input_addr;
        if (i < paddings_[0])
          input_addr = paddings_[0] - 1 - i;
        else if (i > paddings_[0] + i_shape_[0] - 1)
          input_addr = paddings_[0] + 2 * i_shape_[0] - 1 - i;
        else
          input_addr = i - paddings_[0];
        data_out_ptr_[i] = data_in_ptr_[input_addr];
      } else if (mode_ == "REFLECT") {
        int input_addr;
        if (i < paddings_[0])
          input_addr = paddings_[0] - i;
        else if (i > paddings_[0] + i_shape_[0] - 1)
          input_addr = paddings_[0] + 2 * i_shape_[0] - 2 - i;
        else
          input_addr = i - paddings_[0];
        data_out_ptr_[i] = data_in_ptr_[input_addr];
      }
    }
    return;
  }
  if (mode_ == "CONSTANT") {
    for (int i = 0; i < o_prod_[0]; i++) {
       data_out_ptr_[i] = val_[0];
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
      int output_addr =
        dims[dims.size() - 1] + paddings_[2 * (dims.size() - 1)];
      for (unsigned dim = 0; dim < dims.size() - 1; dim++) {
        output_addr += (paddings_[2 * dim] + dims[dim]) * o_prod_[dim + 1];
      }
      data_out_ptr_[output_addr] = data_in_ptr_[i];
    }
  } else {
    for (auto i = 0; i < o_prod_[0]; i++) {
      std::vector<int> dims;
      dims.push_back(i / o_prod_[1]);
      int left = i;
      for (unsigned j = 1; j < o_shape_.size(); j++) {
        left = left - dims[dims.size() - 1] * o_prod_[dims.size()];
        if (j == o_shape_.size() - 1)
          dims.push_back(left % o_prod_[j]);
        else
          dims.push_back(left / o_prod_[j + 1]);
      }
      if (mode_ == "SYMMETRIC") {
        for (unsigned i = 0; i < dims.size(); i++) {
          if (dims[i] < paddings_[2 * i])
            dims[i] = paddings_[2 * i] - 1 - dims[i];
          else if (dims[i] > paddings_[2 * i] + i_shape_[i] - 1)
            dims[i] = paddings_[2 * i] + 2 * i_shape_[i] - 1 - dims[i];
          else
            dims[i] = dims[i] - paddings_[2 * i];
        }
      } else if (mode_ == "REFLECT") {
        for (unsigned i = 0; i < dims.size(); i++) {
          if (dims[i] < paddings_[2 * i])
            dims[i] = paddings_[2 * i] - dims[i];
          else if (dims[i] > paddings_[2 * i] + i_shape_[i] - 1)
            dims[i] = paddings_[2 * i] + 2 * i_shape_[i] - 2 - dims[i];
          else
            dims[i] = dims[i] - paddings_[2 * i];
        }
      }
      int input_addr = dims[dims.size() - 1];
      for (unsigned dim = 0; dim < dims.size() - 1; dim++) {
        input_addr += dims[dim] * i_prod_[dim + 1];
      }
      data_out_ptr_[i] = data_in_ptr_[input_addr];
    }
  }
}

INSTANTIATE_TPCLASS(Pad);
REG_OP_INSTANCE_FUNC("pad", Pad);
REG_OP_INSTANCE_FUNC("pad-fix", Pad);

}  // namespace cpu
}  // namespace vart
