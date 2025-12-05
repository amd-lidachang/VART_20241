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

#include "argmax.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename OType>
const vector<string> ArgMax<DType, OType>::ITName = {
  "input",
};

// constructor and deconstructor
template <typename DType, typename OType>
ArgMax<DType, OType>::ArgMax(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 6);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 6);

  // get pad info
  axis_ = xir_op_->get_attr<int>("axis");
  axis_ = axis_ == -1
            ? (xir_op_->get_input_tensor("input")->get_shape().size() - 1)
            : axis_;
}

template <typename DType, typename OType>
void ArgMax<DType, OType>::run() {
  // // do necessary check
  // print_param();
  // check_param();

  // // read data
  // read();

  // do argmax
  argmax();

  // // do save, debug...
  // save();
}

template <typename DType, typename OType>
void ArgMax<DType, OType>::print_param() {
  UNI_LOG_DEBUG_INFO << "fmap_i_ = " << Vec2Str(fmap_i_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "fmap_o_ = " << Vec2Str(fmap_o_, ", ") << endl;

  UNI_LOG_DEBUG_INFO << "axis = " << axis_ << endl;
}

template <typename DType, typename OType>
void ArgMax<DType, OType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() == 1, VART_SIZE_ERROR);

  // check fmap_out height
  UNI_LOG_CHECK(fmap_i_.size() == fmap_o_.size(), VART_SIZE_ERROR)
    << fmap_i_.size() << " != " << fmap_o_.size();
  for (auto i = 0U; i < fmap_i_.size(); i++) {
    UNI_LOG_CHECK(fmap_i_[i] == fmap_o_[i], VART_INVALID_VALUE)
      << ", " << fmap_i_[i] << " != " << fmap_o_[i];
  }
}

// read data from DDR
template <typename DType, typename OType>
void ArgMax<DType, OType>::read() {
  // read img data
  auto* cputb = inputs_.at(ITName[INPUT]).at(0);
  img_ = GET_CPUTB_DType_PTR(DType, cputb);

  // handle output buffer
  rlt_ = GET_CPUTB_DType_PTR(OType, output_);
}

template <typename DType, typename OType>
uint64_t ArgMax<DType, OType>::get_workload() {
  // TODO
  return 0;
}

template <typename DType, typename OType>
void ArgMax<DType, OType>::argmax() {
  int left = 1, right = 1;
  for (unsigned i = axis_ + 1; i < fmap_i_.size(); i++) right *= fmap_i_[i];
  for (int i = 0; i < axis_; i++) left *= fmap_i_[i];
  for (auto i = 0; i < left; i++)
    for (auto j = 0; j < right; j++) {
      DType max = *(img_ + i * fmap_i_[axis_] * right + j);
      rlt_[i * 1 * right + j] = 0;
      for (auto idx = 0; idx < fmap_i_[axis_]; idx++) {
        auto cur = *(img_ + i * fmap_i_[axis_] * right + idx * right + j);
        if (cur > max) {
          rlt_[i * right + j] = idx;
          max = cur;
        }
      }
    }
}

INSTANTIATE_TPCLASS_SPECIFIED(ArgMax);
REG_SPECIFIED_DATATYPE_OP_INSTANCE_FUNC("argmax", ArgMax);
REG_SPECIFIED_DATATYPE_OP_INSTANCE_FUNC("argmax-fix", ArgMax);

}  // namespace cpu
}  // namespace vart