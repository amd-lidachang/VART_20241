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

#pragma once

#include "cpu_op_base.hpp"
#include "strided_slice.hpp"

namespace vart {
namespace cpu {

template <typename DType>
class StridedSliceFix : public StridedSlice<DType> {
 public:
 public:
  StridedSliceFix(const xir::Subgraph* subg, const xir::Op* op,
                  IMapTBs_t inputs, CPUTBPtr_t output);
  ~StridedSliceFix() = default;

  virtual void print_param() override final;
  virtual void calculate(std::uint32_t start_index, std::uint32_t end_index) override final;

 protected:
  using StridedSlice<DType>::fmap_i_;
  using StridedSlice<DType>::fmap_o_;
  using StridedSlice<DType>::data_in_;
  using StridedSlice<DType>::data_out_;
  using StridedSlice<DType>::begin_;
  using StridedSlice<DType>::end_;
  using StridedSlice<DType>::strides_;

 private:
  int32_t fp_input_;
  int32_t fp_output_;
  int32_t shift_;
};

}  // namespace cpu
}  // namespace vart
