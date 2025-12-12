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

#include "correlation1d_elemwise.hpp"

namespace vart {
namespace cpu {

template <typename DType>
class Correlation1dElemwiseFix : public Correlation1dElemwise<DType> {
 public:
  Correlation1dElemwiseFix(const xir::Subgraph* subg, const xir::Op* op,
                           IMapTBs_t inputs, CPUTBPtr_t output);
  virtual ~Correlation1dElemwiseFix() = default;

  virtual void print_param() override;

 protected:
  virtual void eltwise(std::uint64_t start_index,
                       std::uint64_t end_index) override;

 private:
  using Correlation1dElemwise<DType>::fmap_i_;
  using Correlation1dElemwise<DType>::fmap_o_;
  using Correlation1dElemwise<DType>::pad_size_;
  using Correlation1dElemwise<DType>::data_in_;
  using Correlation1dElemwise<DType>::data_out_;

  vector<int> fp_inputs_;
  int fp_output_;
  int shift_write_;
};

}  // namespace cpu
}  // namespace vart
