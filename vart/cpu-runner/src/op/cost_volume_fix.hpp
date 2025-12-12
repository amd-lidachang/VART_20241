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

#include "cost_volume.hpp"

namespace vart {
namespace cpu {

template <typename DType>
class CostVolumeFix : public CostVolume<DType> {
 public:
  CostVolumeFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                CPUTBPtr_t output);
  ~CostVolumeFix() = default;

  virtual void print_param() override;

  virtual void cost_volume(std::int32_t start_index,
                           std::int32_t end_index) override;

 private:
  vector<int> fp_inputs_;
  int fp_output_;

  vector<int> shift_read_;
  using CostVolume<DType>::input_num_;
  using CostVolume<DType>::fmap_i_;
  using CostVolume<DType>::fmap_o_;
  using CostVolume<DType>::stride_;
  using CostVolume<DType>::data_in_;
  using CostVolume<DType>::data_out_;
};

}  // namespace cpu
}  // namespace vart
