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

#include "conv_base.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename WType>
class TConv2d : public ConvBase<DType, WType> {
public:
  TConv2d(const xir::Subgraph* subg, const xir::Op* op,
      IMapTBs_t inputs, CPUTBPtr_t output);
  virtual ~TConv2d() = default;

  virtual void run() override;

  virtual void check_param() override;

protected:
  using ConvBase<DType, WType>::raw_fmap_i_;
  using ConvBase<DType, WType>::fmap_i_;
  using ConvBase<DType, WType>::raw_fmap_w_;
  using ConvBase<DType, WType>::fmap_w_;
  using ConvBase<DType, WType>::fmap_b_;
  using ConvBase<DType, WType>::fmap_o_;
  using ConvBase<DType, WType>::pad_;
  using ConvBase<DType, WType>::kernel_;
  using ConvBase<DType, WType>::stride_;
  using ConvBase<DType, WType>::dilation_;
  using ConvBase<DType, WType>::has_bias_;
};

} // namespace cpu
} // namespace vart

