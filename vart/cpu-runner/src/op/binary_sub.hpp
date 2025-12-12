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

#include "binary_base.hpp"

namespace vart {
namespace cpu {

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
class BinarySub : public BinaryBase<DTypeIn0, DTypeIn1, DTypeOut> {
 public:
  BinarySub(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
            CPUTBPtr_t output)
      : BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>(subg, op, inputs, output) {
    BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::binary_type_ =
        BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::SUB;
  }
  virtual ~BinarySub() = default;

 protected:
  virtual void calculate(std::uint32_t start_index, std::uint32_t end_index,
                         bool no_broadcast) override final;

 protected:
  using BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::fmap_i_;
  using BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::fmap_o_;
  using BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::data_ina_ptr_;
  using BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::data_inb_ptr_;
  using BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::data_out_ptr_;
};

}  // namespace cpu
}  // namespace vart

