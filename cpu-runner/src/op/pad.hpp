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

namespace vart {
namespace cpu {

template <typename DType>
class Pad : public CPUOPBase {
 public:
  Pad(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
      CPUTBPtr_t output);
  ~Pad() = default;

  virtual void read() override final;
  virtual void run() override final;

 private:
  void pad();

 private:
  std::vector<std::int32_t> i_shape_;
  std::vector<std::int32_t> o_shape_;
  std::vector<std::int32_t> i_prod_;
  std::vector<std::int32_t> o_prod_;
  std::string mode_;
  std::vector<char> val_;
  std::vector<std::int32_t> paddings_;
  DType* data_in_ptr_{nullptr};
  DType* data_out_ptr_{nullptr};

};

}  // namespace cpu
}  // namespace vart
