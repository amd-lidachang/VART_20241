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
class Resize : public CPUOPBase {
 public:
  Resize(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
         CPUTBPtr_t output);
  ~Resize() = default;

  virtual void print_param() override;

  virtual void run() override;
  virtual void read() override final;
  void resize();

 protected:
  std::string mode_;
  bool align_corners_;
  bool half_pixel_centers_;

  FMap_t fmap_i_;
  FMap_t fmap_o_;

  DType* data_in_ptr_{nullptr};
  std::vector<float> output_f_;
  DType* data_out_ptr_{nullptr};
};

}  // namespace cpu
}  // namespace vart

