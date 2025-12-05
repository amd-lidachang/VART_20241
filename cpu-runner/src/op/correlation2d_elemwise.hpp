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
class Correlation2dElemwise : public CPUOPBase {
 public:
  Correlation2dElemwise(const xir::Subgraph* subg, const xir::Op* op,
                        IMapTBs_t inputs, CPUTBPtr_t output);
  virtual ~Correlation2dElemwise() = default;

  virtual void run() override final;

  virtual void print_param() override;
  virtual void check_param() override;

  virtual void read() override final;

  virtual uint64_t get_workload() override final;

 protected:
  virtual void eltwise(std::uint64_t start_index, std::uint64_t end_index);

 private:
  void corr_normal();
  void corr_thread();

 protected:
  uint32_t THREAD_NUM;
  uint32_t THREAD_WORKLOAD;

  std::vector<FMap_t> raw_fmap_i_;
  std::vector<FMap_t> fmap_i_;
  FMap_t fmap_o_;

  Pad_t raw_pad_;
  Pad_t pad_;

  std::int32_t pad_size_;
  std::vector<DType*> data_in_;
  DType* data_out_{nullptr};
};

}  // namespace cpu
}  // namespace vart
