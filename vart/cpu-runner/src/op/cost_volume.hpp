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

#include "align_buf_mgr.hpp"
#include "cpu_op_base.hpp"

namespace vart {
namespace cpu {

template <typename DType>
class CostVolume : public CPUOPBase {
 public:
  enum InputTensorType {
    INPUT,
  };
  // input tensor name
  const static vector<string> ITName;

 public:
  CostVolume(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
             CPUTBPtr_t output);
  virtual ~CostVolume() = default;

  virtual void run() override;

  virtual void print_param() override;
  virtual void check_param() override;

  virtual void read() override final;

  virtual uint64_t get_workload() override final;

 protected:
  virtual void cost_volume(std::int32_t start_index, std::int32_t end_index);

  vector<FMap_t> fmap_i_;
  FMap_t fmap_o_;

  std::int32_t stride_;
  int input_num_;

  // caculate buffer
  vector<DType*> data_in_;
  DType* data_out_{nullptr};

 private:
  void cost_volume_normal();
  void cost_volume_thread();
  uint32_t THREAD_NUM;
  uint32_t THREAD_WORKLOAD;
};

}  // namespace cpu
}  // namespace vart
