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
class StridedSlice : public CPUOPBase {
 public:
  StridedSlice(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
               CPUTBPtr_t output);
  ~StridedSlice() = default;

  virtual void run() override;

  virtual void print_param() override;
  virtual void check_param() override;

  virtual void read() override;

  virtual uint64_t get_workload() override;

 protected:
  void strided_slice();
  void calculate_normal();
  void calculate_thread();
  virtual void calculate(std::uint32_t start_index, std::uint32_t end_index);

  uint32_t THREAD_NUM;
  uint32_t THREAD_WORKLOAD;

 protected:
  vector<int> fmap_i_;
  vector<int> fmap_o_;

  uint32_t fmap_o_num_;

  vector<int32_t> begin_origin_;
  vector<int32_t> end_origin_;
  vector<int32_t> strides_origin_;

  vector<int32_t> begin_;
  vector<int32_t> end_;
  vector<int32_t> strides_;
  int begin_mask_ = 0;
  int end_mask_ = 0;
  int ellipsis_mask_ = 0;
  int new_axis_mask_ = 0;
  int shrink_axis_mask_ = 0;

  // caculate buffer
  DType* data_in_{nullptr};
  DType* data_out_{nullptr};
};

}  // namespace cpu
}  // namespace vart
