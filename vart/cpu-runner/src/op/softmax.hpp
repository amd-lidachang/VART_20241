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


template<typename DType>
class Softmax : public CPUOPBase {
public:
  Softmax(const xir::Subgraph* subg, const xir::Op* op,
      IMapTBs_t inputs, CPUTBPtr_t output);
  ~Softmax() = default;

  virtual void run() override;

  virtual void print_param() override;
  virtual void check_param() override final;

  virtual void read() override final;

  virtual uint64_t get_workload() override final;

protected:
  // an inplace softmax on output_f_, call softmax_normal and softmax_thread
  void softmax();
  // serial implementation of softmax
  void softmax_normal();
  // parallel implementation of softmax
  void softmax_thread();
  // do a single softmax on innermost dimension
  void softmax_one_contiguous(int loop_iter);
  // do a single softmax on non-innermost dimension
  void softmax_one_discrete(int loop_iter);
  uint32_t THREAD_NUM;
  uint32_t THREAD_WORKLOAD;

  uint32_t axis_stride_;
  uint32_t loop_times_;

protected:
  Dimension fmap_i_;
  Dimension fmap_o_;

  std::vector<float> output_f_;
  int axis_;

  // caculate buffer
  DType* data_in_ptr_{nullptr};
  DType* data_out_ptr_{nullptr};
};

} // namespace cpu
} // namespace vart


