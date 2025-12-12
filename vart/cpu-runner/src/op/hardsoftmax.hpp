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
class HardSoftmax : public CPUOPBase {
 public:
  enum InputTensorType {
    INPUT,
  };
  // input tensor name
  const static vector<string> ITName;

 public:
  HardSoftmax(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
              CPUTBPtr_t output);
  ~HardSoftmax() = default;

  virtual void run() override final;

  virtual void print_param() override final;
  virtual void check_param() override final;

  virtual void read() override final;

  virtual uint64_t get_workload() override final;

 private:
  void softmax();
  void bf16_to_int8(int num, DType* rlt);

 private:
  vector<int> fmap_i_;
  vector<int> fmap_o_;

  int axis_;
  int input_fix_point_;
  int output_fix_point_;
  bool is_bf16_in_;
  bool is_bf16_out_;
  // caculate buffer
  DType* img_{nullptr};
  DType* rlt_{nullptr};
};

}  // namespace cpu
}  // namespace vart
