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

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
class BinaryBase : public CPUOPBase {
 public:
  // binary operation type and name
  enum BinaryType { ADD, SUB, MUL, DIV, Unknown };
  const static vector<string> BinaryTypeName;

  // input tensor enum and name
  enum InputTensorType {
    INPUT,
  };
  const static vector<string> ITName;

  uint32_t THREAD_NUM;
  uint32_t THREAD_WORKLOAD;

 public:
  BinaryBase(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
             CPUTBPtr_t output);
  virtual ~BinaryBase() = default;

  virtual void run() override final;

  virtual void print_param() override;
  virtual void check_param() override final;

  virtual void read() override final;

  virtual uint64_t get_workload() override final;

 protected:
  void calculate_normal();
  void calculate_thread();
  virtual void calculate(std::uint32_t start_index, std::uint32_t end_index,
                         bool no_broadcast) {}

 protected:
  vector<Dimension> fmap_i_;
  Dimension fmap_o_;

  int binary_type_{Unknown};
  bool no_broadcast_ = false;

  // i/o buffer
  DTypeIn0* data_ina_ptr_{nullptr};
  DTypeIn1* data_inb_ptr_{nullptr};
  DTypeOut* data_out_ptr_{nullptr};
};

}  // namespace cpu
}  // namespace vart

