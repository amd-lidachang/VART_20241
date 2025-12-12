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

template <typename DType>
class Add : public CPUOPBase {
 public:
  Add(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
      CPUTBPtr_t output);
  virtual ~Add() = default;

 public:
  virtual void align_dim();
  virtual void print_param() override;
  virtual void read() override final;
  virtual void run() override;
  virtual void add_normal();
  virtual void add_thread();
  virtual void add(std::uint32_t start_index, std::uint32_t end_index);

 protected:
  vector<Dimension> fmap_i_;
  Dimension fmap_o_;

  // i/o buffer
  vector<DType*> data_in_ptr_;
  DType* data_out_ptr_{nullptr};

  std::uint32_t THREAD_NUM;
  std::uint32_t THREAD_WORKLOAD;

 private:
  bool shift;
  vector<std::int32_t> fp_inputs_;
  vector<std::int32_t> shift_factor_;
};

template <typename DType>
class AddFix : public Add<DType> {
 public:
  AddFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
         CPUTBPtr_t output);
  virtual ~AddFix() = default;

  virtual void add(std::uint32_t start_index, std::uint32_t end_index) override;

 protected:
  using Add<DType>::fmap_i_;
  using Add<DType>::fmap_o_;
  using Add<DType>::data_in_ptr_;
  using Add<DType>::data_out_ptr_;

 private:
  std::vector<int> fp_inputs_;
  int fp_output_;
  std::vector<int> shift_read_;
  int shift_write_;
  int hsigmoid_in_;
  int shift_hsigmoid_;

  std::string act_type_;
};

}  // namespace cpu
}  // namespace vart
