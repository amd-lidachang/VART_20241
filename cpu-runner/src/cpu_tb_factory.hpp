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

#include "cpu_tensor_buffer.hpp"
#include <unordered_map>

namespace vart {
namespace cpu {

class CPUTBFactory {
 private:
  explicit CPUTBFactory() = default;
  VART_BIG_THREE_LAW(CPUTBFactory);

 public:
  static CPUTBFactory& Instance() {
    static CPUTBFactory fac;
    return fac;
  }

  CPUTBPtr_t create_or_get(const xir::Op* op);

  inline CPUTBPtr_t get_by_op(const xir::Op* op) {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    if (op_map_.find(op) != op_map_.end()) return op_map_.at(op);
    return nullptr;
  }
  inline CPUTBPtr_t get_by_tensor(const xir::Tensor* tensor) {
    std::lock_guard<std::recursive_mutex> lock(mtx_);
    if (tensor_map_.find(tensor) != tensor_map_.end())
      return tensor_map_.at(tensor);
    return nullptr;
  }

 private:
  // using mutex to make sure this class is thread-safe,
  // and in create_or_get func, it will call other routines,
  // so should use a recursive mutex to prevent dead-lock
  std::recursive_mutex mtx_;

  // use list to manage created TBs
  list<unique_ptr<CPUTensorBuffer>> tbs_;

  // key: op, value is tbs_ related element's raw pointer
  std::unordered_map<const xir::Op*, CPUTensorBuffer*> op_map_;
  // key: tensor, value is tbs_ related element's raw pointer
  std::unordered_map<const xir::Tensor*, CPUTensorBuffer*> tensor_map_;
};

}  // namespace cpu
}  // namespace vart
