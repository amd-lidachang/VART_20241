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

#include "cpu_tb_factory.hpp"

namespace vart {
namespace cpu {

CPUTBPtr_t CPUTBFactory::create_or_get(const xir::Op* op) {
  std::lock_guard<std::recursive_mutex> lock(mtx_);

  // if existed, return directly
  auto* lookup_ptr = get_by_op(op);
  if (lookup_ptr != nullptr) return lookup_ptr;

  auto* tensor = op->get_output_tensor();

  // create new cpu tensor buffer
  // and update three related data structure
  auto tb = CPUTensorBuffer::make(op, tensor);
  lookup_ptr = tb.get();

  tbs_.push_back(std::move(tb));
  op_map_[op] = lookup_ptr;
  tensor_map_[tensor] = lookup_ptr;

  if (VART_DEBUG) {
    UNI_LOG_DEBUG_INFO << endl;
    UNI_LOG_DEBUG_INFO << "Created op " << op->get_name()
                       << "\'s output CPUTensorBuffer:" << endl;
    PRINT_DIVIDING_LINE();
    print_cpu_tb(lookup_ptr);
  }

  return lookup_ptr;
}

}  // namespace cpu
}  // namespace vart
