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

#include "hsigmoid.hpp"

namespace vart {
namespace cpu {

template <typename DType>
void UnaryHSigmoid<DType>::calculate() {
  for (auto i = 0; i < fmap_o_.num(); i++) {
    data_out_ptr_[i] = min(6.0, max(0.0, data_in_ptr_[i] + 3.0)) / 6.0f;
  }
}

INSTANTIATE_TPCLASS(UnaryHSigmoid);
REG_OP_INSTANCE_FUNC("hard-sigmoid", UnaryHSigmoid);

}  // namespace cpu
}  // namespace vart
