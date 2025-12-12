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

#include "hsigmoid_fix.hpp"

namespace vart {
namespace cpu {

template <typename DType>
void UnaryHSigmoidFix<DType>::calculate() {
  for (auto i = 0; i < fmap_o_.num(); i++) {
    double tmp =
      min(
        pow(2, 32),
        max(0.0,
            (data_in_ptr_[i] * 2731.0f + 3.0f * 2731 * pow(2, hsigmoid_in_)))) /
      pow(2, shift_hsigmoid_);
    data_out_ptr_[i] =
      DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS(UnaryHSigmoidFix);
REG_OP_INSTANCE_FUNC("hard-sigmoid-fix", UnaryHSigmoidFix);

}  // namespace cpu
}  // namespace vart
