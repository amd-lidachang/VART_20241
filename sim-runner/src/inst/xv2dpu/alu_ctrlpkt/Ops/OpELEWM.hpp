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
#include "BaseDMA.hpp"
#include "BaseOps.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

class OpELEWM : public OpELEWA {
 public:
  OpELEWM(AIE2Cfg AIE_) : OpELEWA(AIE_) {}

  virtual void math_op(const vector<int8_t>& img, const vector<int8_t>& wgt,
                       const vector<int8_t>& bias, vector<int8_t>& rlt) {
    return elew_mul_op(AIE, img.data(), rlt.data());
  }

  virtual void set_layer_params() {
    OpELEWS::set_layer_params();

    int32_t exec_type = 3;
    layer_param[29] = exec_type;
  }
};

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
