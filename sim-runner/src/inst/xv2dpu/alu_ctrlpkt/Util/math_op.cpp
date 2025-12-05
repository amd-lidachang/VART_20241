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

#include "math_op.hpp"

#include "AIE2Cfg.hpp"
namespace cosim::xv2dpu::alu {

double dr(double x) {
  if (x - std::floor(x) >= 0.5)
    return double(std::min(127.0, std::max(-128.0, double(std::ceil(x)))));
  else
    return double(std::min(127.0, std::max(-128.0, double(std::floor(x)))));
}
double activation(double tmp, const AIE2Cfg& A) {
  const int SHIFT_CUT = A.SHIFT_CUT;
  const int HSIGMOID_IN = A.HSIGMOID_IN;
  const int HSIGMOID_OUT = A.HSIGMOID_OUT;
  const int HSWISH_OUT = A.HSWISH_OUT;
  const int ACT_TYPE = A.ACT_TYPE;

  if (ACT_TYPE == 0) {
    tmp /= pow(2, SHIFT_CUT);
  } else if (ACT_TYPE == 1) {
    tmp = (tmp < 0) ? 0 : tmp;
    tmp /= pow(2, SHIFT_CUT);
  } else if (ACT_TYPE == 2) {
    tmp = (tmp < 0) ? 0 : tmp;
    tmp /= pow(2, SHIFT_CUT);
    tmp = (tmp > 6) ? 6 : tmp;
  } else if (ACT_TYPE == 3) {
    tmp = (tmp < 0) ? (tmp * 26. / 256.) : tmp;
    tmp /= pow(2, SHIFT_CUT + 1);
  } else if (ACT_TYPE == 5) {
    tmp /= pow(2, SHIFT_CUT + 1);
    // tmp = dr(tmp);
    tmp = DRound<int8_t>(tmp);
    tmp =
        std::min(pow(2, 32),
                 std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, HSIGMOID_IN)))) *
        pow(2, HSIGMOID_OUT - HSIGMOID_IN - 14);
  } else if (ACT_TYPE == 6) {
    auto x = dr(tmp / pow(2, SHIFT_CUT + 1));
    auto hsigmoid_x = dr(
        std::min(pow(2, 32),
                 std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, HSIGMOID_IN)))) *
        pow(2, HSIGMOID_OUT - HSIGMOID_IN - 14));
    tmp = x * hsigmoid_x * pow(2, HSWISH_OUT - HSIGMOID_IN - HSIGMOID_OUT);
  }

  return tmp;
}
}  // namespace cosim