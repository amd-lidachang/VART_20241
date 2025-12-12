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
#include "AIE2Cfg.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

class MISC2Cfg : public AIE2Cfg {
 public:
  MISC2Cfg(int core_num, int src_h, int src_w, int ic, int oc, int kernel_h,
           int kernel_w, int stride_h, int stride_w, int pad_top,
           int pad_bottom, int pad_left, int pad_right, int owg_tile,
           int ocg_tile, int icg_tile, int ohg_tile, int SHIFT_CUT,
           int SHIFT_BIAS, int HSIGMOID_IN, int HSIGMOID_OUT, int HSWISH_OUT,
           int ACT_TYPE, MISC_CASE EXEC_TYPE)
      : AIE2Cfg(core_num, src_h, src_w, ic, oc, kernel_h, kernel_w, stride_h,
                stride_w, pad_top, pad_bottom, pad_left, pad_right, owg_tile,
                ocg_tile, icg_tile, ohg_tile, SHIFT_CUT, SHIFT_BIAS,
                HSIGMOID_IN, HSIGMOID_OUT, HSWISH_OUT, ACT_TYPE, EXEC_TYPE) {
    this->EXEC_TYPE = EXEC_TYPE;
    // these case force kernel to be 1
    if (EXEC_TYPE == MISC_CASE::CASE_TEST ||
        EXEC_TYPE == MISC_CASE::CASE_ELEWMUL ||
        EXEC_TYPE == MISC_CASE::CASE_ELEWADD ||
        EXEC_TYPE == MISC_CASE::CASE_ELEWS) {
      this->kernel_w = 1;
      this->kernel_h = 1;
      this->dst_h = src_h;
      this->dst_w = src_w;
    }

    // this->src_size     = src_h * src_w * ic;
    // this->output_size  = dst_h * dst_w * oc;
  }

  void Check() {
    assert((src_h > 0 && src_w > 0) && "src_w>0, src_h>0");
    assert((kernel_h > 0 && kernel_w > 0) && "kernel_w>0, kernel_h>0");
    assert((stride_h > 0 && stride_w > 0) && "stride_w>0, stride_h>0");
    assert((ic > 0 && oc > 0) && "ic need >0, oc need >0");
    assert((ic == oc) && "ic == oc");

    // here for misc ic is 16 at least
    assert(ic >= 16 && "ic must >= 16");

    // assert((src_h >= kernel_h && src_w >= kernel_w) && "src_h>kernel_h,
    // src_w>kernel_w");
    assert((oc >= ocg_tile * 16) && "oc need >= ocg_tile*16");
    assert((ACT_TYPE == 0 || ACT_TYPE == 1 || ACT_TYPE == 3 || ACT_TYPE == 4 ||
            ACT_TYPE == 5 || ACT_TYPE == 6) &&
           "ACT_TYPE, only (0:None, 1:ReLU, 3:leakyRelu, 4:Relu6, 5:Hsigmoid, 6:Hswish");

    cout << "====" << endl;
    cout << OHP << endl;
    cout << dst_h << endl;

    // assert(dst_w % (owg_tile * OWP) == 0);  // when disable output w offset
    assert(ic % (icg_tile * ICP) == 0);
    //alu result 32 bit -> 8bit
    assert(std::abs(SHIFT_CUT & 0x3f)<=32);
    // assert((dst_w % 8 == 0) && "dst_w % 8 == 0");
    assert(((src_h + pad_top + pad_bottom - kernel_h) >= 0) &&
           "(src_h + pad_top + pad_bottom -kernel_h) >= 0");
    assert(((src_w + pad_left + pad_right - kernel_w) >= 0) &&
           "(src_w + pad_left + pad_right -kernel_w) >= 0");
  }
};

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
