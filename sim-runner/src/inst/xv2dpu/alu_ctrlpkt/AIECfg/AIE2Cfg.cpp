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

#include "AIE2Cfg.hpp"
#include <assert.h>

namespace cosim {
namespace xv2dpu {
namespace alu {
AIE2Cfg::AIE2Cfg(int core_num, int src_h, int src_w, int ic, int oc,
                int kernel_h, int kernel_w, int stride_h, int stride_w,
                int pad_top, int pad_bottom, int pad_left, int pad_right,
                int owg_tile, int ocg_tile, int icg_tile, int ohg_tile,
                int SHIFT_CUT, int SHIFT_BIAS, int HSIGMOID_IN,
                int HSIGMOID_OUT, int HSWISH_OUT, int ACT_TYPE,
                MISC_CASE EXEC_TYPE) {
 //   single kernel granularity setting
 this->ICp = 16;
 this->ICP = 16;
 this->OCp = 16;
 this->OCP = 16;
 this->OWp = 8;
 this->OWP = 8;
 this->OHp = 1;
 this->OHP = 1;
 this->OCPG = 1;
 this->OHPG = 1;

 //   graph kernel/mt setting
 this->CORE_NUM = core_num;
 this->CASC_NUM = 1;

 this->IMG_MT_GRP = 1;
 this->IMG_MT_NUM = 1;
 this->IMG_LOAD_PP = 1;
 this->WGT_MT_GRP = 1;
 this->WGT_MT_NUM = 1;
 this->WGT_LOAD_PP = 1;
 this->OFM_SAVE_PP = 1;

 // data setting
 this->src_h = src_h;
 this->src_w = src_w;
 this->ic = ic;
 this->oc = oc;
 this->kernel_h = kernel_h;
 this->kernel_w = kernel_w;
 this->stride_h = stride_h;
 this->stride_w = stride_w;
 this->owg_tile = owg_tile;
 this->ocg_tile = ocg_tile;
 this->icg_tile = icg_tile;
 this->ohg_tile = ohg_tile;
 this->ocg = oc / OCp;
 this->icg = ic / ICp;
 this->SHIFT_CUT = SHIFT_CUT;
 this->SHIFT_BIAS = SHIFT_BIAS;
 this->HSIGMOID_IN = HSIGMOID_IN;
 this->HSIGMOID_OUT = HSIGMOID_OUT;
 this->HSWISH_OUT = HSWISH_OUT;
 this->ACT_TYPE = ACT_TYPE;
 this->dilation_h = 1;
 this->dilation_w = 1;
 this->pad_left = pad_left;
 this->pad_right = pad_right;
 this->pad_top = pad_top;
 this->pad_bottom = pad_bottom;

 this->dst_h =
     (src_h + pad_top + pad_bottom - (dilation_h * (kernel_h - 1) + 1)) /
         stride_h +
     1;
 this->dst_w =
     (src_w + pad_left + pad_right - (dilation_w * (kernel_w - 1) + 1)) /
         stride_w +
     1;

 // this->src_size = src_h * src_w * ic;
 // this->output_size = dst_h * dst_w * oc;

 if ((ocg_tile * owg_tile * ohg_tile >= 3) &&
     (icg_tile * kernel_w * kernel_h >= 4)) {
   this->EXEC_TYPE = MISC_CASE::CASE_MACC;
 } else {
   this->EXEC_TYPE = MISC_CASE::CASE_AVGP;
 }
}

void AIE2Cfg::Check() {
 assert((src_h > 0 && src_w > 0) && "src_w>0, src_h>0");
 assert((kernel_h > 0 && kernel_w > 0) && "kernel_w>0, kernel_h>0");
 assert((stride_h > 0 && stride_w > 0) && "stride_w>0, stride_h>0");
 assert((ic > 0 && oc > 0) && "ic need >0, oc need >0");
 assert(ic >= 32 && "ic must >= 32");
 assert((src_h >= kernel_h && src_w >= kernel_w) &&
        "src_h>kernel_h, src_w>kernel_w");
 assert((oc >= ocg_tile * 16) && "oc need >= ocg_tile*16");
 assert((ACT_TYPE == 0 || ACT_TYPE == 1) && "ACT_TYPE, only (0:None, 1:ReLU");

 assert(dst_h % (ohg_tile * OHP) == 0);
 assert(dst_w % (owg_tile * OWP) == 0);  // when disable output w offset
 assert(ic % (icg_tile * ICP) == 0);
 assert((dst_w % 8 == 0) && "dst_w % 8 == 0");
 assert(((src_h + pad_top + pad_bottom - kernel_h) >= 0) &&
        "(src_h + pad_top + pad_bottom -kernel_h) >= 0");
 assert(((src_w + pad_left + pad_right - kernel_w) >= 0) &&
        "(src_w + pad_left + pad_right -kernel_w) >= 0");
}

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
