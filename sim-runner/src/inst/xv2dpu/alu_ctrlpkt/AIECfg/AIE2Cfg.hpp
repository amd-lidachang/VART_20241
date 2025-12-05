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

#include "params.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

class AIE2Cfg {
 public:
  /**********************************
   *   single kernel granularity setting
   *   This is set according to kernel code.
   **********************************/
  // ICp: one core parallel
  // ICP: cascade cores parallel
  // OCPG: cascade cores parallel groups
  int ICp;
  int ICP;
  int OCp;
  int OCP;
  int OWp;
  int OWP;
  int OHp;
  int OHP;
  int OCPG;
  int OHPG;

  /**********************************
   *  Hardware, including
   *  core nums and memtile in/out
   **********************************/
  int CORE_NUM;
  int CASC_NUM;

  int IMG_MT_GRP;
  int IMG_MT_NUM;
  int IMG_LOAD_PP;
  int WGT_MT_GRP;
  int WGT_MT_NUM;
  int WGT_LOAD_PP;
  int OFM_SAVE_PP;

  /**********************************
   *   input ifm/wgt params setting
   *   This is  according to network layer params.
   **********************************/
  int win_step;  // default 32;

  int src_h;
  int src_w;
  int ic;
  int oc;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;

  int ocg;
  int icg;
  int owg_tile;
  int ocg_tile;
  int icg_tile;
  int ohg_tile;

  int pad_left;
  int pad_top;
  int pad_right;
  int pad_bottom;
  int dilation_h;
  int dilation_w;
  int dst_h;
  int dst_w;
  // int src_size; // this is useless because in BaseOps, ifm_size if redefined
  // int output_size;

  int SHIFT_CUT;
  int SHIFT_BIAS;
  int HSIGMOID_IN;
  int HSIGMOID_OUT;
  int HSWISH_OUT;
  int ACT_TYPE;
  MISC_CASE EXEC_TYPE;

 public:
  // init
  AIE2Cfg(int core_num, int src_h, int src_w, int ic, int oc, int kernel_h,
          int kernel_w, int stride_h, int stride_w, int pad_top, int pad_bottom,
          int pad_left, int pad_right, int owg_tile, int ocg_tile, int icg_tile,
          int ohg_tile, int SHIFT_CUT, int SHIFT_BIAS, int HSIGMOID_IN,
          int HSIGMOID_OUT, int HSWISH_OUT, int ACT_TYPE, MISC_CASE EXEC_TYPE);

  // check if numbers are legal
  virtual void Check();
};
}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
