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
#include "OpELEWS.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

class OpELEWA : public OpELEWS {
 public:
  OpELEWA(AIE2Cfg AIE_) : OpELEWS(AIE_) {
    this->ifm_size = AIE.ic * AIE.src_w * AIE.src_h * 2;

    this->img_wnd_byte_size = ICin1Iter * IWin1Iter * IHin1Iter * 2;

    this->img_mt_grp_byte_size = AIE.src_h * AIE.src_w * AIE.ic * 2;

    cout << "macc" << endl;
    cout << "img wnd size: " << img_wnd_byte_size << endl;
    cout << "wgt_wnd size: " << wgt_wnd_byte_size << endl;
    cout << "bias_wnd size: " << bias_wnd_byte_size << endl;
    cout << "ofm_wnd size: " << ofm_wnd_byte_size << endl;
    cout << "ow offset: " << OW_offset << endl;
  }

  virtual void math_op(const vector<int8_t>& img, const vector<int8_t>& wgt,
                       const vector<int8_t>& bias, vector<int8_t>& rlt) {
    return elew_add_op(AIE, img.data(), rlt.data());
  }

  virtual string ifm_core_set_window() {
    return ifm_core_set_window_elew_double(img_wnd_byte_size, IWin1Iter,
                                           IHin1Iter, AIE.icg_tile);
  }

  string set_mt2co_ifm(int idx_mt_grp, int idx_mt_num, uint32_t img_bytes_size,
                       uint32_t mt_img_ifm_addr, int count, int pad_left,
                       int pad_top, int pad_right, int pad_bottom,
                       int IWin1Iter, int IHin1Iter, int src_h, int src_w,
                       int stride_w, int ICG, int ICG_tile, int Lock_Acq_ID,
                       int Lock_Rel_ID, bool BD_casc_fir, bool BD_casc_last,
                       int isPing) {
    return setELEWADDImgMT2CO(idx_mt_grp, idx_mt_num, img_bytes_size,
                              mt_img_ifm_addr, count, pad_left, pad_top,
                              pad_right, pad_bottom, IWin1Iter, IHin1Iter,
                              src_h, src_w, stride_w, ICG, Lock_Acq_ID,
                              Lock_Rel_ID, BD_casc_fir, BD_casc_last, isPing);
  }

  virtual void set_layer_params() {
    OpELEWS::set_layer_params();

    int32_t exec_type = 2;
    layer_param[29] = exec_type;
  }
};

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
