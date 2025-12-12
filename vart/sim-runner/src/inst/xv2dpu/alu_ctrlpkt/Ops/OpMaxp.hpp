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
#include "math_op.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

class OpMaxp : public OpDwconv {
 public:
  OpMaxp(AIE2Cfg AIE_) : OpDwconv(AIE_) {
    this->wgt_size = 0;
    this->bias_size = 0;

    // Note: here x2 because data is copied in DM
    this->wgt_wnd_byte_size = 0;
    this->bias_wnd_byte_size = 0;

    // this part indicate that only use 1 mt group
    this->weights_mt_grp_byte_size = 0;
    this->bias_mt_grp_byte_size = 0;

    cout << "macc" << endl;
    cout << "IWin1Iter: " << IWin1Iter << endl;
    cout << "img wnd size: " << img_wnd_byte_size << endl;
    cout << "wgt_wnd size: " << wgt_wnd_byte_size << endl;
    cout << "bias_wnd size: " << bias_wnd_byte_size << endl;
    cout << "ofm_wnd size: " << ofm_wnd_byte_size << endl;
    cout << "ow offset: " << OW_offset << endl;
  }

  virtual void math_op(const vector<int8_t>& img, const vector<int8_t>& wgt,
                       const vector<int8_t>& bias, vector<int8_t>& rlt) {
    return maxpool_op(AIE, img.data(), rlt.data());
  }

  virtual void set_layer_params() {
    OpDwconv::set_layer_params();
    int32_t incS0 = 32;
    int32_t bias_offset = 0;
    // this part need change to maxp mode
    int32_t exec_type = 4;
    layer_param[0] = incS0;
    layer_param[27] = bias_offset;
    layer_param[29] = exec_type;
    cout << "============= numAl1:" << layer_param[2] << endl;

    cout << "IWin1Iter: " << IWin1Iter << endl;
    layer_param[33] = AIE.pad_top;
    layer_param[34] = AIE.pad_bottom;
    layer_param[35] = AIE.pad_left;
    layer_param[36] = AIE.pad_right;
    layer_param[37] = 4;
    layer_param[38] = IWin1Iter;
    layer_param[39] = IHin1Iter;
    layer_param[40] = AIE.stride_h;
    layer_param[41] = AIE.ocg_tile;
    cout << "Strride: " << AIE.stride_h << endl;
    cout << "ocgtile: " << AIE.ocg_tile << endl;
  }
};

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
