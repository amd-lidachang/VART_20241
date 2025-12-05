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
#include "OpDwconv.hpp"
#include "math_op.hpp"

namespace cosim {
namespace xv2dpu {
namespace alu {

class OpAvgp : public OpDwconv {
 public:
  OpAvgp(AIE2Cfg AIE_) : OpDwconv(AIE_) {
    this->wgt_size = AIE.ICp;
    this->bias_size = AIE.ICp;

    // Note: here x2 because data is copied in DM
    this->wgt_wnd_byte_size = AIE.ICp;
    this->bias_wnd_byte_size = AIE.ICp;

    // this part indicate that only use 1 mt group
    this->weights_mt_grp_byte_size = AIE.ICp;
    this->bias_mt_grp_byte_size = AIE.ICp;

    cout << "macc" << endl;
    cout << "img wnd size: " << img_wnd_byte_size << endl;
    cout << "wgt_wnd size: " << wgt_wnd_byte_size << endl;
    cout << "bias_wnd size: " << bias_wnd_byte_size << endl;
    cout << "ofm_wnd size: " << ofm_wnd_byte_size << endl;
    cout << "ow offset: " << OW_offset << endl;
  }

  virtual void math_op(const vector<int8_t>& img, const vector<int8_t>& wgt,
                       const vector<int8_t>& bias, vector<int8_t>& rlt) {
    return avgpool_op(AIE, img.data(), wgt.data(), bias.data(), rlt.data());
  }

  virtual void set_layer_params() {
    OpDwconv::set_layer_params();
    int32_t incS0 = 0;
    int32_t bias_offset = 0;
    layer_param[0] = incS0;
    layer_param[27] = bias_offset;
  }
};

}  // namespace alu
}  // namespace xv2dpu
}  // namespace cosim
