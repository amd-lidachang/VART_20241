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

#include "tdwconv.hpp"

namespace vart {
namespace cpu {

// constructor and deconstructor
template <typename DType, typename WType>
TDWConv<DType, WType>::TDWConv(const xir::Subgraph* subg, const xir::Op* op,
                        IMapTBs_t inputs, CPUTBPtr_t output)
    : DWConvBase<DType, WType>(subg, op, inputs, output) {
  // get dilation info
  auto xir_dilation = CPUOPBase::xir_op_->get_attr<vector<int>>("stride");
  dilation_ = Dilation_t{xir_dilation[1], xir_dilation[0]};

  // get input tensor's fmap
  GET_INPUT_DIM4_FMAP(raw_fmap_i_, input);
  auto tmp_n = raw_fmap_i_.n;
  auto tmp_h = raw_fmap_i_.h * dilation_.h - (dilation_.h - 1);
  auto tmp_w = raw_fmap_i_.w * dilation_.w - (dilation_.w - 1);
  auto tmp_c = raw_fmap_i_.c;
  fmap_i_ = FMapAttr{tmp_n, tmp_h, tmp_w, tmp_c};

  auto kernel_dilation = CPUOPBase::xir_op_->get_attr<vector<int>>("dilation");
  kernel_.dilation_h = kernel_dilation[1];
  kernel_.dilation_w = kernel_dilation[0];
  
  int32_t tmp_kn = raw_fmap_w_.n;
  int32_t tmp_kh = raw_fmap_w_.h * kernel_.dilation_h - (kernel_.dilation_h - 1);
  int32_t tmp_kw = raw_fmap_w_.w * kernel_.dilation_w - (kernel_.dilation_w - 1);
  int32_t tmp_kc = raw_fmap_w_.c;
  fmap_w_ = FMapAttr{{tmp_kn, tmp_kh, tmp_kw, tmp_kc}};

  // get kernel info
  auto xir_kernel = CPUOPBase::xir_op_->get_attr<vector<int>>("kernel");
  kernel_ = Kernel_t{xir_kernel[1], xir_kernel[0], Dilation_t(kernel_.dilation_h, kernel_.dilation_w)};

  // get pad info
  pad_.l = kernel_.w - 1 - pad_.l;
  pad_.t = kernel_.h - 1 - pad_.t;
  pad_.r = kernel_.w - 1 - pad_.r;
  pad_.b = kernel_.h - 1 - pad_.b;

  fmap_i_.h += pad_.t + pad_.b;
  fmap_i_.w += pad_.l + pad_.r;
}

template <typename DType, typename WType>
void TDWConv<DType, WType>::run() {
  this->dwconv();
  this->bias();
}

template <typename DType, typename WType>
void TDWConv<DType, WType>::check_param() {
  DWConvBase<DType, WType>::check_param();
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(TDWConv);
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("transposed-depthwise-conv2d", TDWConv, "weights");

}  // namespace cpu
}  // namespace vart
