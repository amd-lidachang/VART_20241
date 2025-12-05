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

#include "pixel_shuffle.hpp"

namespace vart {
namespace cpu {

template <typename DType>
PixelShuffle<DType>::PixelShuffle(const xir::Subgraph* subg, const xir::Op* op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  input_shape_ =
      op->get_input_ops("input")[0]->get_output_tensor()->get_shape();
  output_shape_ = op->get_output_tensor()->get_shape();
  upscale_ = xir_op_->get_attr<bool>("upscale");
  scale_ = xir_op_->get_attr<int>("scale");

  print_param();
}

template <typename DType>
void PixelShuffle<DType>::run() {
  read();
  shuffle();
}

template <typename DType>
void PixelShuffle<DType>::print_param() {
  UNI_LOG_DEBUG_INFO << "upscale: " << Bool2Str(upscale_) << endl;
  UNI_LOG_DEBUG_INFO << "scale: " << scale_ << endl;
}

template <typename DType>
void PixelShuffle<DType>::check_param() {}

template <typename DType>
void PixelShuffle<DType>::read() {
  auto* cputb = inputs_.at("input").at(0);
  data_in_ = GET_CPUTB_DType_PTR(DType, cputb);
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
uint64_t PixelShuffle<DType>::get_workload() {
  return 0;
}

template <typename DType>
void PixelShuffle<DType>::shuffle() {
  if (upscale_) {
    int idx = 0;
    for (auto b = 0; b < output_shape_[0]; b++)
      for (auto h = 0; h < output_shape_[1]; h++)
        for (auto w = 0; w < output_shape_[2]; w++)
          for (auto c = 0; c < output_shape_[3]; c++) {
            auto i_index =
                b * input_shape_[1] * input_shape_[2] * input_shape_[3] +
                h / scale_ * input_shape_[2] * input_shape_[3] +
                w / scale_ * input_shape_[3] + c * scale_ * scale_ +
                w % scale_ + h % scale_ * scale_;
            data_out_[idx++] = data_in_[i_index];
          }
  } else {
    int idx = 0;
    for (auto b = 0; b < input_shape_[0]; b++)
      for (auto h = 0; h < input_shape_[1]; h++)
        for (auto w = 0; w < input_shape_[2]; w++)
          for (auto c = 0; c < input_shape_[3]; c++) {
            auto i_index =
                b * output_shape_[1] * output_shape_[2] * output_shape_[3] +
                h / scale_ * output_shape_[2] * output_shape_[3] +
                w / scale_ * output_shape_[3] + c * scale_ * scale_ +
                w % scale_ + h % scale_ * scale_;
            data_out_[i_index] = data_in_[idx++];
          }
  }
}

INSTANTIATE_TPCLASS(PixelShuffle);

REG_OP_INSTANCE_FUNC("pixel-shuffle", PixelShuffle);

}  // namespace cpu
}  // namespace vart
