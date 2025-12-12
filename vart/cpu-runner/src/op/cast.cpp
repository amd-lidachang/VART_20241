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

#include "cast.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename OType>
const vector<string> Cast<DType, OType>::ITName = {
  "input",
};

// constructor and deconstructor
template <typename DType, typename OType>
Cast<DType, OType>::Cast(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 6);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 6);
  data_type_in_ = op->get_input_tensors("input")[0]->get_data_type().type;
  data_type_out_ = op->get_output_tensor()->get_data_type().type;
}

template <typename DType, typename OType>
void Cast<DType, OType>::run() {
  // // do necessary check
  // print_param();
  // check_param();

  // // read data
  // read();

  cast();

  // // do save, debug...
  // save();
}

template <typename DType, typename OType>
void Cast<DType, OType>::print_param() {
  UNI_LOG_DEBUG_INFO << "fmap_i_ = " << Vec2Str(fmap_i_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "fmap_o_ = " << Vec2Str(fmap_o_, ", ") << endl;

}

template <typename DType, typename OType>
void Cast<DType, OType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() == 1, VART_SIZE_ERROR);

  // check fmap_out height
  UNI_LOG_CHECK(fmap_i_.size() == fmap_o_.size(), VART_SIZE_ERROR)
    << fmap_i_.size() << " != " << fmap_o_.size();
  for (auto i = 0U; i < fmap_i_.size(); i++) {
    UNI_LOG_CHECK(fmap_i_[i] == fmap_o_[i], VART_INVALID_VALUE)
      << ", " << fmap_i_[i] << " != " << fmap_o_[i];
  }
}

// read data from DDR
template <typename DType, typename OType>
void Cast<DType, OType>::read() {
  // read img data
  auto* cputb = inputs_.at(ITName[INPUT]).at(0);
  img_ = GET_CPUTB_DType_PTR(DType, cputb);

  // handle output buffer
  rlt_ = GET_CPUTB_DType_PTR(OType, output_);
}

template <typename DType, typename OType>
uint64_t Cast<DType, OType>::get_workload() {
  // TODO
  return 0;
}

template <typename DType, typename OType>
void Cast<DType, OType>::cast() {
  union input_data {
    DType in;
    uint32_t u;
    float f;
  };
  input_data a;
  input_data b;
  int num = 1;
  for (auto dim : fmap_o_) num *= dim;
  for (auto i = 0; i < num; i++) {
    if (data_type_in_ == xir::DataType::BFLOAT &&
        data_type_out_ == xir::DataType::BFLOAT) {
      rlt_[i] = img_[i];
    } else if (data_type_in_ == xir::DataType::BFLOAT) {
      a.in = img_[i];
      a.u = (a.u) << 16;
      rlt_[i] = static_cast<OType>(a.f);
    } else if (data_type_out_ == xir::DataType::BFLOAT) {
      b.f = static_cast<float>(img_[i]);
      std::uint32_t flag = (b.u & 0x00010000) >> 16;
      rlt_[i] = (b.u + 0x7fff + flag) & 0xFFFF0000;
    } else {
      rlt_[i] = static_cast<OType>(img_[i]);
    }
  }
}

INSTANTIATE_TPCLASS_SPECIFIED(Cast);
REG_SPECIFIED_DATATYPE_OP_INSTANCE_FUNC("cast", Cast);

}  // namespace cpu
}  // namespace vart