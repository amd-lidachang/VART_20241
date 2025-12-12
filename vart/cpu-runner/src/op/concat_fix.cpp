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

#include "concat_fix.hpp"

namespace vart {
namespace cpu {

// constructor and deconstructor
template <typename DType>
ConcatFix<DType>::ConcatFix(const xir::Subgraph* subg, const xir::Op* op,
                            IMapTBs_t inputs, CPUTBPtr_t output)
    : Concat<DType>(subg, op, inputs, output) {
  for (auto i = 0; i < input_num_; i++) {
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input", i);
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
    auto fp = xir_tensor_i->get_attr<int>("fix_point");
    fp_inputs_.push_back(fp);
  }

  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
}

template <typename DType>
void ConcatFix<DType>::run() {
  Concat<DType>::run();
}

template <typename DType>
void ConcatFix<DType>::print_param() {
  Concat<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "fix_width = " << fix_width_ << endl;
  for (auto i = 0; i < input_num_; i++) {
    UNI_LOG_DEBUG_INFO << "fp_input" << i << " = " << fp_inputs_[i] << endl;
  }
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
}

// read data from DDR
template <typename DType>
void ConcatFix<DType>::read() {
  auto key = "input";

  // handle input buffer
  data_in_.resize(input_num_);
  shift_read.resize(input_num_);
  for (auto i = 0; i < input_num_; i++) {
    auto* cputb = this->inputs_.at(key).at(i);
    data_in_[i] = GET_CPUTB_DType_PTR(DType, cputb);
    auto shift = fp_output_ - fp_inputs_[i];
    shift_read[i] = shift;
  }

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, CPUOPBase::output_);
}

template <typename DType>
void ConcatFix<DType>::concat_thread() {
  auto THREAD_NUM = input_num_;
  vector<std::future<int>> fut(THREAD_NUM);

  auto acc = 0;
  for (auto id = 0; id < input_num_; id++) {
    fut[id] = std::async(
        std::launch::async,
        [this](decltype(id) id, decltype(acc) acc) {
          auto inner_num = fmap_i_[id][axis_] * fmap_i_[id].cod(axis_);
          auto outer_num = fmap_i_[id].num() / inner_num;
          for (auto o = 0; o < outer_num; o++) {
            auto src_addr = o * inner_num;
            auto coord = fmap_i_[id].pos2coord(src_addr);
            coord[axis_] += acc;
            auto dst_addr = fmap_o_.coord2pos(coord);
            auto factor = pow(2, shift_read[id]);
            for (auto i = 0; i < inner_num; i++) {
              auto data_in =
                  DPURound<DType>((double)(data_in_[id][src_addr + i]) * factor,
                                  CPUOPBase::data_min_, CPUOPBase::data_max_);
              data_out_[dst_addr + i] = data_in;
            }
          }

          return 0;
        },
        id, acc);

    acc += fmap_i_[id][axis_];
  }

  for (auto i = 0; i < THREAD_NUM; i++) {
    fut[i].wait();
  }
}

template <typename DType>
void ConcatFix<DType>::concat_normal() {
  auto acc = 0;
  for (auto id = 0; id < input_num_; id++) {
    auto inner_num = fmap_i_[id][axis_] * fmap_i_[id].cod(axis_);
    auto outer_num = fmap_i_[id].num() / inner_num;
    for (auto o = 0; o < outer_num; o++) {
      auto src_addr = o * inner_num;
      auto coord = fmap_i_[id].pos2coord(src_addr);
      coord[axis_] += acc;
      auto dst_addr = fmap_o_.coord2pos(coord);

      auto factor = pow(2, shift_read[id]);
      for (auto i = 0; i < inner_num; i++) {
        auto data_in =
            DPURound<DType>((double)(data_in_[id][src_addr + i]) * factor,
                            CPUOPBase::data_min_, CPUOPBase::data_max_);
        data_out_[dst_addr + i] = data_in;
      }

      acc += fmap_i_[id][axis_];
    }
  }
}
INSTANTIATE_TPCLASS(ConcatFix);
REG_OP_INSTANCE_FUNC("concat-fix", ConcatFix);

}  // namespace cpu
}  // namespace vart
