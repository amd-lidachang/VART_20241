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

#include "depthwise_fix.hpp"

namespace vart {
namespace cpu {

template <typename DType>
const vector<string> DepthwiseFix<DType>::ITName = {
    "input",
};

// constructor and deconstructor
template <typename DType>
DepthwiseFix<DType>::DepthwiseFix(const xir::Subgraph* subg, const xir::Op* op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAPS(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  input_num_ = fmap_i_.size();
  dpt_type_ = op->get_attr<std::string>("type");
  std::for_each(dpt_type_.begin(), dpt_type_.end(),
                [](char& c) { c = std::toupper(c); });

  for (auto i = 0; i < input_num_; i++) {
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input", i);
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
    auto fp = xir_tensor_i->get_attr<int>("fix_point");
    fp_inputs_.push_back(fp);
  }

  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");

  if (dpt_type_ == "ADD") {
    auto fp_max = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_max - fp_inputs_[i]);
    }
    shift_write_ = fp_output_ - fp_max;
  } else if (dpt_type_ == "MUL") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    auto fp_sum = std::accumulate(fp_inputs_.begin(), fp_inputs_.end(), 0);
    shift_write_ = fp_output_ - fp_sum;
  }

  broadcast_ =
      std::any_of(fmap_i_.begin(), fmap_i_.end(),
                  [this](const Dimension map_i) { return fmap_o_ != map_i; });

  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType>
void DepthwiseFix<DType>::align_dim() {
  for (auto i = 0; i < input_num_; ++i) {
    while (fmap_i_[i].ndims() != fmap_o_.ndims()) {
      fmap_i_[i].insert_dim(0, 1);
    }
  }
}

template <typename DType>
void DepthwiseFix<DType>::run() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    depthwise_thread();
  } else {
    depthwise_normal();
  }
}

template <typename DType>
void DepthwiseFix<DType>::print_param() {
  for (auto i = 0; i < input_num_; i++) {
    fmap_i_[i].print_param("fmap_i" + to_string(i));
  }
  fmap_o_.print_param("fmap_o");

  UNI_LOG_DEBUG_INFO << "dpt_type = " << dpt_type_ << endl;
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

template <typename DType>
void DepthwiseFix<DType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() >= 1, VART_SIZE_ERROR)
      << ", Err: inputs size is " << inputs_.size();
  for (auto dim_iter = 0; dim_iter < fmap_o_.ndims(); dim_iter++) {
    for (auto fp_iter = 0U; fp_iter <= fmap_i_.size(); fp_iter++) {
      if ((fmap_o_[dim_iter] != fmap_i_[fp_iter][dim_iter])) {
        UNI_LOG_CHECK(fmap_i_[fp_iter][dim_iter] == 1, VART_INVALID_VALUE);
      }
    }
  }
}

// read data from DDR
template <typename DType>
void DepthwiseFix<DType>::read() {
  // handle input buffer
  data_in_.resize(input_num_);
  for (auto idx = 0; idx < input_num_; idx++) {
    auto* cputb = inputs_.at(ITName[INPUT]).at(idx);
    data_in_[idx] = GET_CPUTB_DType_PTR(DType, cputb);
  }

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);

  if ("MUL" == dpt_type_) {
    std::fill_n(data_out_, 1, fmap_o_.num());
  } else {
    std::fill_n(data_out_, 0, fmap_o_.num());
  }
}

template <typename DType>
uint64_t DepthwiseFix<DType>::get_workload() {
  return fmap_o_.num();
}

template <typename DType>
void DepthwiseFix<DType>::depthwise(std::uint32_t start_index,
                                    std::uint32_t end_index) {
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;
  for (auto pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    double tmp = (dpt_type_ == "MUL") ? 1 : 0;
    // dim_t pos;
    for (auto fp_iter = 0U; fp_iter < fmap_i_.size(); fp_iter++) {
      auto pos = pos_iter;
      if (broadcast_) {
        fmap_o_.pos2coord(pos_iter, dst_coord);
        for (auto dim_iter = 0U; dim_iter < dst_coord.size(); dim_iter++) {
          src_coord[dim_iter] =
              dst_coord[dim_iter] % fmap_i_[fp_iter][dim_iter];
        }
        pos = fmap_i_[fp_iter].coord2pos(src_coord);
      }
      if (dpt_type_ == "ADD") {
        tmp += data_in_[fp_iter][pos] * pow(2.0, shift_read_[fp_iter]);
      } else if (dpt_type_ == "MUL") {
        tmp *= data_in_[fp_iter][pos];
      }
    }

    tmp *= pow(2.0, shift_write_);
    data_out_[pos_iter] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

template <typename DType>
void DepthwiseFix<DType>::depthwise_normal() {
  depthwise(0, fmap_o_.num());
}

template <typename DType>
void DepthwiseFix<DType>::depthwise_thread() {
  vector<std::future<void>> fut(THREAD_NUM);
  for (auto thread_id = 0U; thread_id < THREAD_NUM; ++thread_id) {
    auto start_index = thread_id * THREAD_WORKLOAD;
    auto FMAP_SIZE = fmap_o_.num();
    auto end_index =
        std::min<uint32_t>(FMAP_SIZE, start_index + THREAD_WORKLOAD);
    fut[thread_id] = std::async(std::launch::async,
                                [this](decltype(start_index) start_index,
                                       decltype(end_index) end_index) {
                                  depthwise(start_index, end_index);
                                },
                                start_index, end_index);
  }

  for (auto thread_id = 0U; thread_id < THREAD_NUM; thread_id++) {
    fut[thread_id].wait();
  }
}

INSTANTIATE_TPCLASS(DepthwiseFix);
REG_OP_INSTANCE_FUNC("depthwise-fix", DepthwiseFix);

}  // namespace cpu
}  // namespace vart
