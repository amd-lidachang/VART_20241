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

#include "eltwise.hpp"

namespace vart {
namespace cpu {

template <typename DType>
const vector<string> Eltwise<DType>::ITName = {
    "input",
};

// constructor and deconstructor
template <typename DType>
Eltwise<DType>::Eltwise(const xir::Subgraph* subg, const xir::Op* op,
                        IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAPS(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  input_num_ = fmap_i_.size();
  if (op->has_attr("type")) {
    elt_type_ = op->get_attr<std::string>("type");
    std::for_each(elt_type_.begin(), elt_type_.end(),
                  [](char& c) { c = std::toupper(c); });
  }

  broadcast_ =
      std::any_of(fmap_i_.begin(), fmap_i_.end(),
                  [this](const Dimension map_i) { return fmap_o_ != map_i; });

  auto input_ops = xir_op_->get_input_ops("input");
  shift = std::all_of(input_ops.begin(), input_ops.end(), [=](auto input_op) {
    return input_op->get_type() == "fix";
  });
  auto fanout_op = xir_op_->get_fanout_ops().front();
  if (fanout_op->get_type() == "fix") {
    shift &= true;
  } else if (fanout_op->get_type() == "relu") {
    shift &= fanout_op->get_fanout_ops().front()->get_type() == "fix";
  }

  if (shift) {
    for (auto input_op : input_ops) {
      fp_inputs_.push_back(input_op->get_attr<std::int32_t>("fix_point"));
    }
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    for (auto i = 0; i < input_num_; ++i) {
      auto shift_read = (elt_type_ == "ADD") ? fp_inputs_[i] - fp_min : 0;
      shift_factor_.push_back(2 + fp_inputs_[i] - shift_read);
    }
  }

  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType>
void Eltwise<DType>::align_dim() {
  for (auto i = 0; i < input_num_; ++i) {
    while (fmap_i_[i].ndims() != fmap_o_.ndims()) {
      fmap_i_[i].insert_dim(0, 1);
    }
  }
}

template <typename DType>
void Eltwise<DType>::run() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    eltwise_thread();
  } else {
    eltwise_normal();
  }
}

template <typename DType>
void Eltwise<DType>::print_param() {
  for (auto i = 0; i < input_num_; i++) {
    fmap_i_[i].print_param("fmap_i" + to_string(i));
  }
  fmap_o_.print_param("fmap_o");

  UNI_LOG_DEBUG_INFO << "elt_type = " << elt_type_ << endl;
  UNI_LOG_DEBUG_INFO << "broadcast = " << broadcast_ << endl;
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

template <typename DType>
void Eltwise<DType>::check_param() {
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
void Eltwise<DType>::read() {
  // handle input buffer
  data_in_.resize(input_num_);
  for (auto idx = 0; idx < input_num_; idx++) {
    auto* cputb = inputs_.at(ITName[INPUT]).at(idx);
    data_in_[idx] = GET_CPUTB_DType_PTR(DType, cputb);
  }

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);

  if ("MUL" == elt_type_) {
    std::fill_n(data_out_, 1, fmap_o_.num());
  } else {
    std::fill_n(data_out_, 0, fmap_o_.num());
  }
}

template <typename DType>
uint64_t Eltwise<DType>::get_workload() {
  return fmap_o_.num();
}

template <typename DType>
void Eltwise<DType>::eltwise(std::uint32_t start_index,
                             std::uint32_t end_index) {
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;
  for (auto pos_iter = start_index; pos_iter < end_index; ++pos_iter) {
    for (auto input_iter = 0; input_iter < input_num_; input_iter++) {
      auto pos = pos_iter;
      if (broadcast_) {
        fmap_o_.pos2coord(pos_iter, dst_coord);
        for (auto dim_iter = 0U; dim_iter < src_coord.size(); dim_iter++) {
          src_coord[dim_iter] =
              dst_coord[dim_iter] % fmap_i_[input_iter][dim_iter];
        }
        pos = fmap_i_[input_iter].coord2pos(src_coord);
      }
      if ("MUL" == elt_type_) {
        if (shift) {
          data_out_[pos_iter] *= floor(data_in_[input_iter][pos] *
                                       pow(2, shift_factor_[input_iter])) /
                                 pow(2, shift_factor_[input_iter]);
        } else {
          data_out_[pos_iter] *= data_in_[input_iter][pos];
        }
      } else if ("ADD" == elt_type_) {
        if (shift) {
          data_out_[pos_iter] += floor(data_in_[input_iter][pos] *
                                       pow(2, shift_factor_[input_iter])) /
                                 pow(2, shift_factor_[input_iter]);
        } else {
          data_out_[pos_iter] += data_in_[input_iter][pos];
        }
      }
    }
  }
}

template <typename DType>
void Eltwise<DType>::eltwise_normal() {
  eltwise(0, fmap_o_.num());
}

template <typename DType>
void Eltwise<DType>::eltwise_thread() {
  vector<std::future<void>> fut(THREAD_NUM);
  for (auto thread_id = 0U; thread_id < THREAD_NUM; ++thread_id) {
    auto start_index = thread_id * THREAD_WORKLOAD;
    auto FMAP_SIZE = fmap_o_.num();
    auto end_index =
        std::min<uint32_t>(FMAP_SIZE, start_index + THREAD_WORKLOAD);
    fut[thread_id] = std::async(std::launch::async,
                                [this](decltype(start_index) start_index,
                                       decltype(end_index) end_index) {
                                  eltwise(start_index, end_index);
                                },
                                start_index, end_index);
  }

  for (auto thread_id = 0U; thread_id < THREAD_NUM; thread_id++) {
    fut[thread_id].wait();
  }
}

INSTANTIATE_TPCLASS(Eltwise);
REG_OP_INSTANCE_FUNC("eltwise", Eltwise);

}  // namespace cpu
}  // namespace vart
