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

#include "binary_add.hpp"

#include "align_buf_mgr.hpp"

namespace vart {
namespace cpu {

double dr(double x);

template <typename DType>
Add<DType>::Add(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_ANY_DIM_FMAPS(fmap_i_, input);
  GET_OUTPUT_ANY_DIM_FMAP(fmap_o_);
  align_dim();

  auto input_ops = xir_op_->get_input_ops("input");
  shift = std::all_of(input_ops.begin(), input_ops.end(), [=](auto input_op) {
    return input_op->get_type() == "fix";
  });
  auto fanout_op = xir_op_->get_fanout_ops().front();
  if (fanout_op->get_type() == "fix") {
    shift &= true;
  } else if (fanout_op->get_type() == "relu") {
    shift &= fanout_op->get_fanout_ops().front()->get_type() == "fix";
  } else {
    shift = false;
  }

  if (shift) {
    for (auto input_op : input_ops) {
      fp_inputs_.push_back(input_op->get_attr<std::int32_t>("fix_point"));
    }
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    for (auto i = 0U; i < fmap_i_.size(); ++i) {
      auto shift_read = fp_inputs_[i] - fp_min;
      shift_factor_.push_back(2 + fp_inputs_[i] - shift_read);
    }
  }
  THREAD_NUM = fmap_o_.num() <= CPU_NUM ? 1 : CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType>
void Add<DType>::align_dim() {
  for (unsigned i = 0; i < fmap_i_.size(); i++) {
    while (fmap_i_[i].ndims() != fmap_o_.ndims()) {
      fmap_i_[i].insert_dim(0, 1);
    }
  }
}

template <typename DType>
void Add<DType>::print_param() {
  for (auto i = 0U; i < fmap_i_.size(); i++) {
    fmap_i_[i].print_param("fmap_i" + to_string(i));
  }
  fmap_o_.print_param("fmap_o");

  UNI_LOG_DEBUG_INFO << "shift = " << shift << endl;

  if (shift) {
    for (auto i = 0U; i < fmap_i_.size(); i++) {
      UNI_LOG_DEBUG_INFO << "fp_inputs[" << i << "] = " << fp_inputs_[i];
      UNI_LOG_DEBUG_INFO << "shift_factor[" << i << "] = " << shift_factor_[i];
    }
  }

  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

// read data from DDR
template <typename DType>
void Add<DType>::read() {
  // assign input buffer
  data_in_ptr_.resize(fmap_i_.size());
  for (auto idx = 0U; idx < fmap_i_.size(); idx++) {
    auto* cputb = inputs_.at("input").at(idx);
    data_in_ptr_[idx] = GET_CPUTB_DType_PTR(DType, cputb);
  }
  // handle output buffer
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
  std::fill_n(data_out_ptr_, fmap_o_.num(), DType(0));
}

template <typename DType>
void Add<DType>::add_normal() {
  add(0, fmap_o_.num());
}

template <typename DType>
void Add<DType>::add_thread() {
  vector<std::future<void>> fut(THREAD_NUM);
  for (auto thread_id = 0U; thread_id < THREAD_NUM; ++thread_id) {
    auto start_index = thread_id * THREAD_WORKLOAD;
    auto FMAP_SIZE = fmap_o_.num();
    auto end_index =
        std::min<uint32_t>(FMAP_SIZE, start_index + THREAD_WORKLOAD);
    fut[thread_id] = std::async(
        std::launch::async,
        [this](decltype(start_index) start_index,
               decltype(end_index) end_index) { add(start_index, end_index); },
        start_index, end_index);
  }

  for (auto thread_id = 0U; thread_id < THREAD_NUM; thread_id++) {
    fut[thread_id].wait();
  }
}

template <typename DType>
void Add<DType>::add(std::uint32_t start_index, std::uint32_t end_index) {
  for (auto i = start_index; i < end_index; ++i) {
    for (auto j = 0U; j < fmap_i_.size(); j++) {
      auto coord = fmap_o_.pos2coord(i);
      for (auto k = 0U; k < coord.size(); k++) {
        if (fmap_i_[j][k] == 1) {
          coord[k] = 0;
        }
      }
      auto pos = fmap_i_[j].coord2pos(coord);
      if (shift) {
        data_out_ptr_[i] +=
            floor(data_in_ptr_[j][pos] * pow(2, shift_factor_[j])) /
            pow(2, shift_factor_[j]);
      } else {
        data_out_ptr_[i] += data_in_ptr_[j][pos];
      }
    }
  }
}

template <typename DType>
void Add<DType>::run() {
  read();
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    add_thread();
  } else {
    add_normal();
  }
}

template <typename DType>
AddFix<DType>::AddFix(const xir::Subgraph* subg, const xir::Op* op,
                      IMapTBs_t inputs, CPUTBPtr_t output)
    : Add<DType>(subg, op, inputs, output) {
  auto input_num = op->get_input_num();
  for (auto i = 0; i < input_num; i++) {
    auto* xir_tensor_i = op->get_input_tensor("input", i);
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
    auto fp = xir_tensor_i->get_attr<int>("fix_point");
    fp_inputs_.push_back(fp);
  }

  auto* xir_tensor_o = op->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");

  auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
  for (auto i = 0; i < input_num; i++) {
    shift_read_.push_back(fp_inputs_[i] - fp_min);
  }
  shift_write_ = fp_min - fp_output_;
  act_type_ = op->get_attr<string>("nonlinear");
  if (act_type_ == "HSIGMOID") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
    shift_write_ = fp_min - hsigmoid_in_;
  }
}

template <typename DType>
void AddFix<DType>::add(std::uint32_t start_index, std::uint32_t end_index) {
  for (auto i = start_index; i < end_index; i++) {
    float tmp = 0;
    for (auto j = 0U; j < fmap_i_.size(); j++) {
      auto coord = fmap_o_.pos2coord(i);
      for (auto k = 0U; k < coord.size(); k++) {
        if (fmap_i_[j][k] == 1) coord[k] = 0;
      }
      auto pos = fmap_i_[j].coord2pos(coord);
      tmp += data_in_ptr_[j][pos] / pow(2.0, shift_read_[j] - 2);
    }
    tmp /= pow(2.0, shift_write_ + 2);
    if (act_type_ == "RELU") {
      if (tmp < 0) tmp = 0;
    } else if (act_type_ == "LEAKYRELU") {
      if (tmp < 0) tmp *= 26.f / 256.f;
    } else if (act_type_ == "RELU6") {
      if (tmp < 0) tmp = 0;
      if (fp_output_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (act_type_ == "HSIGMOID") {
      tmp = dr(tmp);
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    }
    data_out_ptr_[i] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS(Add);
INSTANTIATE_TPCLASS(AddFix);
REG_OP_INSTANCE_FUNC("add", Add);

}  // namespace cpu
}  // namespace vart
