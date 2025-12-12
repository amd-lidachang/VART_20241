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

#include "cost_volume.hpp"

namespace vart {
namespace cpu {

template <typename DType>
const vector<string> CostVolume<DType>::ITName = {
    "input",
};

// constructor and deconstructor
template <typename DType>
CostVolume<DType>::CostVolume(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIM4_FMAPS(fmap_i_, input);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  input_num_ = fmap_i_.size();
  auto maxdisp = op->get_attr<std::int32_t>("maxdisp");
  stride_ = std::floor(maxdisp / 4);
  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType>
void CostVolume<DType>::run() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    cost_volume_thread();
  } else {
    cost_volume_normal();
  }
}

template <typename DType>
void CostVolume<DType>::print_param() {
  for (auto i = 0; i < input_num_; i++) {
    fmap_i_[i].print_param("fmap_i" + to_string(i));
  }
  fmap_o_.print_param("fmap_o");
  UNI_LOG_DEBUG_INFO << "stride = " << stride_ << std::endl;

  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

template <typename DType>
void CostVolume<DType>::check_param() {
  UNI_LOG_CHECK(input_num_ == 2, VART_SIZE_ERROR)
      << ", Err: inputs size is " << input_num_;
  UNI_LOG_CHECK(fmap_i_[0] == fmap_i_[1], VART_SIZE_ERROR);
  UNI_LOG_CHECK(fmap_i_[0].w >= stride_, VART_SIZE_ERROR);
}

// read data from DDR
template <typename DType>
void CostVolume<DType>::read() {
  // handle input buffer
  data_in_.resize(input_num_);
  for (auto idx = 0; idx < input_num_; idx++) {
    auto* cputb = inputs_.at(ITName[INPUT]).at(idx);
    data_in_[idx] = GET_CPUTB_DType_PTR(DType, cputb);
  }

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);
  std::fill_n(data_out_, fmap_o_.num(),DType(0));
}

template <typename DType>
uint64_t CostVolume<DType>::get_workload() {
  return fmap_o_.num();
}

template <typename DType>
void CostVolume<DType>::cost_volume(std::int32_t start_index,
                                    std::int32_t end_index) {
  for (auto idx = start_index; idx < end_index; ++idx) {
    auto fmap_tmp = fmap_o_.pos2coord(idx);
    if (fmap_tmp.w < fmap_tmp.d) {
      data_out_[idx] = 0;
    } else {
      // input_0
      if (fmap_tmp.c < fmap_i_[0].c) {
        auto addr = fmap_tmp.n * fmap_i_[0].ncod() +
                    fmap_tmp.h * fmap_i_[0].hcod() +
                    fmap_tmp.w * fmap_i_[0].wcod() + fmap_tmp.c;
        data_out_[idx] = data_in_[0][addr];
      } else {  // input_1
        auto addr = fmap_tmp.n * fmap_i_[1].ncod() +
                    fmap_tmp.h * fmap_i_[1].hcod() +
                    (fmap_tmp.w - fmap_tmp.d) * fmap_i_[1].wcod() +
                    (fmap_tmp.c - fmap_i_[1].c);
        data_out_[idx] = data_in_[1][addr];
      }
    }
  }
}

template <typename DType>
void CostVolume<DType>::cost_volume_normal() {
  cost_volume(0, fmap_o_.num());
}

template <typename DType>
void CostVolume<DType>::cost_volume_thread() {
  vector<std::future<void>> fut(THREAD_NUM);
  auto FMAP_SIZE = fmap_o_.num();
  for (auto thread_id = 0U; thread_id < THREAD_NUM; ++thread_id) {
    auto start_index = thread_id * THREAD_WORKLOAD;
    auto end_index =
        std::min<uint32_t>(FMAP_SIZE, start_index + THREAD_WORKLOAD);
    fut[thread_id] = std::async(std::launch::async,
                                [this](decltype(start_index) start_index,
                                       decltype(end_index) end_index) {
                                  cost_volume(start_index, end_index);
                                },
                                start_index, end_index);
  }

  for (auto thread_id = 0U; thread_id < THREAD_NUM; ++thread_id) {
    fut[thread_id].wait();
  }
}

INSTANTIATE_TPCLASS(CostVolume);
REG_OP_INSTANCE_FUNC("cost_volume", CostVolume);

}  // namespace cpu
}  // namespace vart
