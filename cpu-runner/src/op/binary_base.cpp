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

#include "binary_base.hpp"

namespace vart {
namespace cpu {

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
const vector<string> BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::BinaryTypeName =
    {
        "add",      //
        "sub",      //
        "mul",      //
        "div",      //
        "unknown",  //
};

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
const vector<string> BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::ITName = {
    "input",  //
};

// constructor and deconstructor
template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::BinaryBase(const xir::Subgraph* subg,
                                                     const xir::Op* op,
                                                     IMapTBs_t inputs,
                                                     CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_ANY_DIM_FMAPS(fmap_i_, input);
  GET_OUTPUT_ANY_DIM_FMAP(fmap_o_);
  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
  no_broadcast_ = std::all_of(
      fmap_i_.begin(), fmap_i_.end(),
      [this](const Dimension map_iter) { return fmap_o_ == map_iter; });
}

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
void BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::run() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    calculate_thread();
  } else {
    calculate_normal();
  }
}

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
void BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::calculate_normal() {
  calculate(0, fmap_o_.num(), no_broadcast_);
}

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
void BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::calculate_thread() {
  vector<std::future<void>> fut(THREAD_NUM);
  for (auto thread_id = 0U; thread_id < THREAD_NUM; ++thread_id) {
    auto start_index = thread_id * THREAD_WORKLOAD;
    auto FMAP_SIZE = fmap_o_.num();
    auto end_index =
        std::min<uint32_t>(FMAP_SIZE, start_index + THREAD_WORKLOAD);
    fut[thread_id] = std::async(
        std::launch::async,
        [this](decltype(start_index) start_index, decltype(end_index) end_index,
               decltype(no_broadcast_) no_broadcast_) {
          calculate(start_index, end_index, no_broadcast_);
        },
        start_index, end_index, no_broadcast_);
  }
  for (auto thread_id = 0U; thread_id < THREAD_NUM; thread_id++) {
    fut[thread_id].wait();
  }
}

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
void BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::print_param() {
  UNI_LOG_DEBUG_INFO << "binary_type: " << BinaryTypeName[binary_type_] << endl;
  fmap_i_[0].print_param("fmap_i0");
  fmap_i_[1].print_param("fmap_i1");
  fmap_o_.print_param("fmap_o");
  UNI_LOG_DEBUG_INFO << "broadcast = " << !no_broadcast_;

  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
void BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::check_param() {
  UNI_LOG_CHECK(inputs_.at(ITName[INPUT]).size() == 2, VART_SIZE_ERROR);
  for (auto dim_iter = 0; dim_iter < fmap_o_.ndims(); dim_iter++) {
    // check input 0
    if ((fmap_o_[dim_iter] != fmap_i_[0][dim_iter])) {
      UNI_LOG_CHECK(fmap_i_[0][dim_iter] == 1, VART_INVALID_VALUE);
    }
    // check input 1
    if ((fmap_o_[dim_iter] != fmap_i_[1][dim_iter])) {
      UNI_LOG_CHECK(fmap_i_[1][dim_iter] == 1, VART_INVALID_VALUE);
    }
  }
}

// read data from DDR
template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
void BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::read() {
  // need check yunfengy
  if ((get_type() == "eltwise-fix") && (get_input_num() == 1)) {
    return;
  }

  // align input dimension
  auto vi0 = fmap_i_[0].vdims();
  while ((int)vi0.size() != fmap_o_.ndims()) {
    vi0.insert(vi0.begin(), 1);
  }
  fmap_i_[0] = Dimension(vi0);

  auto vi1 = fmap_i_[1].vdims();
  while ((int)vi1.size() != fmap_o_.ndims()) {
    vi1.insert(vi1.begin(), 1);
  }
  fmap_i_[1] = Dimension(vi1);

  // assign input buffer
  auto* cputb_a = inputs_.at(ITName[INPUT]).at(0);
  data_ina_ptr_ = GET_CPUTB_DType_PTR(DTypeIn0, cputb_a);

  auto* cputb_b = inputs_.at(ITName[INPUT]).at(1);
  data_inb_ptr_ = GET_CPUTB_DType_PTR(DTypeIn1, cputb_b);

  // handle output buffer
  data_out_ptr_ = GET_CPUTB_DType_PTR(DTypeOut, output_);
}

template <typename DTypeIn0, typename DTypeIn1, typename DTypeOut>
uint64_t BinaryBase<DTypeIn0, DTypeIn1, DTypeOut>::get_workload() {
  return fmap_o_.num();
}

INSTANTIATE_TPCLASS_BINARY(BinaryBase);

}  // namespace cpu
}  // namespace vart
