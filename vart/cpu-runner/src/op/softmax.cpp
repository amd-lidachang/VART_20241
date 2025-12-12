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

#include "softmax.hpp"

namespace vart {
namespace cpu {

// constructor and deconstructor
template <typename DType>
Softmax<DType>::Softmax(const xir::Subgraph* subg, const xir::Op* op,
                        IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 6);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 6);

  axis_ = xir_op_->get_attr<int>("axis");
  axis_ = axis_ < 0
              ? (xir_op_->get_input_tensor("input")->get_shape().size() + axis_)
              : axis_;
  output_f_.resize(fmap_o_.num());
  axis_stride_ = static_cast<int>(fmap_i_.cod(axis_));
  loop_times_ = fmap_i_.num() / fmap_i_[axis_];
  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)loop_times_ / THREAD_NUM);
}

template <typename DType>
void Softmax<DType>::run() {
  std::copy_n(data_in_ptr_, fmap_i_.num(), output_f_.begin());
  softmax();
  std::copy_n(output_f_.begin(), fmap_i_.num(), data_out_ptr_);
}

template <typename DType>
void Softmax<DType>::softmax() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    softmax_thread();
  } else {
    softmax_normal();
  }
}

template <typename DType>
void Softmax<DType>::print_param() {
  fmap_i_.print_param("fmap_i");
  fmap_o_.print_param("fmap_o");
  UNI_LOG_DEBUG_INFO << "axis = " << axis_ << endl;
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

template <typename DType>
void Softmax<DType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() == 1, VART_SIZE_ERROR);
  // check fmap_out height
  UNI_LOG_CHECK(fmap_i_ == fmap_o_, VART_SIZE_ERROR)
      << Vec2Str(fmap_i_.vdims(), ",")
      << " != " << Vec2Str(fmap_o_.vdims(), ",");
}

// read data from DDR
template <typename DType>
void Softmax<DType>::read() {
  auto* cputb = inputs_.at("input").at(0);
  data_in_ptr_ = GET_CPUTB_DType_PTR(DType, cputb);
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
uint64_t Softmax<DType>::get_workload() {
  // find max, minus max, exp, sum, division.
  return 5 * fmap_i_.num();
}

template <typename DType>
void Softmax<DType>::softmax_normal() {
  if (axis_stride_ == 1) {
    for (auto loop_iter = 0u; loop_iter < loop_times_; loop_iter++) {
      softmax_one_contiguous(loop_iter);
    }
  } else {
    for (auto loop_iter = 0u; loop_iter < loop_times_; loop_iter++) {
      softmax_one_discrete(loop_iter);
    }
  }
}

template <typename DType>
void Softmax<DType>::softmax_thread() {
  vector<std::future<int>> thr_fut(THREAD_NUM);
  for (auto i = 0U; i < THREAD_NUM; i++) {
    // thr_fut[i] = std::async(std::launch::async | std::launch::deferred,
    thr_fut[i] = std::async(
        std::launch::async,
        [this](decltype(i) i) {
          auto BASE_POS = i * THREAD_WORKLOAD;
          for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
            auto loop_iter = BASE_POS + j;
            if (loop_iter >= loop_times_) break;
            if (axis_stride_ == 1) {
              softmax_one_contiguous(loop_iter);
            } else {
              softmax_one_discrete(loop_iter);
            }
          }
          return 0;
        },
        i);
  }
  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
}

template <typename DType>
void Softmax<DType>::softmax_one_contiguous(int loop_iter) {
  int offset = loop_iter * fmap_i_[axis_];
  vart::cpu::softmax(output_f_.data() + offset,
                     static_cast<int>(fmap_i_[axis_]),
                     output_f_.data() + offset);
}

template <typename DType>
void Softmax<DType>::softmax_one_discrete(int loop_iter) {
  std::vector<float> data_contiguous(fmap_i_[axis_]);
  int low_dim = axis_stride_;
  int low_iter = loop_iter % low_dim;
  int high_iter = loop_iter / low_dim;
  for (auto iter = 0; iter < fmap_i_[axis_]; iter++) {
    int offset =
        high_iter * fmap_i_[axis_] * low_dim + iter * low_dim + low_iter;
    data_contiguous[iter] = output_f_[offset];
  }
  vart::cpu::softmax(data_contiguous.data(), fmap_i_[axis_],
                     data_contiguous.data());
  for (auto iter = 0; iter < fmap_i_[axis_]; iter++) {
    int offset =
        high_iter * fmap_i_[axis_] * low_dim + iter * low_dim + low_iter;
    output_f_[offset] = data_contiguous[iter];
  }
}

INSTANTIATE_TPCLASS(Softmax);
REG_OP_INSTANCE_FUNC("softmax", Softmax);

}  // namespace cpu
}  // namespace vart
