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

#include "strided_slice.hpp"

namespace vart {
namespace cpu {

// constructor and deconstructor
template <typename DType>
StridedSlice<DType>::StridedSlice(const xir::Subgraph* subg, const xir::Op* op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 4);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 4);
  begin_origin_ = xir_op_->get_attr<vector<int>>("begin");
  end_origin_ = xir_op_->get_attr<vector<int>>("end");
  strides_origin_ = xir_op_->get_attr<vector<int32_t>>("strides");
  if (xir_op_->has_attr("begin_mask"))
    begin_mask_ = xir_op_->get_attr<int>("begin_mask");
  if (xir_op_->has_attr("end_mask"))
    end_mask_ = xir_op_->get_attr<int>("end_mask");
  if (xir_op_->has_attr("ellipsis_mask"))
    ellipsis_mask_ = xir_op_->get_attr<int>("ellipsis_mask");
  if (xir_op_->has_attr("new_axis_mask"))
    new_axis_mask_ = xir_op_->get_attr<int>("new_axis_mask");
  if (xir_op_->has_attr("shrink_axis_mask"))
    shrink_axis_mask_ = xir_op_->get_attr<int>("shrink_axis_mask");

  begin_.resize(fmap_i_.size());
  end_.resize(fmap_i_.size());
  strides_.resize(fmap_i_.size());
  std::vector<int32_t> out_shape;
  xir::validate_strided_slice(op, begin_, end_, strides_, out_shape);
  begin_.insert(begin_.begin(), 4 - begin_.size(), 0);
  end_.insert(end_.begin(), 4 - end_.size(), 1);
  strides_.insert(strides_.begin(), 4 - strides_.size(), 1);
  fmap_i_.insert(fmap_i_.begin(), 4 - fmap_i_.size(), 1);
  fmap_o_.resize(4);
  for (uint32_t i = 0u; i < 4; i++) {
    fmap_o_[i] = (end_[i] - begin_[i]) / strides_[i] +
                 ((end_[i] - begin_[i]) % strides_[i] != 0);
  }
  fmap_o_num_ = get_vec_mul(fmap_o_);
  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)(fmap_o_num_) / THREAD_NUM);
}

template <typename DType>
void StridedSlice<DType>::run() {
  // // do necessary check
  // print_param();
  check_param();

  // // read data
  // read();

  // do strided_slice
  strided_slice();

  // // do save, debug...
  // save();
}

template <typename DType>
void StridedSlice<DType>::print_param() {
  UNI_LOG_DEBUG_INFO << "fmap_i: " << Vec2Str(fmap_i_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "fmap_o: " << Vec2Str(fmap_o_, ", ") << endl;

  UNI_LOG_DEBUG_INFO << "begin: " << Vec2Str(begin_origin_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "end: " << Vec2Str(end_origin_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "strides: " << Vec2Str(strides_origin_, ", ") << endl;

  UNI_LOG_DEBUG_INFO << "begin_mask: " << begin_mask_ << endl;
  UNI_LOG_DEBUG_INFO << "end_mask: " << end_mask_ << endl;
  UNI_LOG_DEBUG_INFO << "ellipsis_mask: " << ellipsis_mask_ << endl;
  UNI_LOG_DEBUG_INFO << "new_axis_mask: " << new_axis_mask_ << endl;
  UNI_LOG_DEBUG_INFO << "shrink_axis_mask: " << shrink_axis_mask_ << endl;

  UNI_LOG_DEBUG_INFO << "transfered begin: " << Vec2Str(begin_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "transfered end: " << Vec2Str(end_, ", ") << endl;
  UNI_LOG_DEBUG_INFO << "transfered strides: " << Vec2Str(strides_, ", ")
                     << endl;

  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

template <typename DType>
void StridedSlice<DType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() == 1, VART_SIZE_ERROR);

  // check begin/end/stride
  UNI_LOG_CHECK(begin_origin_.size() == strides_origin_.size(), VART_SIZE_ERROR)
      << ", " << begin_origin_.size() << " != " << strides_origin_.size();
  UNI_LOG_CHECK(end_origin_.size() == strides_origin_.size(), VART_SIZE_ERROR)
      << ", " << end_origin_.size() << " != " << strides_origin_.size();

  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
  // i.e. there exists only no more than one ellipsis
  UNI_LOG_CHECK(
      !(ellipsis_mask_ && ((ellipsis_mask_ & (ellipsis_mask_ - 1)) != 0)),
      VART_INVALID_VALUE)
      << ", multiple ellipses in slice spec not allowed";
}

// read data from DDR
template <typename DType>
void StridedSlice<DType>::read() {
  // read img data
  auto* cputb = inputs_.at("input").at(0);
  data_in_ = GET_CPUTB_DType_PTR(DType, cputb);

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
uint64_t StridedSlice<DType>::get_workload() {
  // TODO
  return 0;
}

template <typename DType>
void StridedSlice<DType>::strided_slice() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    calculate_thread();
  } else {
    calculate_normal();
  }
}

template <typename DType>
void StridedSlice<DType>::calculate_normal() {
  calculate(0, fmap_o_num_);
}

template <typename DType>
void StridedSlice<DType>::calculate_thread() {
  vector<std::future<void>> fut(THREAD_NUM);
  for (auto thread_id = 0U; thread_id < THREAD_NUM; ++thread_id) {
    auto start_index = thread_id * THREAD_WORKLOAD;
    auto FMAP_SIZE = fmap_o_num_;
    auto end_index =
        std::min<uint32_t>(FMAP_SIZE, start_index + THREAD_WORKLOAD);
    fut[thread_id] = std::async(
        std::launch::async,
        [this](decltype(start_index) start_index,
               decltype(end_index) end_index) {
          calculate(start_index, end_index);
        },
        start_index, end_index);
  }
  for (auto thread_id = 0U; thread_id < THREAD_NUM; thread_id++) {
    fut[thread_id].wait();
  }
}

template <typename DType>
void StridedSlice<DType>::calculate(std::uint32_t start_index,
                                    std::uint32_t end_index) {
  for (uint32_t pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    int32_t dim3_iter = pos_iter % fmap_o_[3];
    int32_t dim2_iter = (pos_iter / (fmap_o_[3])) % fmap_o_[2];
    int32_t dim1_iter = (pos_iter / (fmap_o_[3] * fmap_o_[2])) % fmap_o_[1];
    int32_t dim0_iter = pos_iter / (fmap_o_[3] * fmap_o_[2] * fmap_o_[1]);

    int32_t src_pos =
        ((dim0_iter * strides_[0]) + begin_[0]) * fmap_i_[1] * fmap_i_[2] *
            fmap_i_[3] +
        ((dim1_iter * strides_[1]) + begin_[1]) * fmap_i_[2] * fmap_i_[3] +
        ((dim2_iter * strides_[2]) + begin_[2]) * fmap_i_[3] +
        ((dim3_iter * strides_[3]) + begin_[3]);
    data_out_[pos_iter] = data_in_[src_pos];
  }
}

INSTANTIATE_TPCLASS(StridedSlice);
REG_OP_INSTANCE_FUNC("strided_slice", StridedSlice);

}  // namespace cpu
}  // namespace vart
