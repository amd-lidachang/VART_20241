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

#include "correlation2d_elemwise.hpp"
#include "align_buf_mgr.hpp"
#include "fast_pad.hpp"

namespace vart {
namespace cpu {

template <typename DType>
Correlation2dElemwise<DType>::Correlation2dElemwise(const xir::Subgraph* subg,
                                                    const xir::Op* op,
                                                    IMapTBs_t inputs,
                                                    CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  // get i/o fmap
  GET_INPUT_DIM4_FMAPS(raw_fmap_i_, input);
  fmap_i_ = raw_fmap_i_;

  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  UNI_LOG_CHECK(op->has_attr("pad_size"), VART_INVALID_VALUE);
  pad_size_ = op->get_attr<std::int32_t>("pad_size");
  pad_ = Pad_t{pad_size_, pad_size_, pad_size_, pad_size_};

  fmap_i_[1].h += pad_size_ * 2;
  fmap_i_[1].w += pad_size_ * 2;

  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType>
void Correlation2dElemwise<DType>::print_param() {
  if (!CPUCfg::Instance().debug()) return;

  raw_fmap_i_[0].print_param("raw_fmap_i_0");
  fmap_i_[0].print_param("fmap_i_0");
  raw_fmap_i_[1].print_param("raw_fmap_i_1");
  fmap_i_[1].print_param("fmap_i_1");
  fmap_o_.print_param("fmap_o");

  pad_.print_param("pad");
}

template <typename DType>
void Correlation2dElemwise<DType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() == 2, VART_SIZE_ERROR)
      << ", Err: inputs size is " << inputs_.size();

  UNI_LOG_CHECK(pad_size_ <= 4, VART_INVALID_VALUE)
      << ", Err: pad size is " << pad_size_;

  UNI_LOG_CHECK(fmap_o_.d == std::pow((2 * pad_size_ + 1), 2),
                VART_INVALID_VALUE);

  // check input fmap
  UNI_LOG_CHECK(raw_fmap_i_[0].n == raw_fmap_i_[1].n, VART_INVALID_VALUE)
      << ", " << raw_fmap_i_[0].n << " != " << raw_fmap_i_[1].n;
  UNI_LOG_CHECK(raw_fmap_i_[0].h == raw_fmap_i_[1].h, VART_INVALID_VALUE)
      << ", " << raw_fmap_i_[0].h << " != " << raw_fmap_i_[1].h;
  UNI_LOG_CHECK(raw_fmap_i_[0].w == raw_fmap_i_[1].w, VART_INVALID_VALUE)
      << ", " << raw_fmap_i_[0].w << " != " << raw_fmap_i_[1].w;
  UNI_LOG_CHECK(raw_fmap_i_[0].c == raw_fmap_i_[1].c, VART_INVALID_VALUE)
      << ", " << raw_fmap_i_[0].c << " != " << raw_fmap_i_[1].c;
}

template <typename DType>
uint64_t Correlation2dElemwise<DType>::get_workload() {
  return 0;
}

template <typename DType>
void Correlation2dElemwise<DType>::read() {
  // handle input buffer
  data_in_.resize(fmap_i_.size());
  for (auto idx = 0U; idx < fmap_i_.size(); ++idx) {
    auto* cputb = inputs_.at("input").at(idx);
    auto* input_ptr = GET_CPUTB_DType_PTR(DType, cputb);
    if (0 == idx) {
      data_in_[idx] = input_ptr;
    } else {
      data_in_[idx] =
          reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
              fmap_i_[idx].num() * sizeof(DType), ALIGN_SIZE));
      FastPad(raw_fmap_i_[idx], fmap_i_[idx], pad_)
          .transform<DType>(input_ptr, data_in_[idx]);
    }
  }

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);
  std::fill_n(data_out_, fmap_o_.num(), DType(0));
}

template <typename DType>
void Correlation2dElemwise<DType>::run() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    corr_thread();
  } else {
    corr_normal();
  }
}

template <typename DType>
void Correlation2dElemwise<DType>::eltwise(std::uint64_t start_index,
                                           std::uint64_t end_index) {
  auto stride = 2 * pad_size_ + 1;
  for (auto i = start_index; i < end_index; ++i) {
    auto fmap = fmap_o_.pos2coord(i);
    auto h_offset = fmap.d / stride;
    auto w_offset = fmap.d % stride;
    auto addr_offset = fmap.n * fmap_i_[1].ncod() +
                       h_offset * fmap_i_[1].hcod() +
                       w_offset * fmap_i_[1].wcod();
    auto* data_in_1 = data_in_[1] + addr_offset;

    auto addr_in_0 = fmap.n * fmap_i_[0].ncod() + fmap.h * fmap_i_[0].hcod() +
                     fmap.w * fmap_i_[0].wcod() + fmap.c;
    auto addr_in_1 = fmap.n * fmap_i_[1].ncod() + fmap.h * fmap_i_[1].hcod() +
                     fmap.w * fmap_i_[1].wcod() + fmap.c;
    data_out_[i] = data_in_[0][addr_in_0] * data_in_1[addr_in_1];
  }
}

template <typename DType>
void Correlation2dElemwise<DType>::corr_normal() {
  eltwise(0, fmap_o_.num());
}

template <typename DType>
void Correlation2dElemwise<DType>::corr_thread() {
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

INSTANTIATE_TPCLASS(Correlation2dElemwise);
REG_OP_INSTANCE_FUNC("correlation2d_elemwise", Correlation2dElemwise);

}  // namespace cpu
}  // namespace vart
