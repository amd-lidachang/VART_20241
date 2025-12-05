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

#include "conv1d_base.hpp"

#include "align_buf_mgr.hpp"
#include "conv_2_gemm.hpp"
#include "cpu_gemm.hpp"
#include "fast_pad.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename WType>
const vector<string> Conv1dBase<DType, WType>::ITName = {
    "input",
    "weights",
    "bias",
};

// constructor and deconstructor
template <typename DType, typename WType>
Conv1dBase<DType, WType>::Conv1dBase(const xir::Subgraph* subg,
                                     const xir::Op* op, IMapTBs_t inputs,
                                     CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  // get i/o fmap, treat conv1d as conv2d with h dimension of 1.
  GET_INPUT_DIMX_FMAP(raw_fmap_i_, input, 3);
  auto raw_fmap_i__h = raw_fmap_i_.n;
  raw_fmap_i_.n = raw_fmap_i_.h;
  raw_fmap_i_.h = raw_fmap_i__h;
  fmap_i_ = raw_fmap_i_;
  GET_INPUT_DIMX_FMAP(raw_fmap_w_, weights, 3);
  auto raw_fmap_real_h = raw_fmap_w_.n;
  raw_fmap_w_.n = raw_fmap_w_.h;
  raw_fmap_w_.h = raw_fmap_real_h;
  fmap_w_ = raw_fmap_w_;
  has_bias_ = xir_op_->get_input_num(ITName[BIAS]) > 0 ? true : false;
  if (has_bias_) {
    GET_INPUT_DIM1_FMAP(fmap_b_, bias);
  }

  GET_OUTPUT_DIMX_FMAP(fmap_o_, 3);
  auto fmap_o_real_h = fmap_o_.n;
  fmap_o_.n = fmap_o_.h;
  fmap_o_.h = fmap_o_real_h;
  // NOTE+TODO: in fix mode, xy force it to be 0, not know why now
  std::string xir_pad_mode = "FLOOR";
  if (std::is_floating_point<DType>::value) {
    xir_pad_mode = CPUOPBase::xir_op_->get_attr<std::string>("pad_mode");
  }
  if (CPUOPBase::xir_op_->has_attr("pad")) {
    auto xir_pad = CPUOPBase::xir_op_->get_attr<vector<int>>("pad");
    pad_ = Pad_t{xir_pad_mode, xir_pad[0], 0, xir_pad[1], 0};
  } else {
    pad_ = Pad_t{xir_pad_mode, 0, 0, 0, 0};
  }

  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::print_param() {
  if (!CPUCfg::Instance().debug()) return;

  raw_fmap_i_.print_param("raw_fmap_i");
  fmap_i_.print_param("fmap_i");
  raw_fmap_w_.print_param("raw_fmap_w");
  fmap_w_.print_param("fmap_w");
  UNI_LOG_DEBUG_INFO << "has_bias: " << Bool2Str(has_bias_) << endl;
  fmap_b_.print_param("fmap_b");
  fmap_o_.print_param("fmap_o");

  pad_.print_param("pad");
  kernel_.print_param("kernel");
  stride_.print_param("stride");
  dilation_.print_param("dilation");

  if (CPU_RUN_MODE != CPURunMode::NORMAL) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() >= 2 && inputs_.size() <= 3, VART_SIZE_ERROR)
      << ", Err: inputs size is " << inputs_.size();

  // check fmap_out height
  int dst_w = 0;
  if (pad_.mode == PadAttr::PAD_MODE_FLOOR) {
    dst_w = floor(1.0 * (fmap_i_.w - kernel_.w + stride_.w) / stride_.w);
  } else if (pad_.mode == PadAttr::PAD_MODE_CEIL) {
    dst_w = ceil(1.0 * (fmap_i_.w - kernel_.w + stride_.w) / stride_.w);
  } else if (pad_.mode == PadAttr::PAD_MODE_SAME) {
    dst_w = ceil(1.0 * raw_fmap_i_.w / stride_.w);
  } else if (pad_.mode == PadAttr::PAD_MODE_VALID) {
    dst_w = ceil(1.0 * (fmap_i_.w - kernel_.w + 1.0) / stride_.w);
  } else {
    UNI_LOG_ERROR(VART_NOT_SUPPORT)
        << "Not support pad mode " << pad_.mode << endl;
    abort();
  }

  UNI_LOG_CHECK(fmap_o_.h == 1, VART_INVALID_VALUE)
      << ", " << fmap_o_.h << " != " << 1;
  UNI_LOG_CHECK(dst_w == fmap_o_.w, VART_INVALID_VALUE)
      << ", " << dst_w << " != " << fmap_o_.w;

  // check weights/bias fmap
  UNI_LOG_CHECK(fmap_w_.n == fmap_o_.c, VART_INVALID_VALUE)
      << ", " << fmap_w_.n << " != " << fmap_o_.c;
  UNI_LOG_CHECK(fmap_w_.h == kernel_.h, VART_INVALID_VALUE)
      << ", " << fmap_w_.h << " != " << kernel_.h;
  UNI_LOG_CHECK(fmap_w_.w == kernel_.w, VART_INVALID_VALUE)
      << ", " << fmap_w_.w << " != " << kernel_.w;
  UNI_LOG_CHECK(fmap_w_.c == fmap_i_.c, VART_INVALID_VALUE)
      << ", " << fmap_w_.c << " != " << fmap_i_.c;

  if (has_bias_) {
    UNI_LOG_CHECK(fmap_b_.c == fmap_o_.c, VART_INVALID_VALUE)
        << ", " << fmap_b_.c << " != " << fmap_o_.c;
  }
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::read() {
  read_input();
  read_weights();
  read_bias();

  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);

  std::fill_n(data_out_ptr_, fmap_o_.num(), DType(0));
}

template <typename DType, typename WType>
uint64_t Conv1dBase<DType, WType>::get_workload() {
  // not consider batch
  ops_ = 2 * fmap_i_.c * kernel_.w * fmap_o_.c;
  ops_ *= fmap_o_.w;

  return ops_;
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::read_input() {
  // read img data
  auto* data_in_tb = inputs_.at(ITName[INPUT]).at(0);
  auto* tb_input_ptr = GET_CPUTB_DType_PTR(DType, data_in_tb);

  data_in_ptr_ = reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
      fmap_i_.num() * sizeof(DType), ALIGN_SIZE));

  FastPad(raw_fmap_i_, fmap_i_, pad_, dilation_)
      .transform<DType>(tb_input_ptr, data_in_ptr_);
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::read_weights() {
  // weights order after read: oc, h, w, c
  auto* weights_tb = inputs_.at(ITName[WEIGHTS]).at(0);
  auto* tb_weights_ptr = GET_CPUTB_DType_PTR(WType, weights_tb);
  if (kernel_.dilation_w == 1) {
    weights_ptr_ = tb_weights_ptr;
    return;
  }

  weights_ptr_ = reinterpret_cast<WType*>(AlignBufMgr::Instance()->allocate(
      fmap_w_.num() * sizeof(WType), ALIGN_SIZE));

  FastPad(raw_fmap_w_, fmap_w_, Pad_t{0, 0, 0, 0},
          Dilation_t{kernel_.dilation_h, kernel_.dilation_w})
      .transform<WType>(tb_weights_ptr, weights_ptr_);
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::read_bias() {
  if (has_bias_) {
    auto* bias_tb = inputs_.at(ITName[BIAS]).at(0);
    auto* tb_bias_ptr = GET_CPUTB_DType_PTR(WType, bias_tb);
    bias_ptr_ = tb_bias_ptr;
  } else {
    bias_ptr_ = nullptr;
  }
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::conv() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL) {
    conv_normal();
  } else if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD) {
    conv_normal_thread();
  } else if (CPU_RUN_MODE == CPURunMode::GEMM) {
    conv_gemm();
  } else if (CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    conv_gemm_thread();
  } else {
    conv_normal();
  }
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::conv_normal() {
  for (auto n = 0; n < fmap_o_.n; n++) {
    for (auto h = 0; h < fmap_o_.h; h++) {
      for (auto w = 0; w < fmap_o_.w; w++) {
        for (auto c = 0; c < fmap_o_.c; c++) {
          conv_one(n, h, w, c);
        }
      }
    }
  }
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::conv_normal_thread() {
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    // thr_fut[i] = std::async(std::launch::async | std::launch::deferred,
    thr_fut[i] = std::async(
        std::launch::async,
        [this](decltype(i) i) {
          auto BASE_POS = i * THREAD_WORKLOAD;
          auto FMAP_SIZE = fmap_o_.num();
          for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
            int pos = BASE_POS + j;
            if (pos >= FMAP_SIZE) break;

            auto fmap = fmap_o_.pos2coord(pos);
            conv_one(fmap.n, fmap.h, fmap.w, fmap.c);
          }

          return 0;
        },
        i);
  }

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::conv_gemm() {
  auto WEIGHTS_BATCH_SIZE = fmap_w_.h * fmap_w_.w * fmap_w_.c;

  FMap_t tmp_fmap{fmap_o_.n, fmap_o_.h, fmap_o_.w, WEIGHTS_BATCH_SIZE};
  DType* tmp_ptr = reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
      tmp_fmap.num() * sizeof(DType), ALIGN_SIZE));

  Conv2Gemm(fmap_i_, tmp_fmap, fmap_w_, kernel_, stride_)
      .transform(data_in_ptr_, tmp_ptr);

  matmul<DType, WType>(tmp_ptr, weights_ptr_, data_out_ptr_,
                       fmap_o_.n * fmap_o_.h * fmap_o_.w, fmap_o_.c,
                       WEIGHTS_BATCH_SIZE);

  AlignBufMgr::Instance()->release(tmp_ptr);
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::conv_gemm_thread() {
#if 0
  auto WEIGHTS_BATCH_SIZE = fmap_w_.h * fmap_w_.w * fmap_w_.c;

  FMap_t tmp_fmap{fmap_o_.n, fmap_o_.h, fmap_o_.w, WEIGHTS_BATCH_SIZE};
  DType* tmp_ptr = reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
      tmp_fmap.num() * sizeof(DType), ALIGN_SIZE));

  Conv2Gemm(fmap_i_, tmp_fmap, fmap_w_, kernel_, stride_)
      .transform(data_in_ptr_, tmp_ptr);

  matmul_thread<DType, WType>(tmp_ptr, weights_ptr_, data_out_ptr_,
                       fmap_o_.n * fmap_o_.h * fmap_o_.w, fmap_o_.c,
                       WEIGHTS_BATCH_SIZE);

  AlignBufMgr::Instance()->release(tmp_ptr);
#endif

#if 1
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    // thr_fut[i] = std::async(std::launch::async | std::launch::deferred,
    thr_fut[i] = std::async(
        std::launch::async,
        [this](decltype(i) i) {
          auto BASE_POS = i * THREAD_WORKLOAD;
          auto FMAP_SIZE = fmap_o_.num();
          for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
            int pos = BASE_POS + j;
            if (pos >= FMAP_SIZE) break;

            auto fmap = fmap_o_.pos2coord(pos);

            auto* tmp_ptr_i = data_in_ptr_ + fmap.n * fmap_i_.ncod() +
                              fmap.h * stride_.h * fmap_i_.hcod() +
                              fmap.w * stride_.w * fmap_i_.wcod();
            auto* tmp_ptr_w = weights_ptr_ + fmap.c * fmap_w_.ncod();

            inner_product_with_kernel_stride<DType, WType>(
                tmp_ptr_i, tmp_ptr_w, &data_out_ptr_[pos], fmap_i_, fmap_w_,
                kernel_);
          }

          return 0;
        },
        i);
  }

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
#endif
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::conv_one(int idx_dst_n, int idx_dst_h,
                                        int idx_dst_l, int idx_oc) {
  // assign conv value to data_out_ptr_ var
  auto rlt_addr = idx_dst_n * fmap_o_.ncod() + idx_dst_h * fmap_o_.hcod() +
                  idx_dst_l * fmap_o_.wcod() + idx_oc;

  // NOTE: special handling for conv-fix
  data_out_ptr_[rlt_addr] = 0;

  auto idx_src_h = idx_dst_h * stride_.h;
  auto idx_src_w = idx_dst_l * stride_.w;

  auto addr_base_i = idx_dst_n * fmap_i_.ncod() + idx_src_h * fmap_i_.hcod() +
                     idx_src_w * fmap_i_.wcod();
  auto addr_base_w = idx_oc * fmap_w_.ncod();

  // calculate conv value
  for (auto h = 0; h < kernel_.h; ++h) {
    for (auto w = 0; w < kernel_.w; ++w) {
      auto* tmp_ptr_i =
          data_in_ptr_ + addr_base_i + h * fmap_i_.hcod() + w * fmap_i_.wcod();
      auto* tmp_ptr_w =
          weights_ptr_ + addr_base_w + h * fmap_w_.hcod() + w * fmap_w_.wcod();
      for (auto c = 0; c < fmap_i_.c; ++c) {
        data_out_ptr_[rlt_addr] += tmp_ptr_i[c] * tmp_ptr_w[c];
      }
    }
  }
}

template <typename DType, typename WType>
void Conv1dBase<DType, WType>::bias() {
  if (!has_bias_) return;

  // NOTE: special handling for conv-fix
  if (std::is_floating_point<DType>::value) {
    for (auto i = 0; i < fmap_o_.num(); i++) {
      auto pos = i % fmap_o_.c;
      data_out_ptr_[i] += bias_ptr_[pos];
    }
  } else {
    UNI_LOG_FATAL(VART_EXEC_ERROR)
        << "Pls do not use conv_base's bias while do conv-fix!" << endl;
  }
}

// constructor and deconstructor
template <typename DType, typename WType>
Conv1d<DType, WType>::Conv1d(const xir::Subgraph* subg, const xir::Op* op,
                             IMapTBs_t inputs, CPUTBPtr_t output)
    : Conv1dBase<DType, WType>(subg, op, inputs, output) {
  // get kernel and update fmap_w_
  auto xir_kernel = CPUOPBase::xir_op_->get_attr<vector<int>>("kernel");
  auto xir_dilation = CPUOPBase::xir_op_->get_attr<vector<int>>("dilation");
  auto dilation = Dilation_t{1, xir_dilation[0]};
  kernel_ = Kernel_t{1, xir_kernel[0], dilation};
  fmap_w_.w = fmap_w_.w * kernel_.dilation_w - (kernel_.dilation_w - 1);

  // get stride info
  auto xir_stride = CPUOPBase::xir_op_->get_attr<vector<int>>("stride");
  stride_ = Stride_t{1, xir_stride[0]};

  // update input height and width according to padding info
  pad_ = calc_pad(fmap_i_, fmap_o_, kernel_, stride_, pad_);

  fmap_i_.w += pad_.l + pad_.r;
}

template <typename DType, typename WType>
void Conv1d<DType, WType>::run() {
  // // do necessary check
  // this->print_param();
  // this->check_param();

  // // read data
  // this->read();

  // calc conv
  this->conv();

  // add bias to conv result
  this->bias();

  // // do save, debug...
  // this->save();
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(Conv1dBase);
INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(Conv1d);

REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("conv1d", Conv1d, "weights");

}  // namespace cpu
}  // namespace vart
