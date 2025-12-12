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

#include "dwconv1d_base.hpp"

#include "align_buf_mgr.hpp"
#include "conv_2_gemm.hpp"
#include "cpu_gemm.hpp"
#include "fast_pad.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename WType>
const vector<string> DWConv1dBase<DType, WType>::ITName = {
    "input",
    "weights",
    "bias",
};

// constructor and deconstructor
template <typename DType, typename WType>
DWConv1dBase<DType, WType>::DWConv1dBase(const xir::Subgraph* subg,
    const xir::Op* op, IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {

  // get i/o fmap
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
  if(std::is_floating_point<DType>::value) {
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
void DWConv1dBase<DType, WType>::print_param() {
  if(!CPUCfg::Instance().debug())
    return;

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
void DWConv1dBase<DType, WType>::check_param() {
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
  UNI_LOG_CHECK(fmap_w_.n * fmap_w_.c == fmap_o_.c, VART_INVALID_VALUE)
      << ", " << fmap_w_.n << "*" << fmap_w_.c << " != " << fmap_o_.c;
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
void DWConv1dBase<DType, WType>::read() {
  read_input();
  read_weights();
  read_bias();

  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType, typename WType>
uint64_t DWConv1dBase<DType, WType>::get_workload() {
  // not consider batch
  ops_ = 2 * fmap_i_.c * kernel_.w;
  ops_ *= fmap_o_.w;

  return ops_;
}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::read_input() {
  // read img data
  auto* data_in_tb = inputs_.at(ITName[INPUT]).at(0);
  auto* tb_input_ptr = GET_CPUTB_DType_PTR(DType, data_in_tb);

  data_in_ptr_ = reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
      fmap_i_.num() * sizeof(DType), ALIGN_SIZE));

  FastPad(raw_fmap_i_, fmap_i_, pad_, dilation_)
    .transform<DType>(tb_input_ptr, data_in_ptr_);
}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::read_weights() {
  // weights order after read: oc, h, w, c
  auto* weights_tb = inputs_.at(ITName[WEIGHTS]).at(0);
  auto* tb_weights_ptr = GET_CPUTB_DType_PTR(WType, weights_tb);
  if(kernel_.dilation_w == 1) {
    weights_ptr_ = tb_weights_ptr;
    return;
  }

  weights_ptr_ = reinterpret_cast<WType*>(AlignBufMgr::Instance()
    ->allocate(fmap_w_.num() * sizeof(WType), ALIGN_SIZE));

  FastPad(raw_fmap_w_, fmap_w_, Pad_t{0, 0, 0, 0},
      Dilation_t{kernel_.dilation_h, kernel_.dilation_w})
    .transform<WType>(tb_weights_ptr, weights_ptr_);
}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::read_bias() {
  if (has_bias_) {
    auto* bias_tb = inputs_.at(ITName[BIAS]).at(0);
    auto* tb_bias_ptr = GET_CPUTB_DType_PTR(WType, bias_tb);
    bias_ptr_ = tb_bias_ptr;
  } else {
    bias_ptr_ = nullptr;
  }

}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::dwconv() {
  if(CPU_RUN_MODE == CPURunMode::NORMAL) {
    dwconv_normal();
  } else {
    dwconv_normal_thread();
  }
}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::dwconv_normal() {
  for (auto n = 0; n < fmap_o_.n; n++) {
    for (auto h = 0; h < fmap_o_.h; h++) {
      for (auto w = 0; w < fmap_o_.w; w++) {
        dwconv_one(n, h, w);
      }
    }
  }
}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::dwconv_normal_thread() {
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for(auto i=0U; i<THREAD_NUM; i++) {
    // thr_fut[i] = std::async(std::launch::async | std::launch::deferred,
    thr_fut[i] = std::async(std::launch::async,
        [this](decltype(i) i) {
      auto BASE_POS = i*THREAD_WORKLOAD;
      auto FMAP_SIZE = fmap_o_.num();
      for(auto j=0U; j<THREAD_WORKLOAD; j++) {
        int pos = BASE_POS + j;
        if(pos >= FMAP_SIZE)
          break;

        if(pos % fmap_o_.c == 0) {
          auto fmap = fmap_o_.pos2coord(pos);
          dwconv_one(fmap.n, fmap.h, fmap.w);
        }
      }

      return 0;
    }, i);
  }

  for(auto i=0U; i<THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::dwconv_one(int idx_dst_n, int idx_dst_h,
                                   int idx_dst_w) {
  vector<DType> result(fmap_o_.c, 0);
  int idx_src_n = idx_dst_n;
  int idx_src_h = idx_dst_h * stride_.h;
  int idx_src_w = idx_dst_w * stride_.w;

  for (int c = 0; c < fmap_w_.c; c++) {
    for (auto n = 0; n < fmap_w_.n; n++) {
      auto rlt_pos = c * fmap_w_.n + n;

      for (int h = 0; h < fmap_w_.h; h++) {
        for (int w = 0; w < fmap_w_.w; w++) {
          auto img_addr = idx_src_n * fmap_i_.h * fmap_i_.w * fmap_i_.c +
                          (idx_src_h + h) * fmap_i_.w * fmap_i_.c +
                          (idx_src_w + w) * fmap_i_.c + c;

          auto weights_addr = n * fmap_w_.h * fmap_w_.w * fmap_w_.c +
                              h * fmap_w_.w * fmap_w_.c + w * fmap_w_.c + c;

          result[rlt_pos] +=
              data_in_ptr_[img_addr] * weights_ptr_[weights_addr];
        }
      }
    }
  }

  // assign conv value to data_out_ptr_ var
  auto rlt_addr = idx_dst_n * fmap_o_.h * fmap_o_.w * fmap_o_.c +
                  idx_dst_h * fmap_o_.w * fmap_o_.c + idx_dst_w * fmap_o_.c;
  for (auto i = 0; i < fmap_o_.c; i++) {
    data_out_ptr_[rlt_addr + i] = result[i];
  }
}

template <typename DType, typename WType>
void DWConv1dBase<DType, WType>::bias() {
  if(!has_bias_)
    return;

  // NOTE: special handling for conv-fix
  if(std::is_floating_point<DType>::value) {
    for(auto i=0; i<fmap_o_.num(); i++) {
      auto pos = i % fmap_o_.c;
      data_out_ptr_[i] += bias_ptr_[pos];
    }
  } else {
    UNI_LOG_FATAL(VART_EXEC_ERROR)
        << "Pls do not use dwconv_base's bias while do dwconv-fix!" << endl;
  }
}

// constructor and deconstructor
template <typename DType, typename WType>
DWConv1d<DType, WType>::DWConv1d(const xir::Subgraph* subg, const xir::Op* op,
                      IMapTBs_t inputs, CPUTBPtr_t output)
    : DWConv1dBase<DType, WType>(subg, op, inputs, output) {
  // get kernel info
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
void DWConv1d<DType, WType>::run() {
  this->dwconv();
  this->bias();
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(DWConv1d);
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("depthwise-conv1d", DWConv1d, "weights");

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(DWConv1dBase);

}  // namespace cpu
}  // namespace vart
