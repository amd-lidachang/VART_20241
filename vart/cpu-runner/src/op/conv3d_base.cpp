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

#include "conv3d_base.hpp"

#include "align_buf_mgr.hpp"
#include "conv_2_gemm.hpp"
#include "cpu_gemm.hpp"
#include "fast_pad.hpp"

using std::begin;
using std::end;

namespace vart::cpu {

// constructor and deconstructor
template <typename DType, typename WType>
Conv3dBase<DType, WType>::Conv3dBase(const xir::Subgraph* subg,
                                     const xir::Op* op, IMapTBs_t inputs,
                                     CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  // get i/o fmap
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 5);
  GET_INPUT_DIMX_FMAP(fmap_w_, weights, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  auto xir_dilation_w_ = op->get_attr<std::vector<std::int32_t>>("dilation");
  dilation_w_ =
      Dilation_t{xir_dilation_w_[1], xir_dilation_w_[0], xir_dilation_w_[2]};
  auto xir_kernel = op->get_attr<std::vector<std::int32_t>>("kernel");
  kernel_ = Kernel_t{xir_kernel[1], xir_kernel[0], xir_kernel[2], dilation_w_};
  auto xir_pad = op->get_attr<std::vector<std::int32_t>>("pad");
  pad_ = Pad_t{xir_pad[0], xir_pad[2], xir_pad[4],
               xir_pad[1], xir_pad[3], xir_pad[5]};
  if (op->get_type() == "conv3d" || op->get_type() == "conv3d-fix") {
    auto xir_stride = op->get_attr<std::vector<std::int32_t>>("stride");
    stride_ = Stride_t{xir_stride[1], xir_stride[0], xir_stride[2]};
  } else if (op->get_type() == "depthwise-conv3d" ||
             op->get_type() == "depthwise-conv3d-fix") {
    auto xir_stride = op->get_attr<std::vector<std::int32_t>>("stride");
    stride_ = Stride_t{xir_stride[1], xir_stride[0], xir_stride[2]};
    if_depthwise_ = true;
  } else if (op->get_type() == "transposed-conv3d" ||
             op->get_type() == "transposed-conv3d-fix") {
    auto xir_dilation_f_ = op->get_attr<std::vector<std::int32_t>>("stride");
    dilation_f_ =
        Dilation_t{xir_dilation_f_[1], xir_dilation_f_[0], xir_dilation_f_[2]};
    auto l = kernel_.w - 1 - pad_.l;
    auto t = kernel_.h - 1 - pad_.t;
    auto r = kernel_.w - 1 - pad_.r;
    auto b = kernel_.h - 1 - pad_.b;
    auto near = kernel_.d - 1 - pad_.n;
    auto far = kernel_.d - 1 - pad_.f;
    pad_ = Pad_t{l, t, near, r, b, far};
  } else if (op->get_type() == "transposed-depthwise-conv3d" ||
             op->get_type() == "transposed-depthwise-conv3d-fix") {
    auto xir_dilation_f_ = op->get_attr<std::vector<std::int32_t>>("stride");
    dilation_f_ =
        Dilation_t{xir_dilation_f_[1], xir_dilation_f_[0], xir_dilation_f_[2]};
    auto l = kernel_.w - 1 - pad_.l;
    auto t = kernel_.h - 1 - pad_.t;
    auto r = kernel_.w - 1 - pad_.r;
    auto b = kernel_.h - 1 - pad_.b;
    auto near = kernel_.d - 1 - pad_.n;
    auto far = kernel_.d - 1 - pad_.f;
    pad_ = Pad_t{l, t, near, r, b, far};
    if_depthwise_ = true;
  }
  std::vector<std::int32_t> pad_i_shape_;
  pad_i_shape_.push_back(fmap_i_.n);
  pad_i_shape_.push_back((fmap_i_.h - 1) * dilation_f_.h + 1 + pad_.t + pad_.b);
  pad_i_shape_.push_back((fmap_i_.w - 1) * dilation_f_.w + 1 + pad_.l + pad_.r);
  pad_i_shape_.push_back((fmap_i_.d - 1) * dilation_f_.d + 1 + pad_.n + pad_.f);
  pad_i_shape_.push_back(fmap_i_.c);
  fmap_pad_i_ = FMap_t{pad_i_shape_};

  fmap_dilated_w_ =
      FMap_t{fmap_w_.n, kernel_.h, kernel_.w, kernel_.d, fmap_w_.c};

  has_bias_ = op->get_input_ops("bias").size();
  if (op->has_attr("group")) {
    group_ = op->get_attr<int32_t>("group");
  } else {
    group_ = 1;
  }

  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD =
      (fmap_o_.num() / THREAD_NUM) + (fmap_o_.num() % THREAD_NUM > 0);
}

// TODO
template <typename DType, typename WType>
void Conv3dBase<DType, WType>::print_param() {
  if (!CPUCfg::Instance().debug()) return;
  pad_.print_param("pad");
  kernel_.print_param("kernel");
  if (CPU_RUN_MODE != CPURunMode::NORMAL) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
}

// TODO
template <typename DType, typename WType>
void Conv3dBase<DType, WType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() >= 2 && inputs_.size() <= 3, VART_SIZE_ERROR)
      << ", Err: inputs size is " << inputs_.size();

  UNI_LOG_CHECK(fmap_w_.n == fmap_o_.c, VART_INVALID_VALUE)
      << ", " << fmap_w_.n << " != " << fmap_o_.c;
  UNI_LOG_CHECK(fmap_dilated_w_.h == kernel_.h, VART_INVALID_VALUE)
      << ", " << fmap_dilated_w_.h << " != " << kernel_.h;
  UNI_LOG_CHECK(fmap_dilated_w_.w == kernel_.w, VART_INVALID_VALUE)
      << ", " << fmap_dilated_w_.w << " != " << kernel_.w;
  UNI_LOG_CHECK(fmap_dilated_w_.d == kernel_.d, VART_INVALID_VALUE)
      << ", " << fmap_dilated_w_.d << " != " << kernel_.d;
  UNI_LOG_CHECK(fmap_w_.c == fmap_i_.c, VART_INVALID_VALUE)
      << ", " << fmap_w_.c << " != " << fmap_i_.c;
  UNI_LOG_CHECK(group_ == 1, VART_NOT_SUPPORT)
      << "group convolution 3d is not supported.";
}

template <typename DType, typename WType>
void Conv3dBase<DType, WType>::read() {
  // read img data
  auto* data_in_tb = inputs_.at("input").at(0);
  auto* tb_input_ptr = GET_CPUTB_DType_PTR(DType, data_in_tb);

  data_in_ptr_ = reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
      fmap_pad_i_.num() * sizeof(DType), ALIGN_SIZE));

  FastPad(fmap_i_, fmap_pad_i_, pad_, dilation_f_)
      .transform<DType>(tb_input_ptr, data_in_ptr_);

  // read weights
  auto* weights_tb = inputs_.at("weights").at(0);
  auto* tb_weights_ptr = GET_CPUTB_DType_PTR(WType, weights_tb);
  weights_ptr_ = reinterpret_cast<WType*>(AlignBufMgr::Instance()->allocate(
      fmap_dilated_w_.num() * sizeof(WType), ALIGN_SIZE));

  FastPad(fmap_w_, fmap_dilated_w_, Pad_t{0, 0, 0, 0, 0, 0}, dilation_w_)
      .transform<WType>(tb_weights_ptr, weights_ptr_);

  // read bias
  if (has_bias_) {
    auto* bias_tb = inputs_.at("bias").at(0);
    bias_ptr_ = GET_CPUTB_DType_PTR(WType, bias_tb);
  } else {
    bias_ptr_ = nullptr;
  }

  // output data
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
  std::fill_n(data_out_ptr_, fmap_o_.num(), DType(0));
}

// TODO
template <typename DType, typename WType>
uint64_t Conv3dBase<DType, WType>::get_workload() {
  return 0;
}

template <typename DType, typename WType>
void Conv3dBase<DType, WType>::conv() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD) {
    conv_normal_thread();
  } else if (CPU_RUN_MODE == CPURunMode::GEMM) {
    conv_gemm();
  } else if (CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    if (if_depthwise_)
      conv_normal_thread();
    else
      conv_gemm_thread();
  } else {
    conv_normal();
  }
}

template <typename DType, typename WType>
void Conv3dBase<DType, WType>::conv_normal() {
  for (auto n = 0; n < fmap_o_.n; n++)
    for (auto h = 0; h < fmap_o_.h; h++)
      for (auto w = 0; w < fmap_o_.w; w++)
        for (auto d = 0; d < fmap_o_.d; d++)
          for (auto c = 0; c < fmap_o_.c; c++) {
            if (!if_depthwise_)
              conv_one(n, h, w, d, c);
            else
              dwconv_one(n, h, w, d, c);
          }
}

template <typename DType, typename WType>
void Conv3dBase<DType, WType>::conv_normal_thread() {
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i] = std::async(
        std::launch::async,
        [this](decltype(i) i) {
          auto BASE_POS = i * THREAD_WORKLOAD;
          auto FMAP_SIZE = fmap_o_.num();
          for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
            int pos = BASE_POS + j;
            if (pos >= FMAP_SIZE) break;
            auto fmap = fmap_o_.pos2coord(pos);
            if (!if_depthwise_)
              conv_one(fmap.n, fmap.h, fmap.w, fmap.d, fmap.c);
            else
              dwconv_one(fmap.n, fmap.h, fmap.w, fmap.d, fmap.c);
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
void Conv3dBase<DType, WType>::conv_gemm() {
  auto WEIGHTS_BATCH_SIZE = fmap_dilated_w_.ncod();

  FMap_t tmp_fmap{fmap_o_.n, fmap_o_.h, fmap_o_.w, fmap_o_.d,
                  WEIGHTS_BATCH_SIZE};
  DType* tmp_ptr = reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
      tmp_fmap.num() * sizeof(DType), ALIGN_SIZE));

  Conv2Gemm(fmap_pad_i_, tmp_fmap, fmap_dilated_w_, kernel_, stride_)
      .transform(data_in_ptr_, tmp_ptr);

  matmul<DType, WType>(tmp_ptr, weights_ptr_, data_out_ptr_,
                       fmap_o_.num() / fmap_o_.c, fmap_o_.c,
                       WEIGHTS_BATCH_SIZE);

  AlignBufMgr::Instance()->release(tmp_ptr);
}

template <typename DType, typename WType>
void Conv3dBase<DType, WType>::conv_gemm_thread() {
#if 0
  auto WEIGHTS_BATCH_SIZE = fmap_dilated_w_.ncod();

  FMap_t tmp_fmap{fmap_o_.n, fmap_o_.h, fmap_o_.w, fmap_o_.d,
                  WEIGHTS_BATCH_SIZE};

  auto tmp_ptr = reinterpret_cast<DType*>(AlignBufMgr::Instance()->allocate(
      tmp_fmap.num() * sizeof(DType), ALIGN_SIZE));

  UNI_LOG_CHECK(tmp_ptr != nullptr, VART_NULL_PTR);

  Conv2Gemm(fmap_pad_i_, tmp_fmap, fmap_dilated_w_, kernel_, stride_)
      .transform(data_in_ptr_, tmp_ptr);

  matmul_thread<DType, WType>(tmp_ptr, weights_ptr_, data_out_ptr_,
                              fmap_o_.n * fmap_o_.h * fmap_o_.w * fmap_o_.d,
                              fmap_o_.c, WEIGHTS_BATCH_SIZE);

  AlignBufMgr::Instance()->release(tmp_ptr);
#endif

#if 1
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i] = std::async(
        std::launch::async,
        [this](decltype(i) i) {
          auto BASE_POS = i * THREAD_WORKLOAD;
          auto FMAP_SIZE = fmap_o_.num();
          for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
            int pos = BASE_POS + j;
            if (pos >= FMAP_SIZE) break;
            auto fmap = fmap_o_.pos2coord(pos);

            auto* tmp_ptr_i = data_in_ptr_ + fmap.n * fmap_pad_i_.ncod() +
                              fmap.h * stride_.h * fmap_pad_i_.hcod() +
                              fmap.w * stride_.w * fmap_pad_i_.wcod() +
                              fmap.d * stride_.d * fmap_pad_i_.dcod();
            auto* tmp_ptr_w = weights_ptr_ + fmap.c * fmap_dilated_w_.ncod();

            inner_product_with_kernel_stride<DType, WType>(
                tmp_ptr_i, tmp_ptr_w, &data_out_ptr_[pos], fmap_pad_i_,
                fmap_dilated_w_, kernel_);
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
void Conv3dBase<DType, WType>::conv_one(int idx_on, int idx_oh, int idx_ow,
                                        int idx_od, int idx_oc) {
  auto o_addr = idx_on * fmap_o_.ncod() + idx_oh * fmap_o_.hcod() +
                idx_ow * fmap_o_.wcod() + idx_od * fmap_o_.dcod() + idx_oc;

  auto idx_iw = idx_ow * stride_.w;
  auto idx_ih = idx_oh * stride_.h;
  auto idx_id = idx_od * stride_.d;

  auto addr_base_i = idx_on * fmap_pad_i_.ncod() + idx_ih * fmap_pad_i_.hcod() +
                     idx_iw * fmap_pad_i_.wcod() + idx_id * fmap_pad_i_.dcod();
  auto addr_base_w = idx_oc * fmap_w_.ncod();

  // calculate conv value
  for (auto h = 0; h < kernel_.h; ++h) {
    for (auto w = 0; w < kernel_.w; ++w) {
      for (auto d = 0; d < kernel_.d; ++d) {
        auto* tmp_ptr_i = data_in_ptr_ + addr_base_i + h * fmap_pad_i_.hcod() +
                          w * fmap_pad_i_.wcod() + d * fmap_pad_i_.dcod();
        auto* tmp_ptr_w = weights_ptr_ + addr_base_w + h * fmap_w_.hcod() +
                          w * fmap_w_.wcod() + d * fmap_w_.dcod();
        for (auto c = 0; c < fmap_i_.c; ++c) {
          data_out_ptr_[o_addr] += tmp_ptr_i[c] * tmp_ptr_w[c];
        }
      }
    }
  }
}

template <typename DType, typename WType>
void Conv3dBase<DType, WType>::dwconv_one(int idx_on, int idx_oh, int idx_ow,
                                          int idx_od, int idx_oc) {
  auto addr_o = idx_on * fmap_o_.ncod() + idx_oh * fmap_o_.hcod() +
                idx_ow * fmap_o_.wcod() + idx_od * fmap_o_.dcod() + idx_oc;

  auto idx_iw = idx_ow * stride_.w;
  auto idx_ih = idx_oh * stride_.h;
  auto idx_id = idx_od * stride_.d;

  auto addr_base_i = idx_on * fmap_pad_i_.ncod() + idx_ih * fmap_pad_i_.hcod() +
                     idx_iw * fmap_pad_i_.wcod() + idx_id * fmap_pad_i_.dcod() +
                     idx_oc;
  auto addr_base_w = idx_oc;

  for (auto h = 0; h < fmap_dilated_w_.h; ++h) {
    for (auto w = 0; w < fmap_dilated_w_.w; ++w) {
      for (auto d = 0; d < fmap_dilated_w_.d; ++d) {
        auto addr_i = addr_base_i + h * fmap_pad_i_.hcod() +
                      w * fmap_pad_i_.wcod() + d * fmap_pad_i_.dcod();
        auto addr_w = addr_base_w + h * fmap_dilated_w_.hcod() +
                      w * fmap_dilated_w_.wcod() + d * fmap_dilated_w_.dcod();
        data_out_ptr_[addr_o] += data_in_ptr_[addr_i] * weights_ptr_[addr_w];
      }
    }
  }
}

template <typename DType, typename WType>
void Conv3dBase<DType, WType>::bias() {
  if (!has_bias_) return;

  if (std::is_floating_point<DType>::value) {
    for (auto i = 0; i < fmap_o_.num(); i++) {
      auto pos = i % fmap_o_.dcod();
      data_out_ptr_[i] += bias_ptr_[pos];
    }
  } else {
    UNI_LOG_FATAL(VART_EXEC_ERROR)
        << "Pls do not use conv_base's bias while do conv-fix!" << endl;
  }
}

template <typename DType, typename WType>
Conv3d<DType, WType>::Conv3d(const xir::Subgraph* subg, const xir::Op* op,
                             IMapTBs_t inputs, CPUTBPtr_t output)
    : Conv3dBase<DType, WType>(subg, op, inputs, output) {}

template <typename DType, typename WType>
void Conv3d<DType, WType>::run() {
  this->conv();
  this->bias();
}

template <typename DType, typename WType>
Conv3dFix<DType, WType>::Conv3dFix(const xir::Subgraph* subg, const xir::Op* op,
                                   IMapTBs_t inputs, CPUTBPtr_t output)
    : Conv3dBase<DType, WType>(subg, op, inputs, output) {
  fix_point_i_ = op->get_input_ops("input")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  fix_point_w_ = op->get_input_ops("weights")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  fix_point_b_ =
      op->get_input_ops("bias")[0]->get_output_tensor()->get_attr<std::int32_t>(
          "fix_point");
  fix_point_o_ = op->get_output_tensor()->get_attr<std::int32_t>("fix_point");
  shift_bias_ = fix_point_i_ + fix_point_w_ - fix_point_b_;
  shift_cut_ = fix_point_i_ + fix_point_w_ - fix_point_o_;
  act_type_ = op->get_attr<std::string>("nonlinear");
  std::vector<std::string> supported_{"NONE",      "RELU",     "RELU6",
                                      "LEAKYRELU", "HSIGMOID", "HSWISH"};
  UNI_LOG_CHECK(std::find(supported_.begin(), supported_.end(), act_type_) !=
                    supported_.end(),
                VART_INVALID_VALUE)
      << "we do not activation type: " << act_type_ << " in conv3d-fix.";
  if (act_type_ == "HSIGMOID") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
  } else if (act_type_ == "HSWISH") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
    shift_hswish_ = op->get_attr<int>("shift_hswish");
  }
}

template <typename DType, typename WType>
DWConv3dFix<DType, WType>::DWConv3dFix(const xir::Subgraph* subg,
                                       const xir::Op* op, IMapTBs_t inputs,
                                       CPUTBPtr_t output)
    : Conv3dBase<DType, WType>(subg, op, inputs, output) {
  fix_point_i_ = op->get_input_ops("input")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  fix_point_w_ = op->get_input_ops("weights")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  fix_point_b_ =
      op->get_input_ops("bias")[0]->get_output_tensor()->get_attr<std::int32_t>(
          "fix_point");
  fix_point_o_ = op->get_output_tensor()->get_attr<std::int32_t>("fix_point");
  shift_bias_ = fix_point_i_ + fix_point_w_ - fix_point_b_;
  shift_cut_ = fix_point_i_ + fix_point_w_ - fix_point_o_;
  act_type_ = op->get_attr<std::string>("nonlinear");
  std::vector<std::string> supported_{"NONE",      "RELU",     "RELU6",
                                      "LEAKYRELU", "HSIGMOID", "HSWISH"};
  UNI_LOG_CHECK(std::find(supported_.begin(), supported_.end(), act_type_) !=
                    supported_.end(),
                VART_INVALID_VALUE)
      << "we do not activation type: " << act_type_ << " in conv3d-fix.";
  if (act_type_ == "HSIGMOID") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
  } else if (act_type_ == "HSWISH") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
    shift_hswish_ = op->get_attr<int>("shift_hswish");
  }
}

template <typename DType, typename WType>
TConv3dFix<DType, WType>::TConv3dFix(const xir::Subgraph* subg,
                                     const xir::Op* op, IMapTBs_t inputs,
                                     CPUTBPtr_t output)
    : Conv3dBase<DType, WType>(subg, op, inputs, output) {
  fix_point_i_ = op->get_input_ops("input")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  fix_point_w_ = op->get_input_ops("weights")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  fix_point_b_ =
      op->get_input_ops("bias")[0]->get_output_tensor()->get_attr<std::int32_t>(
          "fix_point");
  fix_point_o_ = op->get_output_tensor()->get_attr<std::int32_t>("fix_point");
  shift_bias_ = fix_point_i_ + fix_point_w_ - fix_point_b_;
  shift_cut_ = fix_point_i_ + fix_point_w_ - fix_point_o_;
  act_type_ = op->get_attr<std::string>("nonlinear");
  std::vector<std::string> supported_{"NONE",      "RELU",     "RELU6",
                                      "LEAKYRELU", "HSIGMOID", "HSWISH"};
  UNI_LOG_CHECK(std::find(supported_.begin(), supported_.end(), act_type_) !=
                    supported_.end(),
                VART_INVALID_VALUE)
      << "we do not activation type: " << act_type_ << " in conv3d-fix.";
  if (act_type_ == "HSIGMOID") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
  } else if (act_type_ == "HSWISH") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
    shift_hswish_ = op->get_attr<int>("shift_hswish");
  }
}

template <typename DType, typename WType>
TDWConv3dFix<DType, WType>::TDWConv3dFix(const xir::Subgraph* subg,
                                         const xir::Op* op, IMapTBs_t inputs,
                                         CPUTBPtr_t output)
    : Conv3dBase<DType, WType>(subg, op, inputs, output) {
  fix_point_i_ = op->get_input_ops("input")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  fix_point_w_ = op->get_input_ops("weights")[0]
                     ->get_output_tensor()
                     ->get_attr<std::int32_t>("fix_point");
  if (this->has_bias_)
    fix_point_b_ = op->get_input_ops("bias")[0]
                       ->get_output_tensor()
                       ->get_attr<std::int32_t>("fix_point");
  fix_point_o_ = op->get_output_tensor()->get_attr<std::int32_t>("fix_point");
  shift_bias_ = fix_point_i_ + fix_point_w_ - fix_point_b_;
  shift_cut_ = fix_point_i_ + fix_point_w_ - fix_point_o_;
  act_type_ = op->get_attr<std::string>("nonlinear");
  std::vector<std::string> supported_{"NONE",      "RELU",     "RELU6",
                                      "LEAKYRELU", "HSIGMOID", "HSWISH"};
  UNI_LOG_CHECK(std::find(supported_.begin(), supported_.end(), act_type_) !=
                    supported_.end(),
                VART_INVALID_VALUE)
      << "we do not activation type: " << act_type_ << " in conv3d-fix.";
  if (act_type_ == "HSIGMOID") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
  } else if (act_type_ == "HSWISH") {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
    shift_hswish_ = op->get_attr<int>("shift_hswish");
  }
}

namespace {
double dr(double x) {
  if (x - std::floor(x) >= 0.5)
    return double(std::min(127.0, std::max(-128.0, double(std::ceil(x)))));
  else
    return double(std::min(127.0, std::max(-128.0, double(std::floor(x)))));
}
}  // namespace

template <typename DType, typename WType>
void Conv3dFix<DType, WType>::fix() {
  for (auto i = 0; i < this->fmap_o_.num(); i++) {
    this->data_out_ptr_[i] *= 2;
    if (this->has_bias_) {
      auto pos = i % this->fmap_o_.dcod();
      this->data_out_ptr_[i] +=
          floor(2.0 * this->bias_ptr_[pos] * pow(2.0, shift_bias_));
    }
  }

  auto factor = pow(2, shift_cut_ + 1);
  for (auto i = 0; i < this->fmap_o_.num(); ++i) {
    double tmp = this->data_out_ptr_[i];
    tmp /= factor;
    if (act_type_ == "RELU") {
      if (tmp < 0) tmp = 0;
    } else if (act_type_ == "LEAKYRELU") {
      if (tmp < 0) tmp = tmp * 26. / 256;
    } else if (act_type_ == "RELU6") {
      if (tmp < 0) tmp = 0;
      if (fix_point_o_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (act_type_ == "HSIGMOID") {
      tmp = dr(tmp);
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    } else if (act_type_ == "HSWISH") {
      auto x = dr(tmp);
      auto hsigmoid_x =
          dr(std::min(
                 pow(2, 32),
                 std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
             pow(2, -shift_hsigmoid_));
      tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
    }
    this->data_out_ptr_[i] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

template <typename DType, typename WType>
void TConv3dFix<DType, WType>::fix() {
  for (auto i = 0; i < this->fmap_o_.num(); i++) {
    this->data_out_ptr_[i] *= 2;
    if (this->has_bias_) {
      auto pos = i % this->fmap_o_.dcod();
      this->data_out_ptr_[i] +=
          floor(2.0 * this->bias_ptr_[pos] * pow(2.0, shift_bias_));
    }
  }

  auto factor = pow(2, shift_cut_ + 1);
  for (auto i = 0; i < this->fmap_o_.num(); i++) {
    double tmp = this->data_out_ptr_[i];
    tmp /= factor;
    if (act_type_ == "RELU") {
      if (tmp < 0) tmp = 0;
    } else if (act_type_ == "LEAKYRELU") {
      if (tmp < 0) tmp = tmp * 26. / 256;
    } else if (act_type_ == "RELU6") {
      if (tmp < 0) tmp = 0;
      if (fix_point_o_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (act_type_ == "HSIGMOID") {
      tmp = dr(tmp);
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    } else if (act_type_ == "HSWISH") {
      auto x = dr(tmp);
      auto hsigmoid_x =
          dr(std::min(
                 pow(2, 32),
                 std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
             pow(2, -shift_hsigmoid_));
      tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
    }
    this->data_out_ptr_[i] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

template <typename DType, typename WType>
void DWConv3dFix<DType, WType>::fix() {
  for (auto i = 0; i < this->fmap_o_.num(); i++) {
    this->data_out_ptr_[i] *= 2;
    if (this->has_bias_) {
      auto pos = i % this->fmap_o_.dcod();
      this->data_out_ptr_[i] +=
          floor(2.0 * this->bias_ptr_[pos] * pow(2.0, shift_bias_));
    }
  }

  auto factor = pow(2, shift_cut_ + 1);
  for (auto i = 0; i < this->fmap_o_.num(); i++) {
    double tmp = this->data_out_ptr_[i];
    tmp /= factor;
    if (act_type_ == "RELU") {
      if (tmp < 0) tmp = 0;
    } else if (act_type_ == "LEAKYRELU") {
      if (tmp < 0) tmp = tmp * 26. / 256;
    } else if (act_type_ == "RELU6") {
      if (tmp < 0) tmp = 0;
      if (fix_point_o_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (act_type_ == "HSIGMOID") {
      tmp = dr(tmp);
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    } else if (act_type_ == "HSWISH") {
      auto x = dr(tmp);
      auto hsigmoid_x =
          dr(std::min(
                 pow(2, 32),
                 std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
             pow(2, -shift_hsigmoid_));
      tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
    }
    this->data_out_ptr_[i] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

template <typename DType, typename WType>
void TDWConv3dFix<DType, WType>::fix() {
  for (auto i = 0; i < this->fmap_o_.num(); ++i) {
    this->data_out_ptr_[i] *= 2;
    if (this->has_bias_) {
      auto pos = i % this->fmap_o_.dcod();
      this->data_out_ptr_[i] +=
          floor(2.0 * this->bias_ptr_[pos] * pow(2.0, shift_bias_));
    }
  }

  auto factor = pow(2, shift_cut_ + 1);
  for (auto i = 0; i < this->fmap_o_.num(); i++) {
    double tmp = this->data_out_ptr_[i];
    tmp /= factor;
    if (act_type_ == "RELU") {
      if (tmp < 0) tmp = 0;
    } else if (act_type_ == "LEAKYRELU") {
      if (tmp < 0) tmp = tmp * 26. / 256;
    } else if (act_type_ == "RELU6") {
      if (tmp < 0) tmp = 0;
      if (fix_point_o_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (act_type_ == "HSIGMOID") {
      tmp = dr(tmp);
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    } else if (act_type_ == "HSWISH") {
      auto x = dr(tmp);
      auto hsigmoid_x =
          dr(std::min(
                 pow(2, 32),
                 std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
             pow(2, -shift_hsigmoid_));
      tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
    }
    this->data_out_ptr_[i] =
        DPURound<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

template <typename DType, typename WType>
void Conv3dFix<DType, WType>::run() {
  this->conv();
  this->fix();
}

template <typename DType, typename WType>
void TConv3dFix<DType, WType>::run() {
  this->conv();
  this->fix();
}

template <typename DType, typename WType>
void DWConv3dFix<DType, WType>::run() {
  this->conv();
  this->fix();
}

template <typename DType, typename WType>
void TDWConv3dFix<DType, WType>::run() {
  this->conv();
  this->fix();
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(Conv3dBase);
INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(Conv3d);
INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(Conv3dFix);
INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(DWConv3dFix);
INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(TConv3dFix);
INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(TDWConv3dFix);

REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("conv3d", Conv3d, "weights");
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("depthwise-conv3d", Conv3d,
                                                "weights");
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("transposed-conv3d", Conv3d,
                                                "weights");
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("transposed-depthwise-conv3d",
                                                Conv3d, "weights");
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("conv3d-fix", Conv3dFix,
                                                "weights");
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("depthwise-conv3d-fix",
                                                DWConv3dFix, "weights");
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("transposed-conv3d-fix",
                                                TConv3dFix, "weights");
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC(
    "transposed-depthwise-conv3d-fix", TDWConv3dFix, "weights");

}  // namespace vart::cpu
