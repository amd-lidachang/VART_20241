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

#pragma once

#include "cpu_op_base.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename WType>
class Conv3dBase : public CPUOPBase {
 public:
  Conv3dBase(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
             CPUTBPtr_t output);
  virtual ~Conv3dBase() = default;
  VART_DISABLE_COPY_AND_ASSIGN(Conv3dBase);

  virtual void run() override {}

  virtual void print_param() override;
  virtual void check_param() override;

  virtual void read() override final;

  virtual uint64_t get_workload() override final;

 protected:
  void conv();
  void conv_normal();
  void conv_normal_thread();

  void conv_gemm();
  void conv_gemm_thread();

  void conv_one(int idx_on, int idx_oh, int idx_ow, int idx_od, int idx_oc);
  void dwconv_one(int idx_on, int idx_oh, int idx_ow, int idx_od, int idx_oc);
  void bias();

 private:
  uint64_t THREAD_NUM;
  uint64_t THREAD_WORKLOAD;

 protected:
  bool has_bias_ = false;
  bool if_depthwise_ = false;

  FMap_t fmap_o_;
  FMap_t fmap_i_;
  FMap_t fmap_pad_i_;
  FMap_t fmap_w_;
  FMap_t fmap_dilated_w_;

  Kernel_t kernel_;
  Stride_t stride_;
  Dilation_t dilation_f_;
  Dilation_t dilation_w_;
  Pad_t pad_;
  int32_t group_ = 1;

  DType* data_in_ptr_{nullptr};
  WType* weights_ptr_{nullptr};
  WType* bias_ptr_{nullptr};
  DType* data_out_ptr_{nullptr};
};

template <typename DType, typename WType>
class Conv3d : public Conv3dBase<DType, WType> {
 public:
  Conv3d(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
         CPUTBPtr_t output);
  virtual ~Conv3d() = default;

  virtual void run() override;
};

template <typename DType, typename WType>
class Conv3dFix : public Conv3dBase<DType, WType> {
 public:
  Conv3dFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
            CPUTBPtr_t output);
  virtual ~Conv3dFix() = default;

  void fix();
  virtual void run() override;

 private:
  int fix_point_i_;
  int fix_point_w_;
  int fix_point_b_;
  int fix_point_o_;
  int shift_bias_;
  int shift_cut_;
  int hsigmoid_in_;
  int shift_hsigmoid_;
  int shift_hswish_;

  std::string act_type_;
};

template <typename DType, typename WType>
class DWConv3dFix : public Conv3dBase<DType, WType> {
 public:
  DWConv3dFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
              CPUTBPtr_t output);
  virtual ~DWConv3dFix() = default;

  void fix();
  virtual void run() override;

 private:
  int fix_point_i_{0};
  int fix_point_w_{0};
  int fix_point_b_{0};
  int fix_point_o_{0};
  int shift_bias_;
  int shift_cut_;
  int hsigmoid_in_;
  int shift_hsigmoid_;
  int shift_hswish_;

  std::string act_type_;
};

template <typename DType, typename WType>
class TConv3dFix : public Conv3dBase<DType, WType> {
 public:
  TConv3dFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
             CPUTBPtr_t output);
  virtual ~TConv3dFix() = default;

  void fix();
  virtual void run() override;

 private:
  int fix_point_i_{0};
  int fix_point_w_{0};
  int fix_point_b_{0};
  int fix_point_o_{0};
  int shift_bias_;
  int shift_cut_;
  int hsigmoid_in_;
  int shift_hsigmoid_;
  int shift_hswish_;

  std::string act_type_;
};

template <typename DType, typename WType>
class TDWConv3dFix : public Conv3dBase<DType, WType> {
 public:
  TDWConv3dFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
               CPUTBPtr_t output);
  virtual ~TDWConv3dFix() = default;

  void fix();
  virtual void run() override;

 private:
  int fix_point_i_{0};
  int fix_point_w_{0};
  int fix_point_b_{0};
  int fix_point_o_{0};
  int shift_bias_;
  int shift_cut_;
  int hsigmoid_in_;
  int shift_hsigmoid_;
  int shift_hswish_;

  std::string act_type_;
};

}  // namespace cpu
}  // namespace vart
