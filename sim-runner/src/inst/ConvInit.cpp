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
#include "ConvInit.hpp"

#include "ConvAddr.hpp"
#include "general_util.h"

template <>
ConvInit<DPUVersion::DPUV2>::ConvInit(int inst_type, int instid,
                                      vector<string>& inst_str,
                                      vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::DPUV2>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  // get convinit instruction field value
  using TABLE = TableInterface<DPUVersion::DPUV2>::inst_table;
  kernel_w_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_H];
  shift_bias_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_BIAS];
  shift_cut_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_CUT];
  act_type_ = field_val_[TABLE::CONVINIT_FIELD_ACT_TYPE];
  stride_w_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_H];
  stride_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OUT];
  valid_pixel_parallel_ =
      field_val_[TABLE::CONVINIT_FIELD_VALID_PIXEL_PARALLEL];
  jump_read_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ];
  jump_read_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_ENDL];
  jump_write_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE];
  jump_write_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_ENDL];
  stride_offset_in_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_IN];
  stride_offset_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_OUT];

  CUROBJ = this;
}

template <>
ConvInit<DPUVersion::DPUV3E>::ConvInit(int inst_type, int instid,
                                       vector<string>& inst_str,
                                       vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::DPUV3E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  // get convinit instruction field value
  using TABLE = TableInterface<DPUVersion::DPUV3E>::inst_table;
  kernel_w_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_H];
  shift_bias_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_BIAS];
  shift_cut_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_CUT];
  act_type_ = field_val_[TABLE::CONVINIT_FIELD_ACT_TYPE];
  stride_w_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_H];
  stride_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OUT];
  valid_pixel_parallel_ =
      field_val_[TABLE::CONVINIT_FIELD_VALID_PIXEL_PARALLEL];
  jump_read_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ];
  jump_read_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_ENDL];
  jump_write_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE];
  jump_write_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_ENDL];
  stride_offset_in_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_IN];
  stride_offset_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_OUT];

  CUROBJ = this;
}

template <>
ConvInit<DPUVersion::DPUV3ME>::ConvInit(int inst_type, int instid,
                                        vector<string>& inst_str,
                                        vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::DPUV3ME>::InstInit(inst_type_, root_debug_path_, debug_,
                                        opcode_, debug_path_);
  // get convinit instruction field value
  using TABLE = TableInterface<DPUVersion::DPUV3ME>::inst_table;
  kernel_w_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_H];
  shift_bias_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_BIAS];
  shift_relusix_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_RELUSIX];
  shift_cut_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_CUT];
  act_type_ = field_val_[TABLE::CONVINIT_FIELD_ACT_TYPE];
  stride_w_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_H];
  stride_out_cb_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OUT_CB];
  stride_out_hbm_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OUT_HBM];
  valid_pixel_parallel_ =
      field_val_[TABLE::CONVINIT_FIELD_VALID_PIXEL_PARALLEL];
  jump_read_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ];
  jump_read_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_ENDL];
  jump_write_cb_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_CB];
  jump_write_endl_cb_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_ENDL_CB];
  jump_write_hbm_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_HBM];
  jump_write_endl_hbm_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_ENDL_HBM];
  stride_offset_in_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_IN];
  stride_offset_out_cb_ =
      field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_OUT_CB];
  stride_offset_out_hbm_ =
      field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_OUT_HBM];

  CUROBJ = this;
}

template <>
ConvInit<DPUVersion::DPUV4E>::ConvInit(int inst_type, int instid,
                                       vector<string>& inst_str,
                                       vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::DPUV4E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  // get convinit instruction field value
  using TABLE = TableInterface<DPUVersion::DPUV4E>::inst_table;
  kernel_w_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_H];
  shift_bias_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_BIAS];
  shift_cut_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_CUT];
  act_type_ = field_val_[TABLE::CONVINIT_FIELD_ACT_TYPE];
  shift_hswish_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_HSWISH];
  stride_w_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_H];
  stride_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OUT];
  length_ = field_val_[TABLE::CONVINIT_FIELD_LENGTH];
  channel_offset_ = field_val_[TABLE::CONVINIT_FIELD_CHANNEL_OFFSET];
  channel_group_ = field_val_[TABLE::CONVINIT_FIELD_CHANNEL_GROUP];
  jump_read_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ];
  jump_read_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_ENDL];
  jump_write_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE];
  jump_write_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_ENDL];
  stride_offset_in_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_IN];
  stride_offset_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_OUT];

  batch_num_ = field_val_[TABLE::CONVINIT_FIELD_BATCH_NUM];
  stride_batch_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_BATCH];
  hsigmoid_in_ = field_val_[TABLE::CONVINIT_FIELD_HSIGMOID_IN];
  shift_hsigmoid_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_HSIGMOID];

  CUROBJ = this;
}

template <>
ConvInit<DPUVersion::XVDPU>::ConvInit(int inst_type, int instid,
                                      vector<string>& inst_str,
                                      vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::XVDPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  // get convinit instruction field value
  using TABLE = TableInterface<DPUVersion::XVDPU>::inst_table;
  kernel_w_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_H];
  shift_bias_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_BIAS];
  shift_cut_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_CUT];
  hsigmoid_in_ = field_val_[TABLE::CONVINIT_FIELD_HSIGMOID_IN];
  shift_hsigmoid_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_HSIGMOID];
  shift_hswish_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_HSWISH];
  act_type_ = field_val_[TABLE::CONVINIT_FIELD_ACT_TYPE];
  stride_w_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_H];
  stride_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OUT];
  jump_read_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ];
  jump_read_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_ENDL];
  jump_write_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE];
  jump_write_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_ENDL];
  stride_offset_in_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_IN];
  stride_offset_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_OUT];

  UNI_LOG_CHECK(SimCfg::Instance().get_xvdpu_conv_remain() == 0,
                SIM_PARAMETER_FAILED)
      << "Conv_num is not used up in previous ConvInit! total "
      << SimCfg::Instance().get_xvdpu_conv_num() << ", "
      << SimCfg::Instance().get_xvdpu_conv_remain() << " left.";
  conv_num_ = field_val_[TABLE::CONVINIT_FIELD_CONV_NUM];
  SimCfg::Instance().set_xvdpu_conv_num(conv_num_);
  SimCfg::Instance().set_xvdpu_conv_remain(conv_num_);

  tile_icg_ = field_val_[TABLE::CONVINIT_FIELD_TILE_ICG];
  tile_ocg_ = field_val_[TABLE::CONVINIT_FIELD_TILE_OCG];
  tile_owg_ = field_val_[TABLE::CONVINIT_FIELD_TILE_OWG];
  ic_iter_ = field_val_[TABLE::CONVINIT_FIELD_IC_ITER];
  channel_offset_ = field_val_[TABLE::CONVINIT_FIELD_CHANNEL_OFFSET];
  ow_iter_ = field_val_[TABLE::CONVINIT_FIELD_OW_ITER];
  ow_offset_ = field_val_[TABLE::CONVINIT_FIELD_OW_OFFSET];
  jump_read_weights_endl_ =
      field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_WEIGHTS_ENDL];
  jump_read_weights_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_WEIGHTS];

  tile_en_ = field_val_[TABLE::CONVINIT_FIELD_TILE_EN];

  CUROBJ = this;
}

template <>
ConvInit<DPUVersion::XV2DPU>::ConvInit(int inst_type, int instid,
                                       vector<string>& inst_str,
                                       vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::XV2DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  // get convinit instruction field value
  using TABLE = TableInterface<DPUVersion::XV2DPU>::inst_table;

  exec_mode_ = field_val_[TABLE::CONVINIT_FIELD_EXEC_MODE];

  kernel_h_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_H];
  kernel_w_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_W];
  stride_h_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_H];
  stride_w_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_W];

  total_in_ = field_val_[TABLE::CONVINIT_FIELD_TOTAL_IN];
  tile_icg_ = field_val_[TABLE::CONVINIT_FIELD_TILE_ICG];
  tile_ocg_ = field_val_[TABLE::CONVINIT_FIELD_TILE_OCG];
  tile_ohg_ = field_val_[TABLE::CONVINIT_FIELD_TILE_OHG];
  tile_owg_ = field_val_[TABLE::CONVINIT_FIELD_TILE_OWG];

  ic_iter_ = field_val_[TABLE::CONVINIT_FIELD_IC_ITER];
  ow_iter_ = field_val_[TABLE::CONVINIT_FIELD_OW_ITER];
  ow_offset_ = field_val_[TABLE::CONVINIT_FIELD_OW_OFFSET];
  shift_hswish_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_HSWISH];
  shift_hsigmoid_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_HSIGMOID];
  hsigmoid_in_ = field_val_[TABLE::CONVINIT_FIELD_HSIGMOID_IN];

  act_type_ = field_val_[TABLE::CONVINIT_FIELD_ACT_TYPE];
  aie_mode_ = field_val_[TABLE::CONVINIT_FIELD_AIE_MODE];
  conv_num_ = field_val_[TABLE::CONVINIT_FIELD_CONV_NUM];

  shift_bias_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_BIAS];
  shift_cut_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_CUT];
  one_height_ = field_val_[TABLE::CONVINIT_FIELD_ONE_HEIGHT];
  one_width_ = field_val_[TABLE::CONVINIT_FIELD_ONE_WIDTH];

  one_cube_ = field_val_[TABLE::CONVINIT_FIELD_ONE_CUBE];
  tile_en_ = field_val_[TABLE::CONVINIT_FIELD_TILE_EN];
  total_tile_ = field_val_[TABLE::CONVINIT_FIELD_TOTAL_TILE];
  one_size_ = field_val_[TABLE::CONVINIT_FIELD_ONE_SIZE];

  shift_elew0_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_ELEW0];
  shift_elew1_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_ELEW1];
  elew_cut_ = field_val_[TABLE::CONVINIT_FIELD_ELEW_CUT];
  elew_act_type_ = field_val_[TABLE::CONVINIT_FIELD_ELEW_ACT_TYPE];

  kh_iter_ = field_val_[TABLE::CONVINIT_FIELD_KH_ITER];
  kw_iter_ = field_val_[TABLE::CONVINIT_FIELD_KW_ITER];

  auto kernel_h_inst = kernel_h_;
  auto kernel_w_inst = kernel_w_;
  std::tie(kernel_h_, kernel_w_) =
      reverse_kernel_iterate(kernel_h_, kernel_w_, kh_iter_, kw_iter_, 1, 16);
  UNI_LOG_CHECK_EQ(kernel_h_inst * kh_iter_, kernel_h_, SIM_PARAMETER_FAILED)
      << UNI_LOG_VALUES(kernel_h_inst, kh_iter_, kernel_h_);
  UNI_LOG_CHECK_EQ(kernel_w_inst * kw_iter_, kernel_w_, SIM_PARAMETER_FAILED)
      << UNI_LOG_VALUES(kernel_w_inst, kw_iter_, kernel_w_);

  UNI_LOG_CHECK(SimCfg::Instance().get_xvdpu_conv_remain() == 0,
                SIM_PARAMETER_FAILED)
      << "Conv_num is not used up in previous ConvInit! total "
      << SimCfg::Instance().get_xvdpu_conv_num() << ", "
      << SimCfg::Instance().get_xvdpu_conv_remain() << " left.";
  SimCfg::Instance().set_xvdpu_conv_num(conv_num_);
  SimCfg::Instance().set_xvdpu_conv_remain(conv_num_);

  CUROBJ = this;
}

template <>
ConvInit<DPUVersion::DPU4F>::ConvInit(int inst_type, int instid,
                                      vector<string>& inst_str,
                                      vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::DPU4F>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  // get convinit instruction field value
  using TABLE = TableInterface<DPUVersion::DPU4F>::inst_table;
  kernel_w_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::CONVINIT_FIELD_KERNEL_H];
  shift_bias_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_BIAS];
  shift_cut_ = field_val_[TABLE::CONVINIT_FIELD_SHIFT_CUT];
  quant_lth_ = field_val_[TABLE::CONVINIT_FIELD_QUANT_LTH];
  calt_mode_ = field_val_[TABLE::CONVINIT_FIELD_CALT_MODE];
  act_type_ = field_val_[TABLE::CONVINIT_FIELD_ACT_TYPE];
  stride_w_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_H];
  stride_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OUT];
  valid_pixel_parallel_ =
      field_val_[TABLE::CONVINIT_FIELD_VALID_PIXEL_PARALLEL];
  jump_read_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ];
  jump_read_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_READ_ENDL];
  jump_write_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE];
  jump_write_endl_ = field_val_[TABLE::CONVINIT_FIELD_JUMP_WRITE_ENDL];
  stride_offset_in_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_IN];
  stride_offset_out_ = field_val_[TABLE::CONVINIT_FIELD_STRIDE_OFFSET_OUT];

  CUROBJ = this;
}

template <DPUVersion T>
void ConvInit<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}
