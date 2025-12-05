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
#include "AluInit.hpp"

#include "AluAddr.hpp"
#include "general_util.h"

template <DPUVersion dv>
AluInit<dv>* AluInit<dv>::CUROBJ = nullptr;
template <DPUVersion dv>
int32_t AluInit<dv>::left_alu_num = 0;

// constructor and deconstructor
template <>
AluInit<DPUVersion::DPUV2>::AluInit(int inst_type, int instid,
                                    vector<string>& inst_str,
                                    vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = false;
  // debug_ =
  //    debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::DPUV2>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV2>::inst_table;
  // get AluInit instruction field value
  jump_read_endl_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_READ_ENDL];
  jump_read_weights_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_READ_WEIGHTS];
  shift_cut_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_CUT];

  jump_read_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_READ];
  shift_bias_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_BIAS];
  act_type_ = field_val_[TABLE::ALUINIT_FIELD_ACT_TYPE];
  exec_mode_ = field_val_[TABLE::ALUINIT_FIELD_EXEC_MODE];
  stride_w_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_H];

  jump_write_endl_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_WRITE_ENDL];
  stride_offset_out_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_OFFSET_OUT];
  stride_offset_in_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_OFFSET_IN];
  channel_offset_ = field_val_[TABLE::ALUINIT_FIELD_CHANNEL_OFFSET];
  channel_group_ = field_val_[TABLE::ALUINIT_FIELD_CHANNEL_GROUP];

  jump_write_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_WRITE];
  length_ = field_val_[TABLE::ALUINIT_FIELD_LENGTH];
  stride_out_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_OUT];
  multi_factor_ = field_val_[TABLE::ALUINIT_FIELD_MULTI_FACTOR];

  kernel_w_ = field_val_[TABLE::ALUINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::ALUINIT_FIELD_KERNEL_H];
  shift_prelu_p_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_PRELU_P];
  shift_prelu_n_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_PRELU_N];

  CUROBJ = this;
}

template <>
AluInit<DPUVersion::XVDPU>::AluInit(int inst_type, int instid,
                                    vector<string>& inst_str,
                                    vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = false;
  // debug_ =
  //    debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::XVDPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XVDPU>::inst_table;
  // get AluInit instruction field value
  jump_read_endl_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_READ_ENDL];
  jump_read_weights_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_READ_WEIGHTS];
  shift_cut_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_CUT];

  jump_read_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_READ];
  shift_bias_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_BIAS];
  act_type_ = field_val_[TABLE::ALUINIT_FIELD_ACT_TYPE];
  exec_mode_ = field_val_[TABLE::ALUINIT_FIELD_EXEC_MODE];
  stride_w_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_H];

  jump_write_endl_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_WRITE_ENDL];
  stride_offset_out_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_OFFSET_OUT];
  stride_offset_in_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_OFFSET_IN];
  channel_offset_ = field_val_[TABLE::ALUINIT_FIELD_CHANNEL_OFFSET];
  channel_group_ = field_val_[TABLE::ALUINIT_FIELD_CHANNEL_GROUP];

  jump_write_ = field_val_[TABLE::ALUINIT_FIELD_JUMP_WRITE];
  length_ = field_val_[TABLE::ALUINIT_FIELD_LENGTH];
  stride_out_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_OUT];
  multi_factor_ = field_val_[TABLE::ALUINIT_FIELD_MULTI_FACTOR];

  kernel_w_ = field_val_[TABLE::ALUINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::ALUINIT_FIELD_KERNEL_H];
  shift_prelu_p_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_PRELU_P];
  shift_prelu_n_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_PRELU_N];

  CUROBJ = this;
}

template <>
AluInit<DPUVersion::XV2DPU>::AluInit(int inst_type, int instid,
                                     vector<string>& inst_str,
                                     vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_ALU);
  Helper<DPUVersion::XV2DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XV2DPU>::inst_table;

  kernel_w_ = field_val_[TABLE::ALUINIT_FIELD_KERNEL_W];
  kernel_h_ = field_val_[TABLE::ALUINIT_FIELD_KERNEL_H];
  exec_mode_ = field_val_[TABLE::ALUINIT_FIELD_EXEC_MODE];

  stride_w_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_W];
  stride_h_ = field_val_[TABLE::ALUINIT_FIELD_STRIDE_H];
  b_mode_ = field_val_[TABLE::ALUINIT_FIELD_B_MODE];
  num_ = field_val_[TABLE::ALUINIT_FIELD_NUM];
  share_channel_group_ = field_val_[TABLE::ALUINIT_FIELD_SHARE_CHANNEL_GROUP];
  share_kernel_ = field_val_[TABLE::ALUINIT_FIELD_SHARE_KERNEL];
  tile_owg_ = field_val_[TABLE::ALUINIT_FIELD_TILE_OWG];
  tile_ohg_ = field_val_[TABLE::ALUINIT_FIELD_TILE_OHG];

  tile_cg_ = field_val_[TABLE::ALUINIT_FIELD_TILE_CG];
  ow_iter_ = field_val_[TABLE::ALUINIT_FIELD_OW_ITER];
  ow_offset_ = field_val_[TABLE::ALUINIT_FIELD_OW_OFFSET];
  shift_hswish_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_HSWISH];
  shift_hsigmoid_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_HSIGMOID];
  hsigmoid_in_ = field_val_[TABLE::ALUINIT_FIELD_HSIGMOID_IN];

  alu_num_ = field_val_[TABLE::ALUINIT_FIELD_ALU_NUM];
  UNI_LOG_CHECK(AluInit::left_alu_num == 0, SIM_PARAMETER_FAILED)
      << "alu_num is not used up in previous aluInit! " << AluInit::left_alu_num
      << " left.";
  AluInit::left_alu_num = alu_num_;
  act_type_ = field_val_[TABLE::ALUINIT_FIELD_ACT_TYPE];
  shift_bias_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_BIAS];
  shift_cut_ = field_val_[TABLE::ALUINIT_FIELD_SHIFT_CUT];

  weights_lines_ = field_val_[TABLE::ALUINIT_FIELD_WEIGHTS_LINES];
  shift_read_list_[0] = field_val_[TABLE::ALUINIT_FIELD_SHIFT_READ_0];
  shift_read_list_[1] = field_val_[TABLE::ALUINIT_FIELD_SHIFT_READ_1];
  shift_read_list_[2] = field_val_[TABLE::ALUINIT_FIELD_SHIFT_READ_2];
  shift_read_list_[3] = field_val_[TABLE::ALUINIT_FIELD_SHIFT_READ_3];

  total_tile_ = field_val_[TABLE::ALUINIT_FIELD_TOTAL_TILE];
  one_height_  = field_val_[TABLE::ALUINIT_FIELD_ONE_HEIGHT];
  one_width_ = field_val_[TABLE::ALUINIT_FIELD_ONE_WIDTH];

  kh_iter_ = field_val_[TABLE::ALUINIT_FIELD_KH_ITER];
  kw_iter_ = field_val_[TABLE::ALUINIT_FIELD_KW_ITER];

  incAO3_ = field_val_[TABLE::ALUINIT_FIELD_INCAO3];
  incAO2_ = field_val_[TABLE::ALUINIT_FIELD_INCAO2];

  std::tie(kernel_h_, kernel_w_) =
      reverse_kernel_iterate(kernel_h_, kernel_w_, kh_iter_, kw_iter_, 1, 256);

  pad_type_ = field_val_[TABLE::ALUINIT_FIELD_PAD_TYPE];
  UNI_LOG_CHECK(pad_type_ == Calc::pad_type_t::PAD_TYPE_ZERO ||
                    pad_type_ == Calc::pad_type_t::PAD_TYPE_MIN,
                SIM_PARAMETER_FAILED);
  CUROBJ = this;
}

template <DPUVersion T>
AluInit<T>::~AluInit() {}

template <DPUVersion T>
void AluInit<T>::Exec() {}

template <DPUVersion T>
void AluInit<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}
