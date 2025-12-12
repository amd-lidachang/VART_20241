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
#include "Alu.hpp"

#include "AluAddr.hpp"
#include "AluInit.hpp"
#include "pub/DumpBank.hpp"
#include "xv2dpu/simCfg.hpp"

double dr(double x);

template <DPUVersion dv>
const DPU_DATA_TYPE Alu<dv>::MIN_DTYPE_VALUE =
    std::numeric_limits<DPU_DATA_TYPE>::min();

template <DPUVersion dv>
AluInit<dv>* AluInit<dv>::CUROBJ;

// constructor and deconstructor

template <>
Alu<DPUVersion::DPUV2>::Alu(int inst_type, int instid, vector<string>& inst_str,
                            vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_ALU);
  Helper<DPUVersion::DPUV2>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV2>::inst_table;
  // get Alu instruction field value
  bank_addr_out_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_OUT];
  bank_id_out_ = field_val_[TABLE::ALU_FIELD_BANK_ID_OUT];

  bank_addr_in_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN];
  bank_id_in_ = field_val_[TABLE::ALU_FIELD_BANK_ID_IN];
  valid_pixel_parallel_ = field_val_[TABLE::ALU_FIELD_VALID_PIXEL_PARALLEL];

  bank_addr_in_1_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN_1];
  pad_bottom_ = field_val_[TABLE::ALU_FIELD_PAD_BOTTOM];
  pad_right_ = field_val_[TABLE::ALU_FIELD_PAD_RIGHT];
  pad_left_ = field_val_[TABLE::ALU_FIELD_PAD_LEFT];
  pad_top_ = field_val_[TABLE::ALU_FIELD_PAD_TOP];

  bank_addr_in_2_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN_2];
  bank_addr_weights_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_WEIGHTS];

  bank_addr_in_3_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN_3];
  bank_addr_bias_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_BIAS];

  // AluInit value
  using ALUINIT = AluInit<DPUVersion::DPUV2>;
  jump_read_endl_ = ALUINIT::CUROBJ->get_jump_read_endl();
  jump_read_weights_ = ALUINIT::CUROBJ->get_jump_read_weights();
  shift_cut_ = ALUINIT::CUROBJ->get_shift_cut();

  jump_read_ = ALUINIT::CUROBJ->get_jump_read();
  shift_bias_ = ALUINIT::CUROBJ->get_shift_bias();
  act_type_ = ALUINIT::CUROBJ->get_act_type();
  exec_mode_ = ALUINIT::CUROBJ->get_exec_mode();
  stride_w_ = ALUINIT::CUROBJ->get_stride_w();
  stride_h_ = ALUINIT::CUROBJ->get_stride_h();

  jump_write_endl_ = ALUINIT::CUROBJ->get_jump_write_endl();
  stride_offset_in_ = ALUINIT::CUROBJ->get_stride_offset_in();
  stride_offset_out_ = ALUINIT::CUROBJ->get_stride_offset_out();
  channel_offset_ = ALUINIT::CUROBJ->get_channel_offset();
  channel_group_ = ALUINIT::CUROBJ->get_channel_group();

  jump_write_ = ALUINIT::CUROBJ->get_jump_write();
  length_ = ALUINIT::CUROBJ->get_length();
  stride_out_ = ALUINIT::CUROBJ->get_stride_out();
  multi_factor_ = ALUINIT::CUROBJ->get_multi_factor();

  kernel_w_ = ALUINIT::CUROBJ->get_kernel_w();
  kernel_h_ = ALUINIT::CUROBJ->get_kernel_h();
  kernel_d_ = field_val_[TABLE::ALU_FIELD_KERNEL_D];

  shift_prelu_p_ = ALUINIT::CUROBJ->get_shift_prelu_p();
  shift_prelu_n_ = ALUINIT::CUROBJ->get_shift_prelu_n();

  shift_p_p =
      shift_prelu_p_ >= 32 ? shift_prelu_p_ - 32 : shift_prelu_p_ * (-1);
  shift_p_n =
      shift_prelu_n_ >= 32 ? shift_prelu_n_ - 32 : shift_prelu_n_ * (-1);

  // constraints from hardware implementation
  UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED);

  UNI_LOG_CHECK(
      (exec_mode_ < Calc::ALU_TYPE_MAX) && (exec_mode_ >= Calc::ALU_TYPE_MIN),
      SIM_PARAMETER_FAILED)
      << "alu engine exec_mode_ error, " << exec_mode_
      << ", which range should be [" << Calc::ALU_TYPE_MIN << ","
      << Calc::ALU_TYPE_MAX << ")" << endl;
  auto alu_support_list = ArchCfg::Instance().get_alu_support_list();
  if (alu_support_list.find(exec_mode_) == alu_support_list.end()) {
    UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
        << "target not support this exec_mode " << exec_mode_;
  }

  if (exec_mode_ == Calc::ALU_TYPE_LEAKYRELU ||
      exec_mode_ == Calc::ALU_TYPE_PRELU ||
      exec_mode_ == Calc::ALU_TYPE_HSIGMOID) {
    UNI_LOG_CHECK(stride_h_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, stride_h must be 1";
    UNI_LOG_CHECK(stride_w_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, stride_w must be 1";
    UNI_LOG_CHECK(kernel_h_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, kernel_h must be 1";
    UNI_LOG_CHECK(kernel_w_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, kernel_w must be 1";
    UNI_LOG_CHECK(kernel_d_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, kernel_d must be 1";
  }

  // self defined
  pp_alu_ = ArchCfg::Instance().get_param().alu_engine().pixel_parallel();
  vpp_ = valid_pixel_parallel_;
  UNI_LOG_CHECK(vpp_ <= pp_alu_, SIM_PARAMETER_FAILED);

  pp_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  UNI_LOG_CHECK(pp_alu_ <= pp_, SIM_PARAMETER_FAILED);
  cp_ = ArchCfg::Instance().get_param().alu_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel
  src_h_ = (pp_alu_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (vpp_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (length_ - 1) * stride_w_ + kernel_w_;
  dst_h_ = pp_alu_;
  real_dst_h_ = vpp_;
  dst_w_ = length_;
  src_size_ = src_h_ * src_w_ * ic_;

  group_id_in_ = bank_id_in_ / pp_;
  group_id_out_ = bank_id_out_ / pp_;

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_ * kernel_d_);

  weights_data_size_ = (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) ? 2 : 1;
  weights_.resize(oc_ * kernel_h_ * kernel_w_ * kernel_d_ * weights_data_size_);
  weights_16bits_.resize(oc_ * kernel_h_ * kernel_w_ * kernel_d_);

  prelu_.resize(oc_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
  if (exec_mode_ == Calc::ALU_TYPE_MAXPOOL ||
      exec_mode_ == Calc::ALU_TYPE_MAXREDUCE) {
    std::fill(rlt_dtype_.begin(), rlt_dtype_.end(), MIN_DTYPE_VALUE);
  }
}

template <>
Alu<DPUVersion::XVDPU>::Alu(int inst_type, int instid, vector<string>& inst_str,
                            vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_ALU);
  Helper<DPUVersion::XVDPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XVDPU>::inst_table;
  // get Alu instruction field value
  bank_addr_out_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_OUT];
  bank_id_out_ = field_val_[TABLE::ALU_FIELD_BANK_ID_OUT];

  bank_addr_in_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN];
  bank_id_in_ = field_val_[TABLE::ALU_FIELD_BANK_ID_IN];
  valid_pixel_parallel_ = field_val_[TABLE::ALU_FIELD_VALID_PIXEL_PARALLEL];

  bank_addr_in_1_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN_1];
  pad_bottom_ = field_val_[TABLE::ALU_FIELD_PAD_BOTTOM];
  pad_right_ = field_val_[TABLE::ALU_FIELD_PAD_RIGHT];
  pad_left_ = field_val_[TABLE::ALU_FIELD_PAD_LEFT];
  pad_top_ = field_val_[TABLE::ALU_FIELD_PAD_TOP];

  bank_addr_in_2_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN_2];
  bank_addr_weights_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_WEIGHTS];

  bank_addr_in_3_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_IN_3];
  bank_addr_bias_ = field_val_[TABLE::ALU_FIELD_BANK_ADDR_BIAS];

  // AluInit value
  using ALUINIT = AluInit<DPUVersion::XVDPU>;
  jump_read_endl_ = ALUINIT::CUROBJ->get_jump_read_endl();
  jump_read_weights_ = ALUINIT::CUROBJ->get_jump_read_weights();
  shift_cut_ = ALUINIT::CUROBJ->get_shift_cut();

  jump_read_ = ALUINIT::CUROBJ->get_jump_read();
  shift_bias_ = ALUINIT::CUROBJ->get_shift_bias();
  act_type_ = ALUINIT::CUROBJ->get_act_type();
  exec_mode_ = ALUINIT::CUROBJ->get_exec_mode();
  stride_w_ = ALUINIT::CUROBJ->get_stride_w();
  stride_h_ = ALUINIT::CUROBJ->get_stride_h();

  jump_write_endl_ = ALUINIT::CUROBJ->get_jump_write_endl();
  stride_offset_in_ = ALUINIT::CUROBJ->get_stride_offset_in();
  stride_offset_out_ = ALUINIT::CUROBJ->get_stride_offset_out();
  channel_offset_ = ALUINIT::CUROBJ->get_channel_offset();
  channel_group_ = ALUINIT::CUROBJ->get_channel_group();

  jump_write_ = ALUINIT::CUROBJ->get_jump_write();
  length_ = ALUINIT::CUROBJ->get_length();
  stride_out_ = ALUINIT::CUROBJ->get_stride_out();
  multi_factor_ = ALUINIT::CUROBJ->get_multi_factor();

  kernel_w_ = ALUINIT::CUROBJ->get_kernel_w();
  kernel_h_ = ALUINIT::CUROBJ->get_kernel_h();
  kernel_d_ = field_val_[TABLE::ALU_FIELD_KERNEL_D];

  shift_prelu_p_ = ALUINIT::CUROBJ->get_shift_prelu_p();
  shift_prelu_n_ = ALUINIT::CUROBJ->get_shift_prelu_n();

  shift_p_p =
      shift_prelu_p_ >= 32 ? shift_prelu_p_ - 32 : shift_prelu_p_ * (-1);
  shift_p_n =
      shift_prelu_n_ >= 32 ? shift_prelu_n_ - 32 : shift_prelu_n_ * (-1);

  // constraints from hardware implementation
  UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED);

  UNI_LOG_CHECK(
      (exec_mode_ < Calc::ALU_TYPE_MAX) && (exec_mode_ >= Calc::ALU_TYPE_MIN),
      SIM_PARAMETER_FAILED)
      << "alu engine exec_mode_ error, " << exec_mode_
      << ", which range should be [" << Calc::ALU_TYPE_MIN << ","
      << Calc::ALU_TYPE_MAX << ")" << endl;
  auto alu_support_list = ArchCfg::Instance().get_alu_support_list();
  if (alu_support_list.find(exec_mode_) == alu_support_list.end()) {
    UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
        << "target not support this exec_mode " << exec_mode_;
  }

  if (exec_mode_ == Calc::ALU_TYPE_LEAKYRELU ||
      exec_mode_ == Calc::ALU_TYPE_PRELU ||
      exec_mode_ == Calc::ALU_TYPE_HSIGMOID) {
    UNI_LOG_CHECK(stride_h_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, stride_h must be 1";
    UNI_LOG_CHECK(stride_w_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, stride_w must be 1";
    UNI_LOG_CHECK(kernel_h_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, kernel_h must be 1";
    UNI_LOG_CHECK(kernel_w_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, kernel_w must be 1";
    UNI_LOG_CHECK(kernel_d_ == 1, SIM_PARAMETER_FAILED)
        << " do relu, kernel_d must be 1";
  }

  // self defined
  pp_alu_ = ArchCfg::Instance().get_param().alu_engine().pixel_parallel();
  vpp_ = valid_pixel_parallel_;
  UNI_LOG_CHECK(vpp_ <= pp_alu_, SIM_PARAMETER_FAILED);

  pp_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  UNI_LOG_CHECK(pp_alu_ <= pp_, SIM_PARAMETER_FAILED);
  cp_ = ArchCfg::Instance().get_param().alu_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel
  src_h_ = (pp_alu_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (vpp_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (length_ - 1) * stride_w_ + kernel_w_;
  dst_h_ = pp_alu_;
  real_dst_h_ = vpp_;
  dst_w_ = length_;
  src_size_ = src_h_ * src_w_ * ic_;

  group_id_in_ = bank_id_in_ / pp_;
  group_id_out_ = bank_id_out_ / pp_;

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_ * kernel_d_);

  weights_data_size_ = (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) ? 2 : 1;
  weights_.resize(oc_ * kernel_h_ * kernel_w_ * kernel_d_ * weights_data_size_);
  weights_16bits_.resize(oc_ * kernel_h_ * kernel_w_ * kernel_d_);

  prelu_.resize(oc_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
  if (exec_mode_ == Calc::ALU_TYPE_MAXPOOL ||
      exec_mode_ == Calc::ALU_TYPE_MAXREDUCE) {
    std::fill(rlt_dtype_.begin(), rlt_dtype_.end(), MIN_DTYPE_VALUE);
  }
}

template <>
Alu<DPUVersion::XV2DPU>::Alu(int inst_type, int instid,
                             vector<string>& inst_str,
                             vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_ALU);
  Helper<DPUVersion::XV2DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XV2DPU>::inst_table;

  reuse_ = field_val_[TABLE::ALU_FIELD_REUSE];
  macc_cont_ = field_val_[TABLE::ALU_FIELD_MACC_CONT];
  pad_left_ = field_val_[TABLE::ALU_FIELD_PAD_LEFT];
  pad_top_ = field_val_[TABLE::ALU_FIELD_PAD_TOP];
  pad_right_ = field_val_[TABLE::ALU_FIELD_PAD_RIGHT_MT];

  pad_bottom_ = field_val_[TABLE::ALU_FIELD_PAD_BOTTOM_MT];

  // AluInit value
  using ALUINIT = AluInit<DPUVersion::XV2DPU>;
  exec_mode_ = ALUINIT::CUROBJ->get_exec_mode();
  num_ = ALUINIT::CUROBJ->get_num();
  share_channel_group_ = ALUINIT::CUROBJ->get_share_channel_group();
  share_kernel_ = ALUINIT::CUROBJ->get_share_kernel();
  kernel_h_ = ALUINIT::CUROBJ->get_kernel_h();
  kernel_w_ = ALUINIT::CUROBJ->get_kernel_w();

  stride_h_ = ALUINIT::CUROBJ->get_stride_h();
  stride_w_ = ALUINIT::CUROBJ->get_stride_w();
  b_mode_ = ALUINIT::CUROBJ->get_b_mode();
  tile_owg_ = ALUINIT::CUROBJ->get_tile_owg();
  tile_ohg_ = ALUINIT::CUROBJ->get_tile_ohg();

  pad_type_ = ALUINIT::CUROBJ->get_pad_type();

  tile_cg_ = ALUINIT::CUROBJ->get_tile_cg();
  ow_iter_ = ALUINIT::CUROBJ->get_ow_iter();
  ow_offset_ = ALUINIT::CUROBJ->get_ow_offset();
  shift_hswish_ = ALUINIT::CUROBJ->get_shift_hswish();
  shift_hsigmoid_ = ALUINIT::CUROBJ->get_shift_hsigmoid();
  hsigmoid_in_ = ALUINIT::CUROBJ->get_hsigmoid_in();

  act_type_ = ALUINIT::CUROBJ->get_act_type();
  shift_bias_ = ALUINIT::CUROBJ->get_shift_bias();
  shift_cut_ = ALUINIT::CUROBJ->get_shift_cut();
  weights_lines_ = ALUINIT::CUROBJ->get_weights_lines();

  UNI_LOG_CHECK(ALUINIT::left_alu_num-- > 0, SIM_PARAMETER_FAILED)
      << "the num of alus following the certain aluInit is larger than alu_num "
         "in that aluInit!";

  // aluaddr value
  using ALUADDR = AluAddr<DPUVersion::XV2DPU>;
  ALUADDR::check();

  auto gen_aluaddr_attrs_in_and_out = [this](const alu_addr_t& alu_addr) {
    auto type = (alu_addr_type_t)alu_addr.type;
    auto invalid = alu_addr.invalid;
    auto id = alu_addr.id;
    auto jump = alu_addr.jump;
    auto jump_endl = alu_addr.jump_endl;
    auto share_pp = alu_addr.jump_endl == 0;
    auto h_num = alu_addr.h_num;
    auto mt_addr = alu_addr.mt_addr;
    switch (type) {
      case alu_addr_type_t::ALU_ADDR_IFM:
        if (1 == invalid) return;
        this->jump_read_ = jump;
        this->jump_read_endl_ = jump_endl;
        for (auto i = 0; i < h_num; ++i) {
          this->aluaddr_jump_read_[id].push_back(jump);
          if ((exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
               exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) &&
              (1 == share_pp)) {
            this->mt_addr_in_[id].push_back(mt_addr);
          } else {
            this->mt_addr_in_[id].push_back(mt_addr + i * jump_endl);
          }
        }
        break;
      case alu_addr_type_t::ALU_ADDR_WGT:
        bank_addr_weights_ = mt_addr;
        jump_read_weights_ = jump;
        break;
      case alu_addr_type_t::ALU_ADDR_BIAS:
        bank_addr_bias_ = mt_addr;
        break;
      case alu_addr_type_t::ALU_ADDR_OFM:
        if (1 == invalid) {
          UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
              << "invalid addr in output aluaddrs " << endl;
        }
        this->jump_write_ = jump;
        this->jump_write_endl_ = jump_endl;
        for (auto i = 0; i < h_num; ++i) {
          this->aluaddr_jump_write_.push_back(jump);
          this->mt_addr_out_.push_back(mt_addr + i * jump_endl);
        }
        break;
    }
  };
  oh_offset_ = 0;
  auto num = ALUADDR::get_num();
  for (auto idx = 0; idx < num; ++idx) {
    auto alu_addr = ALUADDR::get_alu_addr_info(idx);
    gen_aluaddr_attrs_in_and_out(alu_addr);

    auto id = ALUADDR::get_id(idx);
    aluaddr_shift_read_[id] = ALUINIT::CUROBJ->get_shift_read(id);

    auto type = ALUADDR::get_type(idx);
    auto h_num = ALUADDR::get_h_num(idx);
    if (type == alu_addr_type_t::ALU_ADDR_OFM) {
      out_h_num_list_.push_back(h_num);
    } else if (type == alu_addr_type_t::ALU_ADDR_IFM) {
      in_h_num_map_[id].push_back(h_num);
    }
  }
  ALUADDR::clear();

  if (exec_mode_ == Calc::ALU_TYPE_MACC || exec_mode_ == Calc::ALU_TYPE_COMP) {
    UNI_LOG_CHECK(1 == num_, VART_INVALID_VALUE)
        << "num in ALU of macc or comp should be 1";
    UNI_LOG_CHECK(1 == mt_addr_in_.size() && 1 == mt_addr_in_.count(0),
                  VART_INVALID_VALUE)
        << "id in ALUADDR of macc or comp should be 0";
  }

  if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
      exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    UNI_LOG_CHECK((int32_t)(mt_addr_in_.size()) == num_, VART_INVALID_VALUE)
        << "id's set size isn't equal to num_";
    shift_write_ = shift_cut_;
    shift_write_bit_width_ = 7;
  }

  bank_id_out_ = 0;
  bank_id_in_ = 0;

  pp_alu_ = ArchCfg::Instance().get_param().alu_engine().pixel_parallel();
  pp_ = ArchCfg::Instance().get_param().alu_engine().pixel_parallel();
  cp_ = ArchCfg::Instance().get_param().alu_engine().channel_parallel();
  owp_ = 8;

  dst_h_ = mt_addr_out_.size();
  oh_offset_ = tile_ohg_ * pp_ - dst_h_;
  UNI_LOG_CHECK(ow_offset_ < (tile_owg_ * owp_), VART_INVALID_VALUE)
      << "ow_offset " << ow_offset_ << "is not less than (tile_owg_ * owp_) "
      << (tile_owg_ * owp_);
  dst_w_ = ow_iter_ * tile_owg_ * owp_ - ow_offset_;

  // adapt for common pad()
  pad_right_ = pad_right_ ? (pad_right_ - stride_w_ * ow_offset_) : 0;
  pad_right_ = kernel_w_ - pad_right_ - 1;
  pad_bottom_ = pad_bottom_ ? (pad_bottom_ - stride_h_ * oh_offset_) : 0;
  pad_bottom_ = kernel_h_ - pad_bottom_ - 1;

  cg_ = tile_cg_;
  ic_ = cg_ * cp_;
  oc_ = ic_;

  if (exec_mode_ == Calc::ALU_TYPE_MACC || exec_mode_ == Calc::ALU_TYPE_COMP) {
    src_h_ = (dst_h_ - 1) * stride_h_ + kernel_h_;
    real_src_h_ = src_h_;
    src_w_ = (dst_w_ - 1) * stride_w_ + kernel_w_;
  }
  if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
      exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    src_h_ = dst_h_;
    src_w_ = dst_w_;
  }

  kernel_d_ = 1;
  weights_data_size_ = 1;

  UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED);
  UNI_LOG_CHECK((exec_mode_ <= Calc::ALU_TYPE_ELEW_MUL) &&
                    (exec_mode_ >= Calc::ALU_TYPE_MACC),
                SIM_PARAMETER_FAILED)
      << "alu engine exec_mode_ error, " << exec_mode_
      << ", which range should be [" << Calc::ALU_TYPE_MACC << ","
      << Calc::ALU_TYPE_ELEW_MUL << "]" << endl;
  auto alu_support_list = ArchCfg::Instance().get_alu_support_list();
  if (alu_support_list.find(exec_mode_) == alu_support_list.end()) {
    UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
        << "target not support this exec_mode " << exec_mode_;
  }

  // self defined
  src_size_ = src_h_ * num_ * src_w_ * ic_;
  img_.resize(src_size_);

  weights_.resize(oc_ * kernel_h_ * kernel_w_ * kernel_d_ * weights_data_size_);
  weights_16bits_.resize(oc_ * kernel_h_ * kernel_w_ * kernel_d_);

  prelu_.resize(oc_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
  if (exec_mode_ == Calc::ALU_TYPE_COMP) {
    std::fill(rlt_dtype_.begin(), rlt_dtype_.end(), MIN_DTYPE_VALUE);
  }
}

template <DPUVersion T>
Alu<T>::~Alu() {}

template <>
void Alu<DPUVersion::DPUV2>::target_check() {
  // bank access
  /// alu in
  auto access_white_list =
      ArchCfg::Instance().get_bank_access_white_list("alu-in");
  UNI_LOG_CHECK((access_white_list.count(bank_id_in_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_in (= " << bank_id_in_ << ") does not match target!" << endl;
  access_white_list.clear();
  /// alu out
  access_white_list = ArchCfg::Instance().get_bank_access_white_list("alu-out");
  UNI_LOG_CHECK((access_white_list.count(bank_id_out_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_out (= " << bank_id_out_ << ") does not match target!"
      << endl;
  // alu_limit
  /// kernel size
  auto limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("alu-kernel");
  UNI_LOG_CHECK((limit_white_list.count(kernel_h_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_h (= " << kernel_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(kernel_w_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_w (= " << kernel_w_ << ") does not match target!" << endl;
  limit_white_list.clear();
  /// stride
  limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("alu-stride");
  UNI_LOG_CHECK((limit_white_list.count(stride_h_) > 0), SIM_OUT_OF_RANGE)
      << "stride_h (= " << stride_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(stride_w_) > 0), SIM_OUT_OF_RANGE)
      << "stride_w (= " << stride_w_ << ") does not match target!" << endl;
  // nonlinear
  auto support_list =
      ArchCfg::Instance().get_param().alu_engine().nonlinear().nonlinear_type();

  UNI_LOG_CHECK((((act_type_ == Calc::RELU_TYPE_RELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::relu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_PRELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::prelu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_LEAKY_RELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::leaky_relu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_RELUSIX) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::relu_six) > 0)) ||
                 (act_type_ == Calc::RELU_TYPE_NONE)),
                SIM_PARAMETER_FAILED)
      << "Nonlinear type = " << act_type_ << " is unsupported on "
      << ArchCfg::Instance().get_param().name() << endl;
}

template <>
void Alu<DPUVersion::XVDPU>::target_check() {
  // bank access
  /// alu in
  auto access_white_list =
      ArchCfg::Instance().get_bank_access_white_list("alu-in");
  UNI_LOG_CHECK((access_white_list.count(bank_id_in_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_in (= " << bank_id_in_ << ") does not match target!" << endl;
  access_white_list.clear();
  /// alu out
  access_white_list = ArchCfg::Instance().get_bank_access_white_list("alu-out");
  UNI_LOG_CHECK((access_white_list.count(bank_id_out_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_out (= " << bank_id_out_ << ") does not match target!"
      << endl;
  // alu_limit
  /// kernel size
  auto limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("alu-kernel");
  UNI_LOG_CHECK((limit_white_list.count(kernel_h_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_h (= " << kernel_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(kernel_w_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_w (= " << kernel_w_ << ") does not match target!" << endl;
  limit_white_list.clear();
  /// stride
  limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("alu-stride");
  UNI_LOG_CHECK((limit_white_list.count(stride_h_) > 0), SIM_OUT_OF_RANGE)
      << "stride_h (= " << stride_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(stride_w_) > 0), SIM_OUT_OF_RANGE)
      << "stride_w (= " << stride_w_ << ") does not match target!" << endl;
  // nonlinear
  auto support_list =
      ArchCfg::Instance().get_param().alu_engine().nonlinear().nonlinear_type();

  UNI_LOG_CHECK((((act_type_ == Calc::RELU_TYPE_RELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::relu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_PRELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::prelu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_LEAKY_RELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::leaky_relu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_RELUSIX) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::relu_six) > 0)) ||
                 (act_type_ == Calc::RELU_TYPE_NONE)),
                SIM_PARAMETER_FAILED)
      << "Nonlinear type = " << act_type_ << " is unsupported on "
      << ArchCfg::Instance().get_param().name() << endl;
}

template <>
void Alu<DPUVersion::XV2DPU>::target_check() {
  // bank access
  /// alu in
  auto access_white_list =
      ArchCfg::Instance().get_bank_access_white_list("alu-in");
  UNI_LOG_CHECK((access_white_list.count(bank_id_in_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_in (= " << bank_id_in_ << ") does not match target!" << endl;
  access_white_list.clear();
  /// alu out
  access_white_list = ArchCfg::Instance().get_bank_access_white_list("alu-out");
  UNI_LOG_CHECK((access_white_list.count(bank_id_out_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_out (= " << bank_id_out_ << ") does not match target!"
      << endl;
  // alu_limit
  /// kernel size
  auto limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("alu-kernel");
  UNI_LOG_CHECK((limit_white_list.count(kernel_h_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_h (= " << kernel_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(kernel_w_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_w (= " << kernel_w_ << ") does not match target!" << endl;
  limit_white_list.clear();
  /// stride
  limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("alu-stride");
  UNI_LOG_CHECK((limit_white_list.count(stride_h_) > 0), SIM_OUT_OF_RANGE)
      << "stride_h (= " << stride_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(stride_w_) > 0), SIM_OUT_OF_RANGE)
      << "stride_w (= " << stride_w_ << ") does not match target!" << endl;
  // nonlinear
  auto support_list =
      ArchCfg::Instance().get_param().alu_engine().nonlinear().nonlinear_type();
  UNI_LOG_CHECK((((act_type_ == Calc::RELU_TYPE_RELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::relu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_PRELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::prelu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_LEAKY_RELU) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::leaky_relu) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_RELUSIX) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::relu_six) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_HSIGMOID) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::hsigmoid) > 0)) ||
                 ((act_type_ == Calc::RELU_TYPE_HSWISH) &&
                  (std::count(support_list.begin(), support_list.end(),
                              vitis::ai::Target_Nonlinear::hswish) > 0)) ||
                 (act_type_ == Calc::RELU_TYPE_NONE)),
                SIM_PARAMETER_FAILED)
      << "Nonlinear type = " << act_type_ << " is unsupported on "
      << ArchCfg::Instance().get_param().name() << endl;
  // exec_mode
  UNI_LOG_CHECK(
      (exec_mode_ == Calc::ALU_TYPE_MACC || exec_mode_ == Calc::ALU_TYPE_COMP ||
       exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
       exec_mode_ == Calc::ALU_TYPE_ELEW_MUL),
      SIM_PARAMETER_FAILED)
      << "Alu type = " << act_type_ << " is unsupported on "
      << ArchCfg::Instance().get_param().name() << endl;
}

template <>
void Alu<DPUVersion::DPUV2>::Exec() {
  target_check();
  if (exec_mode_ == Calc::ALU_TYPE_DWCV ||
      exec_mode_ == Calc::ALU_TYPE_DWCVB0 ||
      exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
    // read img, weights, bias data from bank
    read();
    debug_input_fmap("read");
    debug_read_pos_tick();
    debug_input_tick("read");
    // padding for img data, also handle channel offset
    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");
    // do conv operation of the whole feature map
    dw_conv();
    debug_output_fmap(0, "alu_rlt_fmap");
    // add bias to conv result
    if (exec_mode_ != Calc::ALU_TYPE_DWCVB0 &&
        exec_mode_ != Calc::ALU_TYPE_DWCVW16B0) {
      bias();
      debug_output_fmap(0, "bias_rlt_fmap");
    }
    // do shift, trunc operation
    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");
    // save result to bank
    save();
  } else if (exec_mode_ == Calc::ALU_TYPE_PRELU) {
    // read img, weights, prelu data from bank
    read();
    debug_input_fmap("read");
    debug_read_pos_tick();

    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");

    // do prelu
    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");

    save();
  } else {
    // read img
    read();
    debug_input_fmap("read");
    debug_read_pos_tick();

    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");

    // do averaging pool
    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");

    save();
  }
}

template <>
void Alu<DPUVersion::XVDPU>::Exec() {
  target_check();
  if (exec_mode_ == Calc::ALU_TYPE_DWCV ||
      exec_mode_ == Calc::ALU_TYPE_DWCVB0 ||
      exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
    // read img, weights, bias data from bank
    read();
    debug_input_fmap("read");
    debug_read_pos_tick();
    debug_input_tick("read");
    // padding for img data, also handle channel offset
    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");
    // do conv operation of the whole feature map
    dw_conv();
    debug_output_fmap(0, "alu_rlt_fmap");
    // add bias to conv result
    if (exec_mode_ != Calc::ALU_TYPE_DWCVB0 &&
        exec_mode_ != Calc::ALU_TYPE_DWCVW16B0) {
      bias();
      debug_output_fmap(0, "bias_rlt_fmap");
    }
    // do shift, trunc operation
    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");
    // save result to bank
    save();
  } else if (exec_mode_ == Calc::ALU_TYPE_PRELU) {
    // read img, weights, prelu data from bank
    read();
    debug_input_fmap("read");
    debug_read_pos_tick();

    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");

    // do prelu
    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");

    save();
  } else {
    // read img
    read();
    debug_input_fmap("read");
    debug_read_pos_tick();

    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");

    // do averaging pool
    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");

    save();
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::Exec() {
  static int s_instid_=-1;
  s_instid_ = instid_;
  target_check();

  UNI_LOG_CHECK_EQ(s_instid_, instid_, SIM_PARAMETER_FAILED) << "after check";
  read();
  debug_input_fmap("read");
  debug_input_tick("read");
  UNI_LOG_CHECK_EQ(s_instid_, instid_, SIM_PARAMETER_FAILED) << "after read";

  if (exec_mode_ == Calc::ALU_TYPE_MACC) {
    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");
    UNI_LOG_CHECK_EQ(s_instid_, instid_, SIM_PARAMETER_FAILED) << "after pad";

    dw_conv();
    debug_output_fmap(0, "macc_rlt_fmap");

    bias();
    debug_output_fmap(0, "bias_rlt_fmap");

    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");
  } else if (exec_mode_ == Calc::ALU_TYPE_COMP) {
    pad();
    debug_input_fmap("pad");
    debug_input_tick("pad");

    maxpool();
    debug_output_fmap(0, "comp_rlt_fmap");

    transform();
    debug_output_fmap(1, "transform_rlt_fmap");
    debug_output_tick("alu_result_tick");
  } else if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
             exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    eltwise();
    debug_output_fmap_elew();
    debug_output_tick_elew();
  }

  save();

  alu_genToken_PL();
  alu_gen_ctrlpkg();
}

template <DPUVersion T>
void Alu<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}

template <DPUVersion dv>
int Alu<dv>::get_group_base_addr(int addr_group) {
  int ret = bank_addr_in_;
  if (addr_group == 0) {
    ret = bank_addr_in_;
  } else if (addr_group == 1) {
    ret = bank_addr_in_1_;
  } else if (addr_group == 2) {
    ret = bank_addr_in_2_;
  } else if (addr_group == 3) {
    ret = bank_addr_in_3_;
  } else {
    UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
        << "bank group addr in is not enough, needed at least " << addr_group
        << ", infact it privide 4" << endl;
  }
  return ret;
}

template <DPUVersion T>
void Alu<T>::read() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "read start..." << endl;

  // read img data from bank
  // UNI_LOG_INFO << "reading img..." << endl;

  UNI_LOG_CHECK(stride_offset_in_ < stride_h_, SIM_PARAMETER_FAILED)
      << "invalid stride_offset_in";
  auto stride_offset = stride_offset_in_;
  auto img_bank_id = bank_id_in_;
  auto base_bank_id = (bank_id_in_ / pp_) * pp_;
  auto addr_group = 0;
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_)->GetH();
  for (auto idx_src_h = 0; idx_src_h < real_src_h_; idx_src_h++) {
    for (auto idx_src_w = 0; idx_src_w < src_w_; idx_src_w++) {
      auto ddr_addr = (idx_src_h * src_w_ + idx_src_w) * ic_ * kernel_d_;
      auto bank_addr_group_base = get_group_base_addr(addr_group);
      auto bank_addr = (bank_addr_group_base + stride_offset * jump_read_endl_ +
                        idx_src_w * jump_read_) %
                       bank_depth;
      auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(img_bank_id);
      bank->Read(bank_addr, ic_ * kernel_d_, img_.data() + ddr_addr);
    }
    if (++stride_offset == stride_h_) {
      stride_offset = 0;
      if (++img_bank_id == base_bank_id + pp_) {
        img_bank_id = base_bank_id;
        ++addr_group;
      }
    }
  }
  if (exec_mode_ == Calc::ALU_TYPE_AVEPOOL ||
      exec_mode_ == Calc::ALU_TYPE_MAXPOOL ||
      exec_mode_ == Calc::ALU_TYPE_LEAKYRELU ||
      exec_mode_ == Calc::ALU_TYPE_MAXREDUCE ||
      exec_mode_ == Calc::ALU_TYPE_HSIGMOID) {
    return;
  }
  if (exec_mode_ == Calc::ALU_TYPE_PRELU) {
    // read prelu from bank
    // UNI_LOG_INFO << "reading prelu..." << endl;
    auto wgt_bank_group =
        ArchCfg::Instance().get_param().alu_engine().weight_bank();
    shared_ptr<Bank<DPU_DATA_TYPE>> bank =
        Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0);
    bank->Read(bank_addr_weights_, oc_, prelu_.data());
    // UNI_LOG_INFO<< "read finish!" << endl;
    return;
  }

  // read weights from bank
  // UNI_LOG_INFO << "reading weights..." << endl;
  auto wgt_bank_group =
      ArchCfg::Instance().get_param().alu_engine().weight_bank();
  auto bank_depth_wgt =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0)->GetH();
  for (int i = 0; i < 1; i++) {
    int bank_id = i % cp_;
    for (int j = 0; j < kernel_h_; j++) {
      for (int k = 0; k < kernel_w_; k++) {
        for (int d = 0; d < kernel_d_; d++) {
          int ddr_addr =
              (i * kernel_h_ * kernel_w_ * kernel_d_ * ic_ +
               j * kernel_w_ * kernel_d_ * ic_ + k * kernel_d_ * ic_ + d * ic_);
          int bank_addr =
              (bank_addr_weights_ + j * kernel_w_ * jump_read_weights_ +
               k * jump_read_weights_ +
               d * channel_group_ * weights_data_size_) %
              bank_depth_wgt;
          auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group,
                                                                bank_id);
          bank->Read(bank_addr, ic_ * weights_data_size_,
                     weights_.data() + ddr_addr);

          if (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
            auto wgt_bank_id = Buffer<DPU_DATA_TYPE>::Instance()
                                   .GetBank(wgt_bank_group, 0)
                                   ->GetID();
            BankShell::read(wgt_bank_id, ic_, bank_addr,
                            weights_16bits_.data() + ddr_addr);
          }
        }
      }
    }
  }

  if (exec_mode_ == Calc::ALU_TYPE_DWCVB0 ||
      exec_mode_ == Calc::ALU_TYPE_DWCVW16B0)
    return;

  // read bias from bank
  // UNI_LOG_INFO << "reading bias..." << endl;
  {
    auto bias_bank_group =
        ArchCfg::Instance().get_param().alu_engine().bias_bank();
    auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bias_bank_group, 0);
    bank->Read(bank_addr_bias_, oc_, bias_.data());
  }
  // UNI_LOG_INFO << "read finish!" << endl;
}

template <>
void Alu<DPUVersion::XV2DPU>::read() {
  auto img_bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_);
  UNI_LOG_CHECK(img_bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_in_ << " out of range!" << endl;
  auto bank_depth = img_bank->GetH();

  auto* img_ptr = reinterpret_cast<char*>(img_.data());
  for (auto id = 0; id < num_; ++id) {
    for (auto idx_src_h = 0; idx_src_h < src_h_; ++idx_src_h) {
      auto bank_addr = mt_addr_in_.at(id).at(idx_src_h);
      auto jump_read = aluaddr_jump_read_.at(id).at(idx_src_h);
      for (auto idx_src_w = 0; idx_src_w < src_w_; ++idx_src_w) {
        auto ddr_addr = idx_src_h * num_ * src_w_ * ic_ + id * src_w_ * ic_ +
                        idx_src_w * ic_;
        UNI_LOG_CHECK_LT(ddr_addr, img_.size(), SIM_ADDR_OVERFLOW);
        img_bank->Read(bank_addr, ic_, (img_ptr + ddr_addr));
        bank_addr = (bank_addr + jump_read) % bank_depth;
      }
    }
  }

  if (exec_mode_ == Calc::ALU_TYPE_COMP ||
      exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
      exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    return;
  }

  auto weights_bank_id_offset = 0;
  auto weights_bank_name =
      ArchCfg::Instance().get_param().alu_engine().weight_bank();
  auto weights_bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(
      weights_bank_name, weights_bank_id_offset);
  auto weights_bank_depth = weights_bank->GetH();
  auto weights_bank_width = weights_bank->GetW();
  auto* weights_ptr = reinterpret_cast<char*>(weights_.data());
  int ocg = div_ceil(oc_, cp_);
  int weights_basic_block_bank_depth = 1;

  for (int idx_ocg = 0; idx_ocg < ocg; ++idx_ocg) {
    for (int idx_kh = 0; idx_kh < kernel_h_; idx_kh++) {
      for (int idx_kw = 0; idx_kw < kernel_w_; idx_kw++) {
        for (int idx_bank = 0; idx_bank < 2; ++idx_bank) {
          auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(
              weights_bank_name, idx_bank);
          int ddr_addr =
              (idx_kh * kernel_w_ * oc_ + idx_kw * oc_ + idx_ocg * cp_) +
              idx_bank * weights_bank_width;
          int jump_length = (1 == share_kernel_) ? 0 : 1;
          int jump_ocg = (1 == share_channel_group_) ? 0 : jump_read_weights_;
          int bank_addr = (bank_addr_weights_ +
                           (idx_ocg * jump_ocg +
                            (idx_kh * kernel_w_ + idx_kw) * jump_length) *
                               weights_basic_block_bank_depth) %
                          weights_bank_depth;
          UNI_LOG_CHECK_LT(ddr_addr, weights_.size(), SIM_ADDR_OVERFLOW);
          bank->Read(bank_addr, weights_bank_width, (weights_ptr + ddr_addr));
        }
      }
    }
  }

  // TODO: parameterize this
  if (b_mode_ == 2) {
    return;
  }

  // read bias from bank
  auto bias_bank_group_name =
      ArchCfg::Instance().get_param().alu_engine().bias_bank();
  auto bias_bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankH(bias_bank_group_name);
  auto bank_width =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankW(bias_bank_group_name);
  auto* bias_ptr = reinterpret_cast<char*>(bias_.data());
  int idx_start_bank_not_zero = 0;
  int idx_end_bank_not_zero = 1;
  for (int idx_cg = 0; idx_cg < tile_cg_; ++idx_cg) {
    for (int idx_bank = idx_start_bank_not_zero;
         idx_bank <= idx_end_bank_not_zero; ++idx_bank) {
      auto bias_bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(
          bias_bank_group_name, idx_bank);
      int cg_jump = (1 == share_channel_group_) ? 0 : 1;
      int bank_addr_offset = idx_cg * cg_jump;
      int ddr_addr = idx_cg * cp_ + idx_bank * bank_width;
      UNI_LOG_CHECK_LT(ddr_addr, bias_.size(), SIM_ADDR_OVERFLOW);
      bias_bank->Read((bank_addr_bias_ + bank_addr_offset) % bias_bank_depth,
                      bank_width, bias_ptr + ddr_addr);
    }
  }
}

template <DPUVersion dv>
void Alu<dv>::pad() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "pad start..." << endl;
  // padding left
  // UNI_LOG_INFO << "padding left..." << endl;
  if (exec_mode_ == Calc::ALU_TYPE_MAXREDUCE) return;
  DPU_DATA_TYPE pad_value = 0;
  if (exec_mode_ == Calc::ALU_TYPE_MAXPOOL) {
    pad_value = MIN_DTYPE_VALUE;
  }
  for (int i = 0; i < real_src_h_; i++) {
    for (int j = 0; j < pad_left_ && j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        img_[addr] = pad_value;
      }
    }
  }

  // padding top
  // UNI_LOG_INFO << "padding top..." << endl;
  for (int i = 0; i < pad_top_ && i < real_src_h_; i++) {
    for (int j = 0; j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        img_[addr] = pad_value;
      }
    }
  }

  // padding right
  // UNI_LOG_INFO << "padding right..." << endl;
  for (int i = 0; i < real_src_h_; i++) {
    for (int j = src_w_ - kernel_w_ + pad_right_ + 1; j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        img_[addr] = pad_value;
      }
    }
  }

  // padding bottom, it's different from padding right
  // UNI_LOG_INFO << "padding bottom..." << endl;
  for (int i = real_src_h_ - kernel_h_ + pad_bottom_ + 1; i < real_src_h_;
       i++) {
    for (int j = 0; j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        img_[addr] = pad_value;
      }
    }
  }
  // UNI_LOG_INFO << "pad finish!" << endl;

  // handle channel offset
  for (int i = 0; i < real_src_h_; i++) {
    for (int j = 0; j < src_w_; j++) {
      for (int d = 0; d < kernel_d_; d++) {
        int ddr_addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ +
                       d * ic_ + ic_ - co_;
        for (int k = 0; k < co_; k++) {
          img_[ddr_addr + k] = pad_value;
        }
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::pad() {
  DPU_DATA_TYPE pad_value = (exec_mode_ == Calc::ALU_TYPE_COMP &&
                             pad_type_ == Calc::pad_type_t::PAD_TYPE_MIN)
                                ? MIN_DTYPE_VALUE
                                : 0;

  for (int i = 0; i < real_src_h_; i++) {
    for (int j = 0; j < pad_left_ && j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        UNI_LOG_CHECK_LT(addr, img_.size(), SIM_PARAMETER_FAILED)
            << "idx1:" << UNI_LOG_VALUES(i, j, k, src_w_, ic_, kernel_d_);
        img_.at(addr) = pad_value;
      }
    }
  }

  for (int i = 0; i < pad_top_ && i < real_src_h_; i++) {
    for (int j = 0; j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        UNI_LOG_CHECK_LT(addr, img_.size(), SIM_PARAMETER_FAILED)
            << "idx2:" << UNI_LOG_VALUES(i, j, k, src_w_, ic_, kernel_d_);
        img_.at(addr) = pad_value;
      }
    }
  }

  for (int i = 0; i < real_src_h_; i++) {
    for (int j = src_w_ - kernel_w_ + pad_right_ + 1; j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        UNI_LOG_CHECK_LT(addr, img_.size(), SIM_PARAMETER_FAILED)
            << "idx3:" << UNI_LOG_VALUES(i, j, k, src_w_, ic_, kernel_d_);
        img_.at(addr) = pad_value;
      }
    }
  }

  for (int i = real_src_h_ - kernel_h_ + pad_bottom_ + 1; i < real_src_h_;
       i++) {
    for (int j = 0; j < src_w_; j++) {
      for (int k = 0; k < ic_ * kernel_d_; k++) {
        int addr = i * src_w_ * kernel_d_ * ic_ + j * kernel_d_ * ic_ + k;
        UNI_LOG_CHECK_LT(addr, img_.size(), SIM_PARAMETER_FAILED)
            << "idx4:" << UNI_LOG_VALUES(i, j, k, src_w_, ic_, kernel_d_);
        img_.at(addr) = pad_value;
      }
    }
  }
}

template <DPUVersion dv>
void Alu<dv>::dw_conv() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "dwconv start..." << endl;
  for (int i = 0; i + kernel_h_ <= real_src_h_; i += stride_h_) {
    for (int j = 0; j + kernel_w_ <= src_w_; j += stride_w_) {
      for (int k = 0; k < oc_; k++) {
        UNI_LOG_CHECK_NE(kernel_h_ * kernel_w_ * stride_h_ * stride_w_, 0,
                         SIM_PARAMETER_FAILED)
            << UNI_LOG_VALUES(kernel_h_, kernel_w_, stride_h_, stride_w_, i, j,
                              k, instid_);
        dw_conv_one(i, j, k);
      }
    }
  }

  // UNI_LOG_INFO << "dwconv finish!" << endl;
}

template <DPUVersion dv>
void Alu<dv>::dw_conv_one(int idx_src_h, int idx_src_w, int idx_oc) {
  int64_t result = 0;
  bool if_weights_share_channel =
      (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) ? true : false;

  // calculate dwconv value
  for (int i = 0; i < kernel_h_; i++) {
    for (int j = 0; j < kernel_w_; j++) {
      for (int d = 0; d < kernel_d_; d++) {
        int img_addr = (idx_src_h + i) * src_w_ * kernel_d_ * ic_ +
                       (idx_src_w + j) * kernel_d_ * ic_ + d * ic_ + idx_oc;
        int weights_addr = i * kernel_w_ * kernel_d_ * ic_ +
                           j * kernel_d_ * ic_ + d * ic_ + idx_oc;
        weights_addr =
            weights_addr - (if_weights_share_channel ? idx_oc % cp_ : 0);
        int16_t weights = (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0)
                              ? weights_16bits_[weights_addr]
                              : weights_[weights_addr];
        result += img_[img_addr] * weights;
      }
    }
  }

  // assign dwconv value to rlt_s64_ var
  int idx_dst_h = idx_src_h / stride_h_;
  int idx_dst_w = idx_src_w / stride_w_;
  int rlt_addr = idx_dst_h * dst_w_ * oc_ + idx_dst_w * oc_ + idx_oc;
  rlt_s64_.at(rlt_addr) = result;
}

template <DPUVersion dv>
void Alu<dv>::maxpool() {
  for (int i = 0; i + kernel_h_ <= src_h_; i += stride_h_) {
    for (int j = 0; j + kernel_w_ <= src_w_; j += stride_w_) {
      maxpool_one(i, j);
    }
  }
}

template <DPUVersion dv>
int Alu<dv>::translate_shift_cut() {
  return (shift_cut_ >= 8) ? (-1 * (shift_cut_ - 8)) : shift_cut_;
}

template <>
int Alu<DPUVersion::XV2DPU>::translate_shift_cut() {
  return ((shift_cut_ & (1U << 6)) ? -1 : 1) * (shift_cut_ & ((1U << 6) - 1));
}

template <DPUVersion dv>
void Alu<dv>::maxpool_one(int idx_src_h, int idx_src_w) {
  int idx_dst_h = idx_src_h / stride_h_;
  int idx_dst_w = idx_src_w / stride_w_;
  for (int k = 0; k < ic_; k++) {
    int rlt_addr = idx_dst_h * dst_w_ * oc_ + idx_dst_w * oc_ + k;
    for (int i = 0; i < kernel_h_; i++) {
      for (int j = 0; j < kernel_w_; j++) {
        // calculate address
        int img_addr =
            (idx_src_h + i) * src_w_ * ic_ + (idx_src_w + j) * ic_ + k;
        if (rlt_dtype_[rlt_addr] < img_[img_addr])
          rlt_dtype_[rlt_addr] = img_[img_addr];
      }
    }
    rlt_dtype_[rlt_addr] = Calc::DPURound<DPU_DATA_TYPE>(
        static_cast<int>(rlt_dtype_[rlt_addr]) * pow(2, translate_shift_cut()));
  }
}

template <DPUVersion dv>
void Alu<dv>::bias() {
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int rlt_addr = i * dst_w_ * oc_ + j * oc_ + k;
        rlt_s64_[rlt_addr] *= 2.0;
        rlt_s64_[rlt_addr] +=
            (2 != b_mode_) ? ((double)bias_[k] * pow(2, shift_bias_)) : 0;
      }
    }
  }
}

template <DPUVersion dv>
void Alu<dv>::avepool_one(int idx_src_h, int idx_src_w) {
  // init result value
  int idx_dst_h = idx_src_h / stride_h_;
  int idx_dst_w = idx_src_w / stride_w_;

  for (int k = 0; k < ic_; k++) {
    auto tmp = 0.f;
    for (int i = 0; i < kernel_h_; i++) {
      for (int j = 0; j < kernel_w_; j++) {
        // calculate address
        int img_addr =
            (idx_src_h + i) * src_w_ * ic_ + (idx_src_w + j) * ic_ + k;
        tmp += img_[img_addr];
      }
    }
    int rlt_addr = idx_dst_h * dst_w_ * oc_ + idx_dst_w * oc_ + k;
    tmp *= multi_factor_;
    tmp = tmp * 1.0 * pow(2.0, translate_shift_cut());
    rlt_dtype_[rlt_addr] = Calc::DPURound<DPU_DATA_TYPE>(tmp);
  }
}

template <DPUVersion dv>
void Alu<dv>::transform() {
  if (exec_mode_ == Calc::ALU_TYPE_LEAKYRELU) {
    // LOG(INFO) << endl;
    // LOG(INFO) << "leakyrelu start..." << endl;
    multi_factor_ = static_cast<DPU_DATA_TYPE>(multi_factor_);
    for (int i = 0; i < dst_h_; i++) {
      for (int j = 0; j < dst_w_; j++) {
        for (int k = 0; k < oc_; k++) {
          int addr = i * dst_w_ * oc_ + j * oc_ + k;
          double tmp = static_cast<double>(img_[addr]);
          tmp *= tmp < 0 ? multi_factor_ : 1;
          tmp /= tmp < 0 ? pow(2, shift_p_n) : pow(2, shift_p_p);
          rlt_dtype_[addr] = Calc::DPURound<DPU_DATA_TYPE>(tmp);
        }
      }
    }
  } else if (exec_mode_ == Calc::ALU_TYPE_MAXPOOL ||
             exec_mode_ == Calc::ALU_TYPE_MAXREDUCE) {
    // LOG(INFO) << endl;
    // LOG(INFO) << "pool start..." << endl;
    for (int i = 0; i + kernel_h_ <= real_src_h_; i += stride_h_) {
      for (int j = 0; j + kernel_w_ <= src_w_; j += stride_w_) {
        maxpool_one(i, j);
      }
    }
    // LOG(INFO) << "pool finish!" << endl;
  } else if (exec_mode_ == Calc::ALU_TYPE_AVEPOOL) {
    // LOG(INFO) << endl;
    // LOG(INFO) << "pool start..." << endl;
    multi_factor_ = static_cast<DPU_DATA_TYPE>(multi_factor_);
    for (int i = 0; i + kernel_h_ <= real_src_h_; i += stride_h_) {
      for (int j = 0; j + kernel_w_ <= src_w_; j += stride_w_) {
        avepool_one(i, j);
      }
    }
    // LOG(INFO) << "pool finish!" << endl;
  } else if (exec_mode_ == Calc::ALU_TYPE_PRELU) {
    // LOG(INFO) << endl;
    // LOG(INFO) << "prelu start..." << endl;
    for (int i = 0; i < dst_h_; i++) {
      for (int j = 0; j < dst_w_; j++) {
        for (int k = 0; k < oc_; k++) {
          int addr = i * dst_w_ * oc_ + j * oc_ + k;
          double tmp = static_cast<double>(img_[addr]);
          tmp *= tmp < 0 ? prelu_[k] : 1;
          tmp /= tmp < 0 ? pow(2, shift_p_n) : pow(2, shift_p_p);
          rlt_dtype_[addr] = Calc::DPURound<DPU_DATA_TYPE>(tmp);
        }
      }
    }
  } else if (exec_mode_ == Calc::ALU_TYPE_DWCV ||
             exec_mode_ == Calc::ALU_TYPE_DWCVB0 ||
             exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
    // LOG(INFO) << endl;
    // LOG(INFO) << "transform start..." << endl;
    for (int i = 0; i < dst_h_; i++) {
      for (int j = 0; j < dst_w_; j++) {
        for (int k = 0; k < oc_; k++) {
          int addr = i * dst_w_ * oc_ + j * oc_ + k;

          double tmp = rlt_s64_[addr];
          if (exec_mode_ == Calc::ALU_TYPE_DWCV) {
            tmp /= pow(2, (shift_cut_ + 1));
          } else if (exec_mode_ == Calc::ALU_TYPE_DWCVB0) {
            tmp /= pow(2, (shift_cut_));
          } else if (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
            tmp *= 2;
            tmp /= pow(2, (shift_cut_ + 1));
          }

          if (act_type_ == Calc::RELU_TYPE_NONE) {
            // do nothing
          } else if (act_type_ == Calc::RELU_TYPE_RELU) {
            tmp = (tmp < 0) ? 0 : tmp;
          } else if (act_type_ == Calc::RELU_TYPE_LEAKY_RELU) {
            tmp = (tmp < 0) ? (tmp * 26. / 256.) : tmp;
          } else if (act_type_ == Calc::RELU_TYPE_RELUSIX) {
            tmp = (tmp < 0) ? 0 : tmp;
            double thr6 = 6 << 4;
            tmp = (tmp < thr6) ? tmp : thr6;
          } else {
            UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
                << "Not support nonlinear type " << act_type_ << endl;
          }
          rlt_dtype_[addr] = Calc::DPURound<DPU_DATA_TYPE>(tmp);
        }
      }
    }
    // LOG(INFO) << "transform finish!" << endl;
  } else if (exec_mode_ == Calc::ALU_TYPE_HSIGMOID) {
    for (int i = 0; i < dst_h_; i++) {
      for (int j = 0; j < dst_w_; j++) {
        for (int k = 0; k < oc_; k++) {
          int addr = i * dst_w_ * oc_ + j * oc_ + k;
          double tmp = static_cast<double>(img_[addr]);
          tmp = min(pow(2, 32), max(0.0, (tmp * 2731.0f +
                                          3.0f * 2731 * pow(2, shift_bias_)))) /
                pow(2, shift_cut_);
          rlt_dtype_[addr] = Calc::DPURound<DPU_DATA_TYPE>(tmp);
        }
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::eltwise() {
  vector<double> shift_read_factor(num_, 0);
  for (auto i = 0; i < num_; i++)
    shift_read_factor[i] = pow(2, aluaddr_shift_read_.at(i));
  auto shift_write_bound = pow(2, shift_write_bit_width_ - 1);
  double shift_write_factor = 0;
  if (shift_write_ >= shift_write_bound)
    shift_write_factor = 1.0 / pow(2, shift_write_ - shift_write_bound);
  else
    shift_write_factor = pow(2, shift_write_);

  for (int r = 0; r < src_h_; r++) {
    int base_ddr_addr = r * num_ * src_w_ * ic_;
    int base_bank_addr = r * src_w_ * ic_;
    for (int i = 0; i < src_w_; i++) {
      for (int j = 0; j < ic_; j++) {
        // elementwise add
        double tmp = (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD) ? 0 : 1;
        for (int k = 0; k < num_; k++) {
          int data_addr = base_ddr_addr + k * src_w_ * ic_ + i * ic_ + j;
          // NOTE: must convert data tyep to doule
          if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD) {
            tmp += floor((double)img_[data_addr] * 4.0 / shift_read_factor[k]);
          } else if (exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
            tmp *= floor((double)img_[data_addr] * 4.0 / shift_read_factor[k]);
          }
        }
        tmp *= shift_write_factor;
        tmp /= (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD) ? 4.0 : 16.0;

        // trunc result
        if (act_type_ == Calc::RELU_TYPE_NONE) {
          // nothing to do
        } else if (act_type_ == Calc::RELU_TYPE_RELU) {
          tmp = (tmp < 0) ? 0 : tmp;
        } else if (act_type_ == Calc::RELU_TYPE_LEAKY_RELU) {
          tmp = (tmp < 0) ? (tmp * 26. / 256.) : tmp;
        } else if (act_type_ == Calc::RELU_TYPE_HSIGMOID) {
          tmp = dr(tmp);
          tmp = std::min(pow(2, 32),
                         std::max(0.0, (tmp * 2731 +
                                        3 * 2731 * pow(2, hsigmoid_in_)))) *
                pow(2, -shift_hsigmoid_);
        } else {
          UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
              << "Not support nonlinear type " << act_type_ << endl;
        }

        int rlt_addr = base_bank_addr + i * ic_ + j;
        rlt_dtype_[rlt_addr] = Calc::DPURound<DPU_DATA_TYPE>(tmp);
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::transform() {
  if (exec_mode_ == Calc::ALU_TYPE_COMP) {
    UNI_LOG_CHECK(rlt_dtype_.size() == rlt_dtype_.size(), SIM_PARAMETER_FAILED);
    std::copy(rlt_dtype_.begin(), rlt_dtype_.end(), rlt_s64_.begin());
  }
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int addr = i * dst_w_ * oc_ + j * oc_ + k;
        double tmp = rlt_s64_[addr];

        if (exec_mode_ == Calc::ALU_TYPE_MACC) {
          tmp /= pow(2, (shift_cut_ + 1));
        }

        if (act_type_ == Calc::RELU_TYPE_NONE) {
          // do nothing
        } else if (act_type_ == Calc::RELU_TYPE_RELU) {
          tmp = (tmp < 0) ? 0 : tmp;
        } else if (act_type_ == Calc::RELU_TYPE_LEAKY_RELU) {
          tmp = (tmp < 0) ? (tmp * 26. / 256.) : tmp;
        } else if (act_type_ == Calc::RELU_TYPE_RELUSIX) {
          tmp = (tmp < 0) ? 0 : tmp;
          double thr6 = 6 << 4;
          tmp = (tmp < thr6) ? tmp : thr6;
        } else if (act_type_ == Calc::RELU_TYPE_HSIGMOID) {
          tmp = dr(tmp);
          tmp = std::min(pow(2, 32),
                         std::max(0.0, (tmp * 2731 +
                                        3 * 2731 * pow(2, hsigmoid_in_)))) *
                pow(2, -shift_hsigmoid_);
        } else if (act_type_ == Calc::RELU_TYPE_HSWISH) {
          tmp = dr(tmp);
          auto x = tmp;
          auto hsigmoid_x = dr(
              std::min(
                  pow(2, 32),
                  std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
              pow(2, -shift_hsigmoid_));
          tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
        } else {
          UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
              << "Not support nonlinear type " << act_type_ << endl;
        }
        rlt_dtype_[addr] = Calc::DPURound<DPU_DATA_TYPE>(tmp);
      }
    }
  }
}

template <DPUVersion T>
void Alu<T>::save() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "dwconv save start..." << endl;
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_)->GetH();
  auto bank_id_out_offset = bank_id_out_ % pp_;
  auto bank_id_out_base = bank_id_out_ - bank_id_out_ % pp_;
  for (auto idx_h = 0; idx_h < real_dst_h_; idx_h++) {
    // bank change, when (idx_h + stride_offset_out) > stride_out
    auto bank_offset = static_cast<uint32_t>(
        floor((float)(idx_h + stride_offset_out_) / stride_out_));
    // bank group inner idx increase
    auto bank_group_offset = static_cast<uint32_t>(
        floor((float)(bank_id_out_offset + bank_offset) / pp_));
    auto bank_id_this_row =
        bank_id_out_base + (bank_id_out_offset + bank_offset) % pp_;
    auto bank_addr_this_row =
        bank_addr_out_                                        // base_addr
        + bank_group_offset * stride_out_ * jump_write_endl_  // change bank
        + ((idx_h + stride_offset_out_) % stride_out_) *
              jump_write_endl_  // change stride offset
        ;
    auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_this_row);
    for (int idx_w = 0; idx_w < dst_w_; idx_w++) {
      auto bank_addr_this_pixel =
          (bank_addr_this_row + idx_w * jump_write_) % bank_depth;
      auto rlt_addr_this_pixel = idx_h * dst_w_ * oc_ + idx_w * oc_;
      bank->Write(bank_addr_this_pixel, oc_,
                  rlt_dtype_.data() + rlt_addr_this_pixel);
    }
  }
  // UNI_LOG_INFO << "dwconv save finish!" << endl;
}

template <>
void Alu<DPUVersion::XV2DPU>::save() {
  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_);
  auto bank_depth = bank->GetH();
  for (auto idx_h = 0; idx_h < dst_h_; ++idx_h) {
    auto bank_addr = mt_addr_out_.at(idx_h);
    auto jump_write = aluaddr_jump_write_.at(idx_h);
    for (int idx_w = 0; idx_w < dst_w_; ++idx_w) {
      auto ddr_addr = idx_h * dst_w_ * oc_ + idx_w * oc_;
      bank->Write(bank_addr, oc_, rlt_dtype_.data() + ddr_addr);

      if (SimCfg::Instance().get_memory_doubleWrite_check()) {
        auto bank_width = bank->GetW();
        Bankblock bankBlock(bank_id_out_, bank_addr, div_ceil(oc_, bank_width));
        Bankblock::insert(debug_path_, bankBlock, TOBANK::ENGINE2BANK);
      }

      bank_addr = (bank_addr + jump_write) % bank_depth;
    }
  }
}

template <DPUVersion T>
void Alu<T>::debug_input_fmap(string prefix) {
  if (!InstBase::debug_) return;

  string head;

  // make save file name
  stringstream ss;
  // save img data
  string save_name_img;

  ss.clear();
  ss.str("");
  ss << InstBase::debug_path_ << "img_" << prefix << "_fmap."
     << InstBase::instid_;
  ss >> save_name_img;
  vector<DPU_DATA_TYPE> img(ic_ * src_h_ * src_w_);
  Calc::HWC2CHW(src_h_, src_w_, ic_, img_.data(), img.data());
  for (int i = 0; i < ic_; i++) {
    // save input_channel title to file
    ss.clear();
    ss.str("");
    ss << "channel_" << i << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_img, head.data(), head.size(), SM_APPEND);
    for (int j = 0; j < src_h_; j++) {
      int addr = i * src_h_ * src_w_ + j * src_w_;
      Util::SaveDec(save_name_img, img.data() + addr, src_w_, src_w_,
                    SM_APPEND);
    }
  }

  if (exec_mode_ == Calc::ALU_TYPE_AVEPOOL ||
      exec_mode_ == Calc::ALU_TYPE_MAXPOOL ||
      exec_mode_ == Calc::ALU_TYPE_MAXREDUCE ||
      exec_mode_ == Calc::ALU_TYPE_LEAKYRELU)
    return;

  // save prelu data
  if (exec_mode_ == Calc::ALU_TYPE_PRELU) {
    string save_name_prelu;
    ss.clear();
    ss.str("");
    ss << InstBase::debug_path_ << "prelu_" << prefix << "_fmap."
       << InstBase::instid_;
    ss >> save_name_prelu;
    for (int i = 0; i < oc_; i++) {
      // save output_channel title to file
      ss.clear();
      ss.str("");
      ss << "output_channel_" << i << "-------------------------------------";
      ss >> head;
      head += "\n";
      Util::SaveBin(save_name_prelu, head.data(), head.size(), SM_APPEND);
      Util::SaveDec(save_name_prelu, prelu_.data() + i, 1, 1, SM_APPEND);
    }
    return;
  }

  // save weights data
  string save_name_weights;
  ss.clear();
  ss.str("");
  ss << InstBase::debug_path_ << "weights_" << prefix << "_fmap."
     << InstBase::instid_;
  ss >> save_name_weights;
  vector<int16_t> weights(ic_ * kernel_h_ * kernel_w_ * kernel_d_ *
                          weights_data_size_);
  if (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
    Calc::HWC2CHW(kernel_h_, kernel_w_ * kernel_d_, ic_, weights_16bits_.data(),
                  weights.data());
  } else {
    Calc::HWC2CHW(kernel_h_, kernel_w_ * kernel_d_, ic_, weights_.data(),
                  weights.data());
  }
  for (int j = 0; j < ic_; j++) {
    // save input_channel title to file
    ss.clear();
    ss.str("");
    ss << "input_channel_" << j << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_weights, head.data(), head.size(), SM_APPEND);
    for (int k = 0; k < kernel_h_; k++) {
      int addr =
          j * kernel_h_ * kernel_w_ * kernel_d_ + k * kernel_w_ * kernel_d_;
      Util::SaveDec(save_name_weights, weights.data() + addr,
                    kernel_w_ * kernel_d_, kernel_w_ * kernel_d_, SM_APPEND);
    }
  }
  if (exec_mode_ == Calc::ALU_TYPE_DWCVB0 ||
      exec_mode_ == Calc::ALU_TYPE_DWCVW16B0)
    return;
  // save bias data
  string save_name_bias;
  ss.clear();
  ss.str("");
  ss << InstBase::debug_path_ << "bias_" << prefix << "_fmap."
     << InstBase::instid_;
  ss >> save_name_bias;
  for (int i = 0; i < oc_; i++) {
    // save output_channel title to file
    ss.clear();
    ss.str("");
    ss << "output_channel_" << i << "-------------------------------------";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_bias, head.data(), head.size(), SM_APPEND);
    Util::SaveDec(save_name_bias, bias_.data() + i, 1, 1, SM_APPEND);
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_input_fmap(string prefix) {
  if (exec_mode_ == Calc::ALU_TYPE_MACC || exec_mode_ == Calc::ALU_TYPE_COMP) {
    debug_input_fmap_macc_comp(prefix);
  } else if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
             exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    debug_input_fmap_elew(prefix);
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_input_fmap_elew(string prefix) {
  if (!debug_) return;

  string head;
  string save_name;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "elew_input_fmap." << instid_;
  ss >> save_name;

  // save elew data
  vector<DPU_DATA_TYPE> tmp(ic_ * num_ * src_w_);
  for (int r = 0; r < src_h_; r++) {
    int addr_offset = r * num_ * src_w_ * ic_;
    Calc::HWC2CHW(num_, src_w_, ic_, img_.data() + addr_offset, tmp.data());
    ss.clear();
    ss.str("");
    ss << "parallel_" << r << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name, head.data(), head.size(), SM_APPEND);
    for (int i = 0; i < ic_; i++) {
      // save input_channel title to file
      ss.clear();
      ss.str("");
      ss << "input_channel_" << i << ":";
      ss >> head;
      head += "\n";
      Util::SaveBin(save_name, head.data(), head.size(), SM_APPEND);
      for (int j = 0; j < num_; j++) {
        int addr = i * num_ * src_w_ + j * src_w_;
        Util::SaveDec(save_name, tmp.data() + addr, src_w_, src_w_, SM_APPEND);
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_input_fmap_macc_comp(string prefix) {
  if (!InstBase::debug_) return;

  string head;

  // make save file name
  stringstream ss;
  // save img data
  string save_name_img;

  ss.clear();
  ss.str("");
  ss << InstBase::debug_path_ << "img_" << prefix << "_fmap."
     << InstBase::instid_;
  ss >> save_name_img;
  vector<DPU_DATA_TYPE> img(ic_ * src_h_ * src_w_);
  Calc::HWC2CHW(src_h_, src_w_, ic_, img_.data(), img.data());
  for (int i = 0; i < ic_; i++) {
    // save input_channel title to file
    ss.clear();
    ss.str("");
    ss << "channel_" << i << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_img, head.data(), head.size(), SM_APPEND);
    for (int j = 0; j < src_h_; j++) {
      int addr = i * src_h_ * src_w_ + j * src_w_;
      Util::SaveDec(save_name_img, img.data() + addr, src_w_, src_w_,
                    SM_APPEND);
    }
  }

  if (exec_mode_ == Calc::ALU_TYPE_COMP) {
    return;
  }

  // save weights data
  string save_name_weights;
  ss.clear();
  ss.str("");
  ss << InstBase::debug_path_ << "weights_" << prefix << "_fmap."
     << InstBase::instid_;
  ss >> save_name_weights;
  vector<int16_t> weights(ic_ * kernel_h_ * kernel_w_ * kernel_d_ *
                          weights_data_size_);
  if (exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
    Calc::HWC2CHW(kernel_h_, kernel_w_ * kernel_d_, ic_, weights_16bits_.data(),
                  weights.data());
  } else {
    Calc::HWC2CHW(kernel_h_, kernel_w_ * kernel_d_, ic_, weights_.data(),
                  weights.data());
  }
  for (int j = 0; j < ic_; j++) {
    // save input_channel title to file
    ss.clear();
    ss.str("");
    ss << "input_channel_" << j << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_weights, head.data(), head.size(), SM_APPEND);
    for (int k = 0; k < kernel_h_; k++) {
      int addr =
          j * kernel_h_ * kernel_w_ * kernel_d_ + k * kernel_w_ * kernel_d_;
      Util::SaveDec(save_name_weights, weights.data() + addr,
                    kernel_w_ * kernel_d_, kernel_w_ * kernel_d_, SM_APPEND);
    }
  }

  if (b_mode_ == 2) {
    return;
  }

  // save bias data
  string save_name_bias;
  ss.clear();
  ss.str("");
  ss << InstBase::debug_path_ << "bias_" << prefix << "_fmap."
     << InstBase::instid_;
  ss >> save_name_bias;
  for (int i = 0; i < oc_; i++) {
    // save output_channel title to file
    ss.clear();
    ss.str("");
    ss << "output_channel_" << i << "-------------------------------------";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_bias, head.data(), head.size(), SM_APPEND);
    Util::SaveDec(save_name_bias, bias_.data() + i, 1, 1, SM_APPEND);
  }
}

template <DPUVersion T>
void Alu<T>::debug_read_pos_tick() {
  if (!debug_) {
    return;
  }
  string save_name;
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "alu_read_pos_tick." << instid_;
  ss >> save_name;
  ofstream fout(save_name);
  UNI_LOG_CHECK(stride_offset_in_ < stride_h_, SIM_PARAMETER_FAILED)
      << "invalid stride_offset_in";
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_)->GetH();
  auto bank_id_group_base = (bank_id_in_ / pp_) * pp_;
  auto bank_id_kernel_base = bank_id_in_ - bank_id_group_base;
  for (auto idx_length = 0; idx_length < dst_w_; idx_length++) {
    for (auto idx_channel_group = 0; idx_channel_group < cg_;
         idx_channel_group++) {
      for (auto idx_kernel_h = 0; idx_kernel_h < kernel_h_; idx_kernel_h++) {
        for (auto idx_kernel_w = 0; idx_kernel_w < kernel_w_; idx_kernel_w++) {
          auto stride_offset = (stride_offset_in_ + idx_kernel_h) % stride_h_;
          auto bank_id_pixel_base =
              (bank_id_kernel_base +
               (stride_offset_in_ + idx_kernel_h) / stride_h_) %
              pp_;
          auto addr_group_pixel_base =
              (bank_id_kernel_base +
               (stride_offset_in_ + idx_kernel_h) / stride_h_) /
              pp_;
          for (auto idx_pixel_parallel = 0; idx_pixel_parallel < pp_alu_;
               idx_pixel_parallel++) {
            auto bank_id = bank_id_group_base +
                           (bank_id_pixel_base + idx_pixel_parallel) % pp_;
            auto addr_group = addr_group_pixel_base +
                              (bank_id_pixel_base + idx_pixel_parallel) / pp_;
            auto bank_addr_group_base = get_group_base_addr(addr_group);
            auto bank_addr =
                (bank_addr_group_base + idx_length * stride_w_ * jump_read_ +
                 stride_offset * jump_read_endl_ + idx_kernel_w * jump_read_ +
                 idx_channel_group) %
                bank_depth;
            fout << setw(2) << setfill('0') << bank_id << "-" << setw(4)
                 << setfill('0') << bank_addr << " ";
          }
          fout << endl;
        }
      }
    }
  }
}

template <DPUVersion T>
void Alu<T>::debug_input_tick(string prefix) {
  if (!debug_) return;

  string head;
  string save_name_img;
  string save_name_wgt;
  string save_name_bias;
  string save_name_prelu;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_img_tick." << instid_;
  ss >> save_name_img;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_wgt_tick." << instid_;
  ss >> save_name_wgt;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_bias_tick." << instid_;
  ss >> save_name_bias;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_prelu_tick." << instid_;
  ss >> save_name_prelu;

  if (exec_mode_ == Calc::ALU_TYPE_PRELU) {
    auto addr_prelu = bank_addr_weights_;
    auto prelu_bank_group =
        ArchCfg::Instance().get_param().alu_engine().weight_bank();
    auto prelu_bank =
        Buffer<DPU_DATA_TYPE>::Instance().GetBank(prelu_bank_group, 0);
    auto id_prelu = prelu_bank->GetID();
    auto bank_depth_prelu = prelu_bank->GetH();
    for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
      // save prelu data in smallend mode
      Util::SaveHexContSmallEndBankAddr(
          save_name_prelu,
          reinterpret_cast<const char*>(prelu_.data() + idx_cg * cp_),
          cp_ * sizeof(DPU_DATA_TYPE), cp_ * sizeof(DPU_DATA_TYPE),
          bank_depth_prelu, id_prelu, addr_prelu++, SM_APPEND);
    }
  }

  if (exec_mode_ == Calc::ALU_TYPE_DWCV) {
    auto addr_bias = bank_addr_bias_;
    auto bias_bank_group =
        ArchCfg::Instance().get_param().alu_engine().bias_bank();
    auto bias_bank =
        Buffer<DPU_DATA_TYPE>::Instance().GetBank(bias_bank_group, 0);
    auto id_bias = bias_bank->GetID();
    auto bank_depth_bias = bias_bank->GetH();
    for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
      // save bias data in smallend mode
      Util::SaveHexContSmallEndBankAddr(
          save_name_bias,
          reinterpret_cast<const char*>(bias_.data() + idx_cg * cp_),
          cp_ * sizeof(DPU_DATA_TYPE), cp_ * sizeof(DPU_DATA_TYPE),
          bank_depth_bias, id_bias, addr_bias++, SM_APPEND);
    }
  }

  // save img and weights data in smallend mode
  auto bank_id_group_base = (bank_id_in_ / pp_) * pp_;
  auto bank_id_offset_base = bank_id_in_ - bank_id_group_base;
  auto bank_depth_img =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_)->GetH();
  for (auto idx_stride_w = 0; idx_stride_w + kernel_w_ <= src_w_;
       idx_stride_w += stride_w_) {
    for (auto idx_channel_group = 0; idx_channel_group < cg_;
         idx_channel_group++) {
      for (auto idx_kernel_h = 0; idx_kernel_h < kernel_h_; idx_kernel_h++) {
        auto stride_offset = (stride_offset_in_ + idx_kernel_h) % stride_h_;
        for (auto idx_kernel_w = 0; idx_kernel_w < kernel_w_; idx_kernel_w++) {
          for (auto idx_vpp = 0; idx_vpp < vpp_; idx_vpp++) {
            // generate bank_id & bank_addr
            auto bank_id_offset =
                (stride_offset_in_ + idx_vpp * stride_h_ + idx_kernel_h) /
                stride_h_;
            auto bank_id_img = bank_id_group_base +
                               (bank_id_offset_base + bank_id_offset) % pp_;
            auto wrap_back = (bank_id_offset_base + bank_id_offset) / pp_;
            auto addr_base = get_group_base_addr(wrap_back);
            auto bank_addr_img = (addr_base + stride_offset * jump_read_endl_ +
                                  (idx_stride_w + idx_kernel_w) * jump_read_ +
                                  idx_channel_group) %
                                 bank_depth_img;
            // prepare data
            vector<DPU_DATA_TYPE> tmp_tick(cp_);
            for (auto idx_channel_parallel = 0; idx_channel_parallel < cp_;
                 idx_channel_parallel++) {
              auto ori_addr = (idx_vpp * stride_h_ + idx_kernel_h) * src_w_ *
                                  cg_ * cp_ +  // source_h offset
                              (idx_stride_w * stride_w_ + idx_kernel_w) * cg_ *
                                  cp_ +  // source_w offset
                              idx_channel_group * cp_;
              tmp_tick[idx_channel_parallel] =
                  img_[ori_addr + idx_channel_parallel];
            }
            Util::SaveHexContSmallEndBankAddr(
                save_name_img, reinterpret_cast<const char*>(tmp_tick.data()),
                tmp_tick.size() * sizeof(DPU_DATA_TYPE),
                cp_ * sizeof(DPU_DATA_TYPE), bank_depth_img, bank_id_img,
                bank_addr_img, SM_APPEND);
          }
          if (exec_mode_ == Calc::ALU_TYPE_DWCV ||
              exec_mode_ == Calc::ALU_TYPE_DWCVB0 ||
              exec_mode_ == Calc::ALU_TYPE_DWCVW16B0) {
            // weights data order ohwi: cp_/kernel_h_/kernel_w_/cg_*cp_
            auto wgt_bank_group =
                ArchCfg::Instance().get_param().alu_engine().weight_bank();
            auto bank_wgt =
                Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0);
            auto bank_id_weights = bank_wgt->GetID();
            auto bank_depth_wgt = bank_wgt->GetID();
            auto bank_addr_weights_base =
                bank_addr_weights_ + idx_kernel_h * jump_read_weights_ +
                idx_kernel_w * kernel_h_ * jump_read_weights_ +
                idx_channel_group;
            auto bank_addr_weights = bank_addr_weights_base % bank_depth_wgt;

            auto ori_weights_addr =
                idx_kernel_h * kernel_w_ * jump_read_weights_ * cp_ +
                idx_kernel_w * jump_read_weights_ * cp_ +
                idx_channel_group * cp_;
            Util::SaveHexContSmallEndBankAddr(
                save_name_wgt,
                reinterpret_cast<const char*>(weights_.data() +
                                              ori_weights_addr),
                cp_ * weights_data_size_, cp_, bank_depth_wgt, bank_id_weights,
                bank_addr_weights, SM_APPEND);
          }
        }
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_input_tick(string prefix) {
  if (exec_mode_ == Calc::ALU_TYPE_MACC || exec_mode_ == Calc::ALU_TYPE_COMP) {
    debug_input_tick_macc_comp(prefix);
  } else if (exec_mode_ == Calc::ALU_TYPE_ELEW_ADD ||
             exec_mode_ == Calc::ALU_TYPE_ELEW_MUL) {
    debug_input_tick_elew(prefix);
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_input_tick_elew(string prefix) {
  if (!debug_) return;

  string head;
  string save_name;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "elew_input_tick." << instid_;
  ss >> save_name;

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_);
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();
  auto bank_delta = static_cast<int32_t>(std::ceil(cp_ * 1.0 / bank_width));

  for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
    for (auto idx_w = 0; idx_w < src_w_; idx_w++) {
      for (auto idx_num = 0; idx_num < num_; idx_num++) {
        auto bank_id = bank_id_in_;
        for (auto idx_vpp = 0; idx_vpp < src_h_; idx_vpp++) {
          auto ori_offset =
              ((idx_vpp * num_ + idx_num) * src_w_ + idx_w) * ic_ +
              idx_cg * cp_;
          UNI_LOG_CHECK(ori_offset + cp_ <= (int)(img_.size()),
                        SIM_OUT_OF_RANGE)
              << ori_offset + cp_ << " > " << img_.size();

          auto bank_addr = mt_addr_in_.at(idx_num).at(idx_vpp);
          bank_addr += (idx_w * aluaddr_jump_read_.at(idx_num).at(idx_vpp) +
                        idx_cg * bank_delta);
          bank_addr %= bank_depth;
          Util::SaveHexContSmallEndBankAddr(
              save_name,
              reinterpret_cast<const char*>(img_.data() + ori_offset), cp_,
              bank_width, bank_depth, bank_id, bank_addr, SM_APPEND);
        }
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_input_tick_macc_comp(string prefix) {
  if (!debug_) return;

  string head;
  string save_name_img;
  string save_name_wgt;
  string save_name_bias;
  string save_name_prelu;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_img_tick." << instid_;
  ss >> save_name_img;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_wgt_tick." << instid_;
  ss >> save_name_wgt;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_bias_tick." << instid_;
  ss >> save_name_bias;
  ss.clear();
  ss.str("");
  ss << debug_path_ << prefix << "_prelu_tick." << instid_;
  ss >> save_name_prelu;

  // save img and weights data in smallend mode
  auto img_bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_);
  auto img_bank_depth = img_bank->GetH();
  auto img_bank_width = img_bank->GetW();
  auto bank_delta = static_cast<int32_t>(std::ceil(cp_ * 1.0 / img_bank_width));
  auto tile_ow = tile_owg_ * owp_;
  auto* img_ptr = reinterpret_cast<const char*>(img_.data());
  for (auto idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    for (auto idx_ohg = 0; idx_ohg < tile_ohg_; ++idx_ohg) {
      for (auto idx_cg = 0; idx_cg < tile_cg_; ++idx_cg) {
        for (auto idx_kh = 0; idx_kh < kernel_h_; ++idx_kh) {
          for (auto idx_kw = 0; idx_kw < kernel_w_; ++idx_kw) {
            for (auto idx_tile_ow = 0; idx_tile_ow < tile_ow; ++idx_tile_ow) {
              auto idx_dst_w = idx_ow_iter * tile_ow + idx_tile_ow;
              if (idx_dst_w >= dst_w_) continue;
              auto idx_src_w = idx_dst_w * stride_w_ + idx_kw;
              for (auto idx_pp = 0; idx_pp < pp_alu_; ++idx_pp) {
                auto idx_dst_h = idx_ohg * pp_alu_ + idx_pp;
                if (idx_dst_h >= dst_h_) continue;
                auto idx_src_h = idx_dst_h * stride_h_ + idx_kh;
                auto jump_read = aluaddr_jump_read_.at(0).at(idx_src_h);
                auto bank_addr = (mt_addr_in_.at(0).at(idx_src_h) +
                                  idx_src_w * jump_read + idx_cg * bank_delta) %
                                 img_bank_depth;
                auto ddr_addr =
                    (idx_src_h * src_w_ * cg_ + idx_src_w * cg_ + idx_cg) * cp_;
                UNI_LOG_CHECK(ddr_addr + cp_ <= (int)(img_.size()),
                              SIM_OUT_OF_RANGE)
                    << ddr_addr + cp_ << " > " << img_.size();
                Util::SaveHexContSmallEndBankAddr(
                    save_name_img, (img_ptr + ddr_addr), cp_, img_bank_width,
                    img_bank_depth, bank_id_in_, bank_addr, SM_APPEND);
              }
            }
          }
        }
      }
    }
  }

  if (exec_mode_ == Calc::ALU_TYPE_COMP) {
    return;
  }

  auto weights_bank_group =
      ArchCfg::Instance().get_param().alu_engine().weight_bank();
  auto weights_bank_num =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankNum(weights_bank_group);
  auto weights_bank =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(weights_bank_group, 0);
  auto weights_base_bank_id = weights_bank->GetID();
  auto weights_bank_depth = weights_bank->GetH();
  auto weights_bank_width = weights_bank->GetW();
  auto weights_size_scale = 2;
  int basic_block_bank_depth =
      div_ceil(cp_ * weights_size_scale, weights_bank_num * weights_bank_width);
  auto* weights_ptr = reinterpret_cast<const char*>(weights_.data());
  for (auto idx_cg = 0; idx_cg < tile_cg_; ++idx_cg) {
    for (auto idx_kh = 0; idx_kh < kernel_h_; ++idx_kh) {
      for (auto idx_kw = 0; idx_kw < kernel_w_; ++idx_kw) {
        int jump_length = (1 == share_kernel_) ? 0 : 1;
        int jump_ocg = (1 == share_channel_group_) ? 0 : jump_read_weights_;
        int weights_bank_addr =
            (bank_addr_weights_ +
             (idx_cg * jump_ocg + (idx_kh * kernel_w_ + idx_kw) * jump_length) *
                 basic_block_bank_depth) %
            weights_bank_depth;
        for (int idx_bank = 0; idx_bank < 2; ++idx_bank) {
          int wgts_bank_id = weights_base_bank_id + idx_bank;
          auto ddr_addr =
              (idx_kh * kernel_w_ * tile_cg_ + idx_kw * tile_cg_ + idx_cg) *
                  cp_ +
              idx_bank * (cp_ / 2);
          UNI_LOG_CHECK(ddr_addr + cp_ / 2 <= (int)(weights_.size()),
                        SIM_OUT_OF_RANGE)
              << ddr_addr + cp_ / 2 << " > " << weights_.size();
          Util::SaveHexContSmallEndBankAddr(
              save_name_wgt, (weights_ptr + ddr_addr), cp_ / 2,
              weights_bank_width, weights_bank_depth, wgts_bank_id,
              weights_bank_addr, SM_APPEND);
        }
      }
    }
  }

  if (b_mode_ == 2) {
    return;
  }

  auto bias_bank_group =
      ArchCfg::Instance().get_param().alu_engine().bias_bank();
  auto bias_bank_num =
      Buffer<DPU_DATA_TYPE>::Instance().GetBankNum(bias_bank_group);
  auto bias_bank =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bias_bank_group, 0);
  auto bias_base_bank_id = bias_bank->GetID();
  auto bias_bank_depth = bias_bank->GetH();
  auto bias_bank_width = bias_bank->GetW();
  auto bias_size_scale = 2;
  int bias_data_block_bank_depth =
      div_ceil(cp_ * bias_size_scale, bias_bank_num * bias_bank_width);
  auto* bias_ptr = reinterpret_cast<const char*>(bias_.data());
  for (int idx_ocg = 0; idx_ocg < tile_cg_; ++idx_ocg) {
    int cg_jump = (1 == share_channel_group_) ? 0 : bias_data_block_bank_depth;
    auto bias_bank_addr = bank_addr_bias_ + idx_ocg * cg_jump;
    for (int idx_bank = 0; idx_bank < 2; ++idx_bank) {
      int bias_bank_id = bias_base_bank_id + idx_bank;
      auto ddr_addr = idx_ocg * cp_ + idx_bank * (cp_ / 2);
      Util::SaveHexContSmallEndBankAddr(
          save_name_bias, bias_ptr + ddr_addr, (cp_ / 2), bias_bank_width,
          bias_bank_depth, bias_bank_id, bias_bank_addr, SM_APPEND);
    }
  }
}

template <DPUVersion T>
void Alu<T>::debug_output_fmap(int type, string save_name) {
  if (!InstBase::debug_) return;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << InstBase::debug_path_ << save_name << "." << InstBase::instid_;
  ss >> save_name;

  // save img data
  vector<int64_t> rlt_s64(oc_ * dst_h_ * dst_w_);
  vector<DPU_DATA_TYPE> rlt_dtype(oc_ * dst_h_ * dst_w_);
  if (type == 0) {
    Calc::HWC2CHW(dst_h_, dst_w_, oc_, rlt_s64_.data(), rlt_s64.data());
  } else {
    Calc::HWC2CHW(dst_h_, dst_w_, oc_, rlt_dtype_.data(), rlt_dtype.data());
  }

  for (int i = 0; i < oc_; i++) {
    // save input_channel title to file
    string head;
    ss.clear();
    ss.str("");
    ss << "channel_" << i << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name, head.data(), head.size(), SM_APPEND);
    for (int j = 0; j < dst_h_; j++) {
      int addr = i * dst_h_ * dst_w_ + j * dst_w_;
      if (type == 0) {
        Util::SaveDec(save_name, rlt_s64.data() + addr, dst_w_, dst_w_,
                      SM_APPEND);
      } else {
        Util::SaveDec(save_name, rlt_dtype.data() + addr, dst_w_, dst_w_,
                      SM_APPEND);
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_output_fmap_elew() {
  if (!debug_) return;

  string head;
  string save_name;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "elew_output_fmap." << instid_;
  ss >> save_name;

  // save elew data
  vector<DPU_DATA_TYPE> tmp(ic_ * 1 * src_w_);
  for (int r = 0; r < src_h_; r++) {
    int addr_off = r * src_w_ * ic_;
    Calc::HWC2CHW(1, src_w_, ic_, rlt_dtype_.data() + addr_off, tmp.data());
    ss.clear();
    ss.str("");
    ss << "parallel_" << r << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name, head.data(), head.size(), SM_APPEND);
    for (int i = 0; i < ic_; i++) {
      // save input_channel title to file
      ss.clear();
      ss.str("");
      ss << "channel_" << i << ":";
      ss >> head;
      head += "\n";
      Util::SaveBin(save_name, head.data(), head.size(), SM_APPEND);

      int addr = i * src_w_;
      Util::SaveDec(save_name, tmp.data() + addr, src_w_, src_w_, SM_APPEND);
    }
  }
}

template <DPUVersion T>
void Alu<T>::debug_output_tick(string save_name) {
  if (!debug_) return;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << save_name << "." << instid_;
  ss >> save_name;

  vector<DPU_DATA_TYPE> rlt_tmp(rlt_dtype_.size());

  // change data order from hwc to wch
  for (int i = 0; i < length_; i++) {
    for (int j = 0; j < cp_; j++) {
      for (int k = 0; k < pp_alu_; k++) {
        int addr_tmp = i * cp_ * pp_alu_ + j * pp_alu_ + k;
        int addr_ori = k * length_ * cp_ + i * cp_ + j;

        rlt_tmp[addr_tmp] = rlt_dtype_[addr_ori];
      }
    }
  }

  // save img
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_)->GetH();
  auto* p = reinterpret_cast<const char*>(rlt_dtype_.data());
  auto line_size = cp_ * sizeof(DPU_DATA_TYPE);
  // bank id of the first bank in this bank_group
  auto bank_id_out_offset = bank_id_out_ % pp_;
  auto bank_id_out_base = bank_id_out_ - bank_id_out_ % pp_;
  for (int idx_w = 0; idx_w < length_; idx_w++) {
    for (auto idx_h = 0; idx_h < real_dst_h_; idx_h++) {
      auto bank_offset = static_cast<uint32_t>(
          floor((float)(idx_h + stride_offset_out_) / stride_out_));
      auto bank_group_offset = static_cast<uint32_t>(
          floor((float)(bank_id_out_offset + bank_offset) / pp_));
      auto bank_id_this_row =
          bank_id_out_base + (bank_id_out_offset + bank_offset) % pp_;
      auto bank_addr_this_row =
          bank_addr_out_                                        // base_addr
          + bank_group_offset * stride_out_ * jump_write_endl_  // change bank
          + ((idx_h + stride_offset_out_) % stride_out_) *
                jump_write_endl_  // change stride offset
          ;
      auto bank_addr_this_pixel =
          (bank_addr_this_row + idx_w * jump_write_) % bank_depth;
      auto offset = (idx_w + idx_h * length_) * line_size;
      Util::SaveHexContSmallEndBankAddr(save_name, p + offset, line_size,
                                        line_size, bank_depth, bank_id_this_row,
                                        bank_addr_this_pixel, SM_APPEND);
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_output_tick(string save_name) {
  if (!debug_) return;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << save_name << "." << instid_;
  ss >> save_name;

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_);
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();
  auto bank_delta = static_cast<int32_t>(std::ceil(cp_ * 1.0 / bank_width));
  auto* output_ptr = reinterpret_cast<const char*>(rlt_dtype_.data());
  auto line_size = cp_ * sizeof(DPU_DATA_TYPE);
  for (auto idx_ow_iter = 0; idx_ow_iter < ow_iter_; ++idx_ow_iter) {
    for (int idx_owg = 0; idx_owg < tile_owg_; ++idx_owg) {
      for (auto idx_cg = 0; idx_cg < tile_cg_; ++idx_cg) {
        for (int idx_owp = 0; idx_owp < owp_; ++idx_owp) {
          auto idx_ow =
              idx_ow_iter * tile_owg_ * owp_ + idx_owg * owp_ + idx_owp;
          if (idx_ow >= dst_w_) continue;
          for (auto idx_dst_h = 0; idx_dst_h < dst_h_; ++idx_dst_h) {
            auto bank_addr = (mt_addr_out_.at(idx_dst_h) +
                              idx_ow * aluaddr_jump_write_.at(idx_dst_h) +
                              idx_cg * bank_delta) %
                             bank_depth;
            auto ddr_addr =
                idx_dst_h * dst_w_ * oc_ + idx_ow * oc_ + idx_cg * cp_;
            UNI_LOG_CHECK(ddr_addr + line_size <= rlt_dtype_.size(),
                          SIM_OUT_OF_RANGE)
                << ddr_addr + line_size << " > " << rlt_dtype_.size();
            Util::SaveHexContSmallEndBankAddr(
                save_name, output_ptr + ddr_addr, line_size, bank_width,
                bank_depth, bank_id_out_, bank_addr, SM_APPEND);
          }
        }
      }
    }
  }
}

template <>
void Alu<DPUVersion::XV2DPU>::debug_output_tick_elew() {
  if (!debug_) return;

  string head;
  string save_name;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "elew_output_tick." << instid_;
  ss >> save_name;

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_);
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();
  auto* output_ptr = reinterpret_cast<const char*>(rlt_dtype_.data());
  for (int i = 0; i < cg_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int r = 0; r < dst_h_; r++) {
        int ddr_addr = r * dst_w_ * cg_ * cp_ + j * cg_ * cp_ + i * cp_;
        int bank_addr = (mt_addr_out_.at(r) + j * aluaddr_jump_write_.at(r) +
                         i * cp_ / bank_width) %
                        bank_depth;
        Util::SaveHexContSmallEndBankAddr(
            save_name, output_ptr + ddr_addr, cp_ * sizeof(DPU_DATA_TYPE),
            cp_ * sizeof(DPU_DATA_TYPE), bank_depth, bank_id_out_, bank_addr,
            SM_APPEND);
      }
    }
  }
}
