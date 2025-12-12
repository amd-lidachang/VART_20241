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
#include "DptWise.hpp"
#include "DWInit.hpp"

// NOTE: it's famous Wundefined-var-template issue,
// if you want to compile using Clang, add instantiation here
template <DPUVersion dv>
DWInit<dv>* DWInit<dv>::CUROBJ;

template <>
DptWise<DPUVersion::DPUV2>::DptWise(int inst_type, int instid,
                                    vector<string>& inst_str,
                                    vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // dwconv instruction fields
  // debug
  debug_ =
      debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::DPUV2>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV2>::inst_table;
  // get dwconv instruction field value
  channel_group_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_GROUP];
  bank_addr_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN];
  pad_left_ = field_val_[TABLE::DPTWISE_FIELD_PAD_LEFT];
  pad_right_ = field_val_[TABLE::DPTWISE_FIELD_PAD_RIGHT];
  pad_top_ = field_val_[TABLE::DPTWISE_FIELD_PAD_TOP];
  pad_bottom_ = field_val_[TABLE::DPTWISE_FIELD_PAD_BOTTOM];
  channel_offset_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_OFFSET];
  bank_id_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_IN];
  bank_id_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_OUT];
  bank_addr_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_OUT];
  bank_addr_weights_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_WEIGHTS];
  length_ = field_val_[TABLE::DPTWISE_FIELD_LENGTH];
  bank_addr_bias_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_BIAS];

  bank_addr_in_1_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_1];
  bank_addr_in_2_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_2];
  bank_addr_in_3_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_3];
  act_type_ = field_val_[TABLE::DPTWISE_FIELD_ACT_TYPE];

  // dwinit value
  using DWINIT = DWInit<DPUVersion::DPUV2>;
  jump_read_ = DWINIT::CUROBJ->get_jump_read();
  jump_write_ = DWINIT::CUROBJ->get_jump_write();
  jump_read_endl_ = DWINIT::CUROBJ->get_jump_read_endl();
  stride_offset_in_ = DWINIT::CUROBJ->get_stride_offset_in();
  valid_pixel_parallel_ = DWINIT::CUROBJ->get_valid_pixel_parallel();
  stride_w_ = DWINIT::CUROBJ->get_stride_w();
  stride_h_ = DWINIT::CUROBJ->get_stride_h();
  kernel_w_ = DWINIT::CUROBJ->get_kernel_w();
  kernel_h_ = DWINIT::CUROBJ->get_kernel_h();
  jump_write_endl_ = DWINIT::CUROBJ->get_jump_write_endl();
  stride_offset_out_ = DWINIT::CUROBJ->get_stride_offset_out();
  stride_out_ = DWINIT::CUROBJ->get_stride_out();
  shift_bias_ = DWINIT::CUROBJ->get_shift_bias();
  shift_cut_ = DWINIT::CUROBJ->get_shift_cut();

  // constraints from hardware implementation
  UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED)
      << "invalid shift_bias";

  // self defined
  vpp_ = valid_pixel_parallel_;
  pp_ = ArchCfg::Instance().get_param().dwconv_engine().pixel_parallel();
  assert(vpp_ <= pp_);
  pp_conv_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  assert(pp_ <= pp_conv_);
  cp_ = ArchCfg::Instance().get_param().dwconv_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel
  src_h_ = (pp_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (vpp_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (length_ - 1) * stride_w_ + kernel_w_;
  real_src_w_ = src_w_;
  dst_h_ = pp_;
  real_dst_h_ = vpp_;
  dst_w_ = length_;
  real_dst_w_ = dst_w_;
  src_size_ = src_h_ * src_w_ * ic_;

  group_id_in_ = bank_id_in_ / pp_conv_;
  group_id_out_ = bank_id_out_ / pp_conv_;

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_);
  weights_.resize(oc_ * kernel_h_ * kernel_w_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
}

template <>
DptWise<DPUVersion::DPUV3E>::DptWise(int inst_type, int instid,
                                     vector<string>& inst_str,
                                     vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // dwconv instruction fields
  // debug
  debug_ =
      debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::DPUV3E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV3E>::inst_table;
  // get dwconv instruction field value
  channel_group_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_GROUP];
  bank_addr_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN];
  pad_left_ = field_val_[TABLE::DPTWISE_FIELD_PAD_LEFT];
  pad_right_ = field_val_[TABLE::DPTWISE_FIELD_PAD_RIGHT];
  pad_top_ = field_val_[TABLE::DPTWISE_FIELD_PAD_TOP];
  pad_bottom_ = field_val_[TABLE::DPTWISE_FIELD_PAD_BOTTOM];
  channel_offset_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_OFFSET];
  bank_id_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_IN];
  bank_id_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_OUT];
  bank_addr_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_OUT];
  bank_addr_weights_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_WEIGHTS];
  length_ = field_val_[TABLE::DPTWISE_FIELD_LENGTH];
  bank_addr_bias_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_BIAS];

  bank_addr_in_1_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_1];
  bank_addr_in_2_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_2];
  bank_addr_in_3_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_3];

  // dwinit value
  using DWINIT = DWInit<DPUVersion::DPUV3E>;
  jump_read_ = DWINIT::CUROBJ->get_jump_read();
  jump_write_ = DWINIT::CUROBJ->get_jump_write();
  jump_read_endl_ = DWINIT::CUROBJ->get_jump_read_endl();
  stride_offset_in_ = DWINIT::CUROBJ->get_stride_offset_in();
  valid_pixel_parallel_ = DWINIT::CUROBJ->get_valid_pixel_parallel();
  stride_w_ = DWINIT::CUROBJ->get_stride_w();
  stride_h_ = DWINIT::CUROBJ->get_stride_h();
  kernel_w_ = DWINIT::CUROBJ->get_kernel_w();
  kernel_h_ = DWINIT::CUROBJ->get_kernel_h();
  jump_write_endl_ = DWINIT::CUROBJ->get_jump_write_endl();
  stride_offset_out_ = DWINIT::CUROBJ->get_stride_offset_out();
  stride_out_ = DWINIT::CUROBJ->get_stride_out();
  shift_bias_ = DWINIT::CUROBJ->get_shift_bias();
  shift_cut_ = DWINIT::CUROBJ->get_shift_cut();
  act_type_ = DWINIT::CUROBJ->get_act_type();

  // constraints from hardware implementation
  if (shift_bias_ < 32) {
    UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED)
        << "invalid shift_bias";
  }

  // self defined
  vpp_ = valid_pixel_parallel_;
  pp_ = ArchCfg::Instance().get_param().dwconv_engine().pixel_parallel();
  assert(vpp_ <= pp_);
  pp_conv_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  assert(pp_ <= pp_conv_);
  cp_ = ArchCfg::Instance().get_param().dwconv_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel
  src_h_ = (pp_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (vpp_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (length_ - 1) * stride_w_ + kernel_w_;
  real_src_w_ = src_w_;
  dst_h_ = pp_;
  real_dst_h_ = vpp_;
  dst_w_ = length_;
  real_dst_w_ = dst_w_;
  src_size_ = src_h_ * src_w_ * ic_;

  group_id_in_ = bank_id_in_ / pp_conv_;
  group_id_out_ = bank_id_out_ / pp_conv_;

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_);
  weights_.resize(oc_ * kernel_h_ * kernel_w_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
}

template <>
DptWise<DPUVersion::DPUV3ME>::DptWise(int inst_type, int instid,
                                      vector<string>& inst_str,
                                      vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // dwconv instruction fields
  // debug
  debug_ =
      debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::DPUV3ME>::InstInit(inst_type_, root_debug_path_, debug_,
                                        opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV3ME>::inst_table;
  channel_group_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_GROUP];
  bank_addr_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN];
  pad_left_ = field_val_[TABLE::DPTWISE_FIELD_PAD_LEFT];
  pad_right_ = field_val_[TABLE::DPTWISE_FIELD_PAD_RIGHT];
  pad_top_ = field_val_[TABLE::DPTWISE_FIELD_PAD_TOP];
  pad_bottom_ = field_val_[TABLE::DPTWISE_FIELD_PAD_BOTTOM];
  channel_offset_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_OFFSET];
  bank_id_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_IN];
  bank_id_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_OUT];
  bank_addr_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_OUT];
  bank_addr_weights_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_WEIGHTS];
  length_ = field_val_[TABLE::DPTWISE_FIELD_LENGTH];
  bank_addr_bias_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_BIAS];

  // dwinit value
  using DWINIT = DWInit<DPUVersion::DPUV3ME>;
  jump_read_ = DWINIT::CUROBJ->get_jump_read();
  jump_write_ = DWINIT::CUROBJ->get_jump_write();
  jump_read_endl_ = DWINIT::CUROBJ->get_jump_read_endl();
  stride_offset_in_ = DWINIT::CUROBJ->get_stride_offset_in();
  valid_pixel_parallel_ = DWINIT::CUROBJ->get_valid_pixel_parallel();
  stride_w_ = DWINIT::CUROBJ->get_stride_w();
  stride_h_ = DWINIT::CUROBJ->get_stride_h();
  kernel_w_ = DWINIT::CUROBJ->get_kernel_w();
  kernel_h_ = DWINIT::CUROBJ->get_kernel_h();
  jump_write_endl_ = DWINIT::CUROBJ->get_jump_write_endl();
  stride_offset_out_ = DWINIT::CUROBJ->get_stride_offset_out();
  stride_out_ = DWINIT::CUROBJ->get_stride_out();
  shift_relusix_ = DWINIT::CUROBJ->get_shift_relusix();
  shift_bias_ = DWINIT::CUROBJ->get_shift_bias();
  shift_cut_ = DWINIT::CUROBJ->get_shift_cut();
  act_type_ = DWINIT::CUROBJ->get_act_type();

  // constraints from hardware implementation
  if (shift_bias_ < 32) {
    UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED)
        << "invalid shift_bias";
  }

  // self defined
  vpp_ = valid_pixel_parallel_;
  pp_ = ArchCfg::Instance().get_param().dwconv_engine().pixel_parallel();
  assert(vpp_ <= pp_);
  pp_conv_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  assert(pp_ <= pp_conv_);
  cp_ = ArchCfg::Instance().get_param().dwconv_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel
  src_h_ = (pp_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (vpp_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (length_ - 1) * stride_w_ + kernel_w_;
  real_src_w_ = src_w_;
  dst_h_ = pp_;
  real_dst_h_ = vpp_;
  dst_w_ = length_;
  real_dst_w_ = dst_w_;
  src_size_ = src_h_ * src_w_ * ic_;

  int icp =
      ArchCfg::Instance().get_param().conv_engine().input_channel_parallel();
  scale_ = cp_ / icp;

  // input bank id check
  int fmap_bank_group = SimCfg::Instance().get_fmap_bank_group();
  int group_id_in = bank_id_in_ / pp_conv_;
  UNI_LOG_CHECK(group_id_in == (fmap_bank_group - 1), SIM_PARAMETER_FAILED)
      << "invalid input bank id";

  UNI_LOG_CHECK(jump_read_ == channel_group_, SIM_PARAMETER_FAILED)
      << "Jump read should equal to channel_group. Now jump_read = "
      << jump_read_ << ", channel_group = " << channel_group_;

  // output bank id check
  int group_id_out = bank_id_out_ / pp_conv_;
  UNI_LOG_CHECK(group_id_out < (fmap_bank_group - 1), SIM_PARAMETER_FAILED)
      << "invalid bank_id for write HBM";

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_);
  weights_.resize(oc_ * kernel_h_ * kernel_w_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
}

template <>
DptWise<DPUVersion::DPUV4E>::DptWise(int inst_type, int instid,
                                     vector<string>& inst_str,
                                     vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // dwconv instruction fields
  // debug
  debug_ =
      debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::DPUV4E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV4E>::inst_table;
  // get dwconv instruction field value
  bank_addr_bias_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_BIAS];
  bank_addr_weights_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_WEIGHTS];

  bank_addr_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_OUT];
  valid_pixel_parallel_ = field_val_[TABLE::DPTWISE_FIELD_VALID_PIXEL_PARALLEL];
  pad_left_ = field_val_[TABLE::DPTWISE_FIELD_PAD_LEFT];
  pad_right_ = field_val_[TABLE::DPTWISE_FIELD_PAD_RIGHT];
  pad_top_ = field_val_[TABLE::DPTWISE_FIELD_PAD_TOP];
  pad_bottom_ = field_val_[TABLE::DPTWISE_FIELD_PAD_BOTTOM];

  bank_addr_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN];
  bank_addr_in_1_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_1];
  bank_addr_in_2_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_2];
  bank_addr_in_3_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_3];

  bank_id_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_IN];
  bank_id_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_OUT];

  // dwinit value
  using DWINIT = DWInit<DPUVersion::DPUV4E>;
  stride_out_ = DWINIT::CUROBJ->get_stride_out();
  stride_w_ = DWINIT::CUROBJ->get_stride_w();
  stride_h_ = DWINIT::CUROBJ->get_stride_h();
  kernel_w_ = DWINIT::CUROBJ->get_kernel_w();
  kernel_h_ = DWINIT::CUROBJ->get_kernel_h();

  length_ = DWINIT::CUROBJ->get_length();
  channel_group_ = DWINIT::CUROBJ->get_channel_group();
  channel_offset_ = DWINIT::CUROBJ->get_channel_offset();
  act_type_ = DWINIT::CUROBJ->get_act_type();

  jump_read_endl_ = DWINIT::CUROBJ->get_jump_read_endl();
  jump_read_ = DWINIT::CUROBJ->get_jump_read();
  stride_offset_in_ = DWINIT::CUROBJ->get_stride_offset_in();
  shift_bias_ = DWINIT::CUROBJ->get_shift_bias();

  jump_write_endl_ = DWINIT::CUROBJ->get_jump_write_endl();
  jump_write_ = DWINIT::CUROBJ->get_jump_write();
  stride_offset_out_ = DWINIT::CUROBJ->get_stride_offset_out();
  shift_cut_ = DWINIT::CUROBJ->get_shift_cut();

  // constraints from hardware implementation
  if (shift_bias_ < 32) {
    UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED)
        << "invalid shift_bias";
  }

  // self defined
  vpp_ = valid_pixel_parallel_;
  pp_ = ArchCfg::Instance().get_param().dwconv_engine().pixel_parallel();
  assert(vpp_ <= pp_);
  pp_conv_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  assert(pp_ <= pp_conv_);
  cp_ = ArchCfg::Instance().get_param().dwconv_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel

  dst_h_ = pp_;
  real_dst_h_ = vpp_;
  dst_w_ = align_upper(length_, 8);
  real_dst_w_ = length_;

  src_h_ = (dst_h_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (real_dst_h_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (dst_w_ - 1) * stride_w_ + kernel_w_;
  real_src_w_ = (real_dst_w_ - 1) * stride_w_ + kernel_w_;
  src_size_ = src_h_ * src_w_ * ic_;

  group_id_in_ = bank_id_in_ / pp_conv_;
  group_id_out_ = bank_id_out_ / pp_conv_;

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_);
  weights_.resize(oc_ * kernel_h_ * kernel_w_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
}

template <>
DptWise<DPUVersion::DPU4F>::DptWise(int inst_type, int instid,
                                    vector<string>& inst_str,
                                    vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // dwconv instruction fields
  // debug
  debug_ =
      debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::DPU4F>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPU4F>::inst_table;
  // get dwconv instruction field value
  channel_group_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_GROUP];
  bank_addr_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN];
  pad_left_ = field_val_[TABLE::DPTWISE_FIELD_PAD_LEFT];
  pad_right_ = field_val_[TABLE::DPTWISE_FIELD_PAD_RIGHT];
  pad_top_ = field_val_[TABLE::DPTWISE_FIELD_PAD_TOP];
  pad_bottom_ = field_val_[TABLE::DPTWISE_FIELD_PAD_BOTTOM];
  channel_offset_ = field_val_[TABLE::DPTWISE_FIELD_CHANNEL_OFFSET];
  bank_id_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_IN];
  bank_id_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_OUT];
  bank_addr_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_OUT];
  bank_addr_weights_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_WEIGHTS];
  length_ = field_val_[TABLE::DPTWISE_FIELD_LENGTH];
  bank_addr_bias_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_BIAS];

  bank_addr_in_1_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_1];
  bank_addr_in_2_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_2];
  bank_addr_in_3_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_3];
  act_type_ = field_val_[TABLE::DPTWISE_FIELD_ACT_TYPE];

  // dwinit value
  using DWINIT = DWInit<DPUVersion::DPU4F>;
  jump_read_ = DWINIT::CUROBJ->get_jump_read();
  jump_write_ = DWINIT::CUROBJ->get_jump_write();
  jump_read_endl_ = DWINIT::CUROBJ->get_jump_read_endl();
  stride_offset_in_ = DWINIT::CUROBJ->get_stride_offset_in();
  valid_pixel_parallel_ = DWINIT::CUROBJ->get_valid_pixel_parallel();
  stride_w_ = DWINIT::CUROBJ->get_stride_w();
  stride_h_ = DWINIT::CUROBJ->get_stride_h();
  kernel_w_ = DWINIT::CUROBJ->get_kernel_w();
  kernel_h_ = DWINIT::CUROBJ->get_kernel_h();
  jump_write_endl_ = DWINIT::CUROBJ->get_jump_write_endl();
  stride_offset_out_ = DWINIT::CUROBJ->get_stride_offset_out();
  stride_out_ = DWINIT::CUROBJ->get_stride_out();
  // shift_bias_ = DWINIT::CUROBJ->get_shift_bias();
  // shift_cut_ = DWINIT::CUROBJ->get_shift_cut();

  // self defined
  vpp_ = valid_pixel_parallel_;
  pp_ = ArchCfg::Instance().get_param().dwconv_engine().pixel_parallel();
  assert(vpp_ <= pp_);
  pp_conv_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  assert(pp_ <= pp_conv_);
  cp_ = ArchCfg::Instance().get_param().dwconv_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel
  src_h_ = (pp_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (vpp_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (length_ - 1) * stride_w_ + kernel_w_;
  real_src_w_ = src_w_;
  dst_h_ = pp_;
  real_dst_h_ = vpp_;
  dst_w_ = length_;
  real_dst_w_ = dst_w_;
  src_size_ = src_h_ * src_w_ * ic_;

  group_id_in_ = bank_id_in_ / pp_conv_;
  group_id_out_ = bank_id_out_ / pp_conv_;

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_);
  weights_.resize(oc_ * kernel_h_ * kernel_w_);
  bias_.resize(oc_);
  shift_bias_buf_.resize(oc_);
  shift_cut_buf_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
  for (auto& shift : shift_bias_buf_)
    UNI_LOG_CHECK(shift <= 20, SIM_PARAMETER_FAILED) << "invalid shift_bias";
}

template <>
DptWise<DPUVersion::XVDPU>::DptWise(int inst_type, int instid,
                                    vector<string>& inst_str,
                                    vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // dwconv instruction fields
  // debug
  debug_ =
      debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_DWCONV);
  Helper<DPUVersion::XVDPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XVDPU>::inst_table;
  // get dwconv instruction field value
  bank_addr_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_OUT];
  bank_id_out_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_OUT];

  bank_addr_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN];
  bank_id_in_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ID_IN];

  pad_left_ = field_val_[TABLE::DPTWISE_FIELD_PAD_LEFT];
  pad_right_ = field_val_[TABLE::DPTWISE_FIELD_PAD_RIGHT];
  pad_top_ = field_val_[TABLE::DPTWISE_FIELD_PAD_TOP];
  pad_bottom_ = field_val_[TABLE::DPTWISE_FIELD_PAD_BOTTOM];

  bank_addr_in_1_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_1];
  bank_addr_weights_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_WEIGHTS];
  valid_pixel_parallel_ = field_val_[TABLE::DPTWISE_FIELD_VALID_PIXEL_PARALLEL];

  bank_addr_in_3_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_3];
  bank_addr_in_2_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_IN_2];
  bank_addr_bias_ = field_val_[TABLE::DPTWISE_FIELD_BANK_ADDR_BIAS];

  // dwinit value
  using DWINIT = DWInit<DPUVersion::XVDPU>;
  channel_group_ = DWINIT::CUROBJ->get_channel_group();
  jump_read_endl_ = DWINIT::CUROBJ->get_jump_read_endl();

  stride_h_ = DWINIT::CUROBJ->get_stride_h();
  stride_w_ = DWINIT::CUROBJ->get_stride_w();
  kernel_h_ = DWINIT::CUROBJ->get_kernel_h();
  kernel_w_ = DWINIT::CUROBJ->get_kernel_w();
  stride_offset_in_ = DWINIT::CUROBJ->get_stride_offset_in();
  stride_offset_out_ = DWINIT::CUROBJ->get_stride_offset_out();
  jump_read_ = DWINIT::CUROBJ->get_jump_read();

  stride_out_ = DWINIT::CUROBJ->get_stride_out();
  act_type_ = DWINIT::CUROBJ->get_act_type();
  channel_offset_ = DWINIT::CUROBJ->get_channel_offset();
  jump_write_endl_ = DWINIT::CUROBJ->get_jump_write_endl();

  shift_cut_ = DWINIT::CUROBJ->get_shift_cut();
  shift_bias_ = DWINIT::CUROBJ->get_shift_bias();
  length_ = DWINIT::CUROBJ->get_length();
  jump_write_ = DWINIT::CUROBJ->get_jump_write();

  // constraints from hardware implementation
  UNI_LOG_CHECK(shift_bias_ <= 20, SIM_PARAMETER_FAILED)
      << "invalid shift_bias";

  // self defined
  vpp_ = valid_pixel_parallel_;
  pp_ = ArchCfg::Instance().get_param().dwconv_engine().pixel_parallel();
  assert(vpp_ <= pp_);
  pp_conv_ = ArchCfg::Instance().get_param().conv_engine().pixel_parallel();
  assert(pp_ <= pp_conv_);
  cp_ = ArchCfg::Instance().get_param().dwconv_engine().channel_parallel();
  cg_ = channel_group_;
  co_ = channel_offset_;
  ic_ = cg_ * cp_;  // input_channel is channel_group multiply channel_parallel
  oc_ = ic_;        // output_channel is channel_parallel
  src_h_ = (pp_ - 1) * stride_h_ + kernel_h_;
  real_src_h_ = (vpp_ - 1) * stride_h_ + kernel_h_;
  src_w_ = (length_ - 1) * stride_w_ + kernel_w_;
  real_src_w_ = src_w_;
  dst_h_ = pp_;
  real_dst_h_ = vpp_;
  dst_w_ = length_;
  real_dst_w_ = dst_w_;
  src_size_ = src_h_ * src_w_ * ic_;

  group_id_in_ = bank_id_in_ / pp_conv_;
  group_id_out_ = bank_id_out_ / pp_conv_;

  // resize related buffer size
  img_.resize(src_h_ * src_w_ * ic_);
  weights_.resize(oc_ * kernel_h_ * kernel_w_);
  bias_.resize(oc_);
  rlt_s64_.resize(dst_h_ * dst_w_ * oc_);
  rlt_dtype_.resize(dst_h_ * dst_w_ * oc_);
}

template <DPUVersion T>
DptWise<T>::~DptWise() {}

// public funcs
template <DPUVersion T>
void DptWise<T>::Exec() {
  target_check();
  // read img, weights, bias data from bank
  read();
  debug_input_fmap("read");

  // padding for img data, also handle channel offset
  pad();
  debug_input_fmap("pad");

  debug_read_pos_tick();
  debug_input_tick();

  // do conv operation of the whole feature map
  dw_conv();
  debug_output_fmap(0, "dwconv_rlt_fmap");

  // add bias to conv result
  bias();
  debug_output_fmap(0, "bias_rlt_fmap");

  // do shift, trunc operation
  transform();
  debug_output_fmap(1, "transform_rlt_fmap");
  debug_output_tick();

  post_transform();
  debug_output_fmap(1, "post_transform_rlt_fmap");

  // save result to bank
  save();
}

// private funcs
template <DPUVersion T>
void DptWise<T>::read() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "read start..." << endl;

  // read img data from bank
  // UNI_LOG_INFO << "reading img..." << endl;

  UNI_LOG_CHECK(stride_offset_in_ < stride_h_, SIM_PARAMETER_FAILED)
      << "invalid stride_offset_in";
  auto stride_offset = stride_offset_in_;
  auto img_bank_id = bank_id_in_;
  auto base_bank_id = (bank_id_in_ / pp_conv_) * pp_conv_;
  auto addr_group = 0;
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_)->GetH();
  for (auto idx_src_h = 0; idx_src_h < real_src_h_; idx_src_h++) {
    for (auto idx_src_w = 0; idx_src_w < src_w_; idx_src_w++) {
      auto ddr_addr = (idx_src_h * src_w_ + idx_src_w) * ic_;
      auto bank_addr_group_base = get_group_base_addr(addr_group);
      auto bank_addr = (bank_addr_group_base + stride_offset * jump_read_endl_ +
                        idx_src_w * jump_read_) %
                       bank_depth;
      auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(img_bank_id);
      bank->Read(bank_addr, ic_, img_.data() + ddr_addr);
    }
    if (++stride_offset == stride_h_) {
      stride_offset = 0;
      if (++img_bank_id == base_bank_id + pp_conv_) {
        img_bank_id = base_bank_id;
        ++addr_group;
      }
    }
  }

  // read weights from bank
  // UNI_LOG_INFO << "reading weights..." << endl;
  auto wgt_bank_group =
      ArchCfg::Instance().get_param().dwconv_engine().weight_bank();
  auto bank_width_wgt =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0)->GetW();
  auto bank_depth_wgt =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0)->GetH();
  for (int i = 0; i < 1; i++) {
    int bank_id = i % cp_;
    for (int j = 0; j < kernel_h_; j++) {
      for (int k = 0; k < kernel_w_; k++) {
        int ddr_addr =
            i * kernel_h_ * kernel_w_ * ic_ + j * kernel_w_ * ic_ + k * ic_;
        int bank_addr =
            (bank_addr_weights_ + j * kernel_w_ * cg_ * (cp_ / bank_width_wgt) +
             k * cg_ * (cp_ / bank_width_wgt)) %
            bank_depth_wgt;
        auto bank =
            Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, bank_id);
        bank->Read(bank_addr, ic_, weights_.data() + ddr_addr);
      }
    }
  }

  // read bias from bank
  // UNI_LOG_INFO << "reading bias..." << endl;
  {
    auto bias_bank_group =
        ArchCfg::Instance().get_param().dwconv_engine().bias_bank();
    auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bias_bank_group, 0);
    bank->Read(bank_addr_bias_, oc_, bias_.data());
  }
  // UNI_LOG_INFO << "read finish!" << endl;
}

// private funcs
template <>
void DptWise<DPUVersion::DPUV4E>::read() {
  auto bank_id_group_base = ArchCfg::Instance().get_base_bank_id(bank_id_in_);
  for (auto idx_src_h = 0; idx_src_h < src_h_; idx_src_h++) {
    auto bank_id_offset = bank_id_in_ - bank_id_group_base +
                          div_floor(stride_offset_in_ + idx_src_h, stride_h_);
    auto bank_id = bank_id_group_base + bank_id_offset % pp_conv_;
    auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id);
    auto bank_addr_base =
        get_group_base_addr(div_floor(bank_id_offset, pp_conv_)) +
        ((stride_offset_in_ + idx_src_h) % stride_h_) * jump_read_endl_;
    for (auto idx_src_w = 0; idx_src_w < src_w_; idx_src_w++) {
      auto offset = (idx_src_h * src_w_ + idx_src_w) * ic_;
      auto bank_addr = bank_addr_base + idx_src_w * jump_read_;
      bank->Read(bank_addr, ic_, img_.data() + offset);
    }
  }
  // read wgt
  auto weight_bank =
      ArchCfg::Instance().get_param().dwconv_engine().weight_bank();
  auto shuffled_weights_size = kernel_h_ * kernel_w_ * ic_;
  auto shuffled_weights = std::vector<DPU_DATA_TYPE>(shuffled_weights_size);
  Buffer<DPU_DATA_TYPE>::Instance()
      .GetBank(weight_bank, 0)
      ->Read(bank_addr_weights_, shuffled_weights_size,
             shuffled_weights.data());

  // reverse shuffling
  /* the shuffling will only occurred in channel direction
   * for each pixel, the channel will be aligned to the channel parallel, which
   * in v4e now is 64, and one shuffle unit is 8 char.
   * clang-format off
   * big endian
   * ---------------------------------------------------------
   * | Wgt0: 48-55 | Wgt0: 32-39 | Wgt0: 16-23 | Wgt0: 00-07 |
   * ---------------------------------------------------------
   * | Wgt0: 56-64 | Wgt0: 40-47 | Wgt0: 24-31 | Wgt0: 08-15 |
   * ---------------------------------------------------------
   * | Wgt1: 48-55 | Wgt1: 32-39 | Wgt1: 16-23 | Wgt1: 00-07 |
   * ---------------------------------------------------------
   * | Wgt1: 56-64 | Wgt1: 40-47 | Wgt1: 24-31 | Wgt1: 08-15 |
   * ---------------------------------------------------------
   * clang-format on
   */
  auto offset_dst = 0;
  for (auto idx_kh = 0; idx_kh < kernel_h_; idx_kh++) {
    for (auto idx_kw = 0; idx_kw < kernel_w_; idx_kw++) {
      for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
        // base idx for current channel group
        auto base_id_src =
            idx_kh * kernel_w_ * ic_ + idx_kw * ic_ + idx_cg * cp_;
        for (auto idx_unit = 0; idx_unit < (cp_ / 8); idx_unit++) {
          auto offset_src =
              base_id_src + (idx_unit % 2) * cp_ / 2 + (idx_unit / 2) * 8;
          for (auto idx = 0; idx < 8; idx++) {
            weights_[offset_dst++] = shuffled_weights[offset_src++];
          }
        }
      }  // chanel group
    }    // kernel w
  }      // kernel h
  // read bias
  auto bias_bank = ArchCfg::Instance().get_param().dwconv_engine().bias_bank();
  auto shuffled_bias = std::vector<DPU_DATA_TYPE>(oc_);
  Buffer<DPU_DATA_TYPE>::Instance()
      .GetBank(bias_bank, 0)
      ->Read(bank_addr_bias_, oc_, shuffled_bias.data());
  // reverse shuffling
  /* shuffling order
     loop range(channel_group)       // cg
       map 2*32B                     // cp=64
        -----------------------------------------------
       | cp[55:48] | cp[39:32] | cp[23:16] | cp[ 7: 0] |
        -----------------------------------------------
       | cp[63:56] | cp[47:40] | cp[31:24] | cp[15: 8] |
        -----------------------------------------------
   */
  offset_dst = 0;
  for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
    for (auto idx_cp = 0; idx_cp < cp_; idx_cp++) {
      // offset, first line or second line
      auto offset_src = idx_cg * cp_ + (div_floor(idx_cp, 8) % 2) * 32;
      // offset inside one line
      offset_src = offset_src + div_floor(idx_cp, 16) * 8 + idx_cp % 8;
      bias_[offset_dst++] = shuffled_bias[offset_src];
    }
  }
}

template <>
void DptWise<DPUVersion::DPU4F>::read() {
  UNI_LOG_CHECK(stride_offset_in_ < stride_h_, SIM_PARAMETER_FAILED)
      << "invalid stride_offset_in";
  auto stride_offset = stride_offset_in_;
  auto img_bank_id = bank_id_in_;
  auto base_bank_id = (bank_id_in_ / pp_conv_) * pp_conv_;
  auto addr_group = 0;
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_)->GetH();
  for (auto idx_src_h = 0; idx_src_h < real_src_h_; idx_src_h++) {
    for (auto idx_src_w = 0; idx_src_w < src_w_; idx_src_w++) {
      auto ddr_addr = (idx_src_h * src_w_ + idx_src_w) * ic_;
      auto bank_addr_group_base = get_group_base_addr(addr_group);
      auto bank_addr = (bank_addr_group_base + stride_offset * jump_read_endl_ +
                        idx_src_w * jump_read_) %
                       bank_depth;
      auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(img_bank_id);
      BankShell::read(true, img_bank_id, ic_, bank_addr,
                      img_.data() + ddr_addr);
    }
    if (++stride_offset == stride_h_) {
      stride_offset = 0;
      if (++img_bank_id == base_bank_id + pp_conv_) {
        img_bank_id = base_bank_id;
        ++addr_group;
      }
    }
  }

  // read weights from bank
  auto wgt_bank_group =
      ArchCfg::Instance().get_param().dwconv_engine().weight_bank();
  auto bank_depth_wgt =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0)->GetH();
  for (int i = 0; i < 1; i++) {
    int bank_id = i % cp_;
    for (int j = 0; j < kernel_h_; j++) {
      for (int k = 0; k < kernel_w_; k++) {
        int ddr_addr =
            i * kernel_h_ * kernel_w_ * ic_ + j * kernel_w_ * ic_ + k * ic_;
        int bank_addr = (bank_addr_weights_ + j * kernel_w_ * cg_ + k * cg_) %
                        bank_depth_wgt;
        auto bank =
            Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, bank_id);
        bank->Read(bank_addr, ic_, weights_.data() + ddr_addr);
      }
    }
  }

  // read bias from bank
  {
    auto bias_bank_group =
        ArchCfg::Instance().get_param().dwconv_engine().bias_bank();
    auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bias_bank_group, 0);
    auto addr = bank_addr_bias_;
    for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
      bank->Read(addr++, cp_, bias_.data() + idx_cg * cp_);
      bank->Read(addr++, cp_, shift_bias_buf_.data() + idx_cg * cp_);
      bank->Read(addr++, cp_, shift_cut_buf_.data() + idx_cg * cp_);
    }
  }
}

template <DPUVersion dv>
int DptWise<dv>::get_group_base_addr(int addr_group) {
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

template <>
int DptWise<DPUVersion::DPUV3ME>::get_group_base_addr(int addr_group) {
  return bank_addr_in_ + addr_group * stride_h_ * jump_read_endl_;
}

template <DPUVersion dv>
void DptWise<dv>::pad() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "pad start..." << endl;
  // padding left
  // UNI_LOG_INFO << "padding left..." << endl;
  for (int i = 0; i < src_h_; i++) {
    for (int j = 0; j < pad_left_ && j < src_w_; j++) {
      for (int k = 0; k < ic_; k++) {
        int addr = i * src_w_ * ic_ + j * ic_ + k;
        img_[addr] = 0;
      }
    }
  }

  // padding top
  // UNI_LOG_INFO << "padding top..." << endl;
  for (int i = 0; i < pad_top_ && i < src_h_; i++) {
    for (int j = 0; j < src_w_; j++) {
      for (int k = 0; k < ic_; k++) {
        int addr = i * src_w_ * ic_ + j * ic_ + k;
        img_[addr] = 0;
      }
    }
  }

  // padding right
  // UNI_LOG_INFO << "padding right..." << endl;
  for (int i = 0; i < src_h_; i++) {
    for (int j = real_src_w_ - kernel_w_ + pad_right_ + 1; j < src_w_; j++) {
      for (int k = 0; k < ic_; k++) {
        int addr = i * src_w_ * ic_ + j * ic_ + k;
        img_[addr] = 0;
      }
    }
  }

  // padding bottom, it's different from padding right
  // UNI_LOG_INFO << "padding bottom..." << endl;
  for (int i = real_src_h_ - kernel_h_ + pad_bottom_ + 1; i < src_h_; i++) {
    for (int j = 0; j < src_w_; j++) {
      for (int k = 0; k < ic_; k++) {
        int addr = i * src_w_ * ic_ + j * ic_ + k;
        img_[addr] = 0;
      }
    }
  }
  // UNI_LOG_INFO << "pad finish!" << endl;

  // handle channel offset
  for (int i = 0; i < src_h_; i++) {
    for (int j = 0; j < src_w_; j++) {
      int ddr_addr = i * src_w_ * ic_ + j * ic_ + ic_ - co_;
      for (int k = 0; k < co_; k++) {
        img_[ddr_addr + k] = 0;
      }
    }
  }
}

template <DPUVersion dv>
void DptWise<dv>::dw_conv() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "dwconv start..." << endl;
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        dw_conv_one(i * stride_h_, j * stride_w_, k);
      }
    }
  }

  // UNI_LOG_INFO << "dwconv finish!" << endl;
}

template <DPUVersion dv>
void DptWise<dv>::dw_conv_one(int idx_src_h, int idx_src_w, int idx_oc) {
  int64_t result = 0;

  // calculate dwconv value
  for (int i = 0; i < kernel_h_; i++) {
    for (int j = 0; j < kernel_w_; j++) {
      int img_addr =
          (idx_src_h + i) * src_w_ * ic_ + (idx_src_w + j) * ic_ + idx_oc;
      int weights_addr = i * kernel_w_ * ic_ + j * ic_ + idx_oc;
      result += img_[img_addr] * weights_[weights_addr];
    }
  }

  // assign dwconv value to rlt_s64_ var
  int idx_dst_h = idx_src_h / stride_h_;
  int idx_dst_w = idx_src_w / stride_w_;
  int rlt_addr = idx_dst_h * dst_w_ * oc_ + idx_dst_w * oc_ + idx_oc;
  rlt_s64_[rlt_addr] = result;
}

template <DPUVersion dv>
void DptWise<dv>::bias() {
  auto shift_bias = (shift_bias_ >= 32) ? (32 - shift_bias_) : shift_bias_;
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int rlt_addr = i * dst_w_ * oc_ + j * oc_ + k;
        rlt_s64_[rlt_addr] *= 2.0;
        rlt_s64_[rlt_addr] +=
            floor((double)bias_[k] * 2.0 * pow(2, shift_bias));
      }
    }
  }
}

template <>
void DptWise<DPUVersion::XVDPU>::bias() {
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int rlt_addr = i * dst_w_ * oc_ + j * oc_ + k;
        rlt_s64_[rlt_addr] *= 2.0;
        rlt_s64_[rlt_addr] += (double)bias_[k] * pow(2, shift_bias_);
      }
    }
  }
}

template <>
void DptWise<DPUVersion::DPUV2>::bias() {
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int rlt_addr = i * dst_w_ * oc_ + j * oc_ + k;
        rlt_s64_[rlt_addr] *= 2.0;
        rlt_s64_[rlt_addr] += (double)bias_[k] * pow(2, shift_bias_);
      }
    }
  }
}

template <>
void DptWise<DPUVersion::DPU4F>::bias() {
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int rlt_addr = i * dst_w_ * oc_ + j * oc_ + k;
        rlt_s64_[rlt_addr] *= 2.0;
        rlt_s64_[rlt_addr] += (bias_[k] << shift_bias_buf_[k]);
      }
    }
  }
}

template <DPUVersion T>
void DptWise<T>::target_check() {
  // bank access
  /// dwconv in
  auto access_white_list =
      ArchCfg::Instance().get_bank_access_white_list("dwconv-in");
  UNI_LOG_CHECK((access_white_list.count(bank_id_in_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_in (= " << bank_id_in_ << ") does not match target!" << endl;
  access_white_list.clear();
  /// dwconv out
  access_white_list =
      ArchCfg::Instance().get_bank_access_white_list("dwconv-out");
  UNI_LOG_CHECK((access_white_list.count(bank_id_out_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id_out (= " << bank_id_out_ << ") does not match target!"
      << endl;
  // dwconv_limit
  /// kernel size
  auto limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("dwconv-kernel");
  UNI_LOG_CHECK((limit_white_list.count(kernel_h_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_h (= " << kernel_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(kernel_w_) > 0), SIM_OUT_OF_RANGE)
      << "kernel_w (= " << kernel_w_ << ") does not match target!" << endl;
  limit_white_list.clear();
  /// stride
  limit_white_list =
      ArchCfg::Instance().get_instr_limit_white_list("dwconv-stride");
  UNI_LOG_CHECK((limit_white_list.count(stride_h_) > 0), SIM_OUT_OF_RANGE)
      << "stride_h (= " << stride_h_ << ") does not match target!" << endl;
  UNI_LOG_CHECK((limit_white_list.count(stride_w_) > 0), SIM_OUT_OF_RANGE)
      << "stride_w (= " << stride_w_ << ") does not match target!" << endl;
  // nonlinear
  auto support_list = ArchCfg::Instance()
                          .get_param()
                          .dwconv_engine()
                          .nonlinear()
                          .nonlinear_type();

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

template <DPUVersion dv>
void DptWise<dv>::transform() {
  // UNI_LOG_INFO << endl;
  // UNI_LOG_INFO << "transform start..." << endl;
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int addr = i * dst_w_ * oc_ + j * oc_ + k;

        double tmp = rlt_s64_[addr];
        tmp /= pow(2, (shift_cut_ + 1));

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
  // UNI_LOG_INFO << "transform finish!" << endl;
}

template <DPUVersion dv>
void DptWise<dv>::post_transform() {}

template <>
void DptWise<DPUVersion::DPUV4E>::transform() {
  UNI_LOG_CHECK(act_type_ == Calc::RELU_TYPE_NONE ||
                    act_type_ == Calc::RELU_TYPE_RELU ||
                    act_type_ == Calc::RELU_TYPE_LEAKY_RELU ||
                    act_type_ == Calc::RELU_TYPE_RELUSIX,
                SIM_PARAMETER_FAILED)
      << "Not support nonlinear type " << act_type_ << endl;
  std::transform(rlt_s64_.begin(), rlt_s64_.end(), rlt_dtype_.begin(),
                 [this](const int64_t& rlt) {
                   auto tmp = static_cast<double>(rlt) /
                              std::pow(2., (shift_cut_ + 1));
                   if (act_type_ == Calc::RELU_TYPE_RELU) {
                     tmp = tmp < 0 ? 0 : tmp;
                   } else if (act_type_ == Calc::RELU_TYPE_LEAKY_RELU) {
                     tmp = tmp < 0 ? tmp * 26. / 256. : tmp;
                   }
                   return Calc::DPURound<DPU_DATA_TYPE>(tmp);
                 });
}

template <>
void DptWise<DPUVersion::DPUV4E>::post_transform() {
  const DPU_DATA_TYPE thr6 = 6 << 4;
  if (act_type_ == Calc::RELU_TYPE_RELUSIX) {
    std::transform(rlt_dtype_.begin(), rlt_dtype_.end(), rlt_dtype_.begin(),
                   [&thr6](const DPU_DATA_TYPE& rlt) {
                     auto tmp = rlt < 0 ? 0 : rlt;
                     return tmp < thr6 ? tmp : thr6;
                   });
  }
}

template <>
void DptWise<DPUVersion::DPU4F>::transform() {
  for (int i = 0; i < dst_h_; i++) {
    for (int j = 0; j < dst_w_; j++) {
      for (int k = 0; k < oc_; k++) {
        int addr = i * dst_w_ * oc_ + j * oc_ + k;

        double tmp = rlt_s64_[addr];
        tmp /= pow(2, (shift_cut_buf_[k] + 1));

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
}

template <DPUVersion T>
void DptWise<T>::save() {
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_)->GetH();
  auto bank_id_out_offset = bank_id_out_ % pp_conv_;
  auto bank_id_out_base = bank_id_out_ - bank_id_out_ % pp_conv_;
  for (auto idx_h = 0; idx_h < real_dst_h_; idx_h++) {
    // bank change, when (idx_h + stride_offset_out) > stride_out
    auto bank_offset = static_cast<uint32_t>(
        floor((float)(idx_h + stride_offset_out_) / stride_out_));
    // bank group inner idx increase
    auto bank_group_offset = static_cast<uint32_t>(
        floor((float)(bank_id_out_offset + bank_offset) / pp_conv_));
    auto bank_id_this_row =
        bank_id_out_base + (bank_id_out_offset + bank_offset) % pp_conv_;
    auto bank_addr_this_row =
        bank_addr_out_                                        // base_addr
        + bank_group_offset * stride_out_ * jump_write_endl_  // change bank
        + ((idx_h + stride_offset_out_) % stride_out_) *
              jump_write_endl_  // change stride offset
        ;
    auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_this_row);
    for (int idx_w = 0; idx_w < real_dst_w_; idx_w++) {
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
void DptWise<DPUVersion::DPUV3ME>::save() {
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_)->GetH();

  auto bank_id_out_offset = bank_id_out_ % pp_conv_;
  auto bank_id_out_base = bank_id_out_ - bank_id_out_ % pp_conv_;
  auto line_size = cp_;
  for (auto idx_c = 0; idx_c < cg_; idx_c++) {
    if ((idx_c == cg_ - 1) && (co_ >= cp_ / 2)) line_size = cp_ / 2;

    for (auto idx_h = 0; idx_h < real_dst_h_; idx_h++) {
      // bank id
      auto bank_shift = (idx_h + stride_offset_out_) / stride_out_;
      auto stride_offset = (idx_h + stride_offset_out_) % stride_out_;
      auto addr_group = (bank_id_out_offset + bank_shift) / pp_conv_;
      auto bank_id =
          bank_id_out_base + (bank_id_out_offset + bank_shift) % pp_conv_;
      auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id);

      // bank address
      auto bank_addr_base = bank_addr_out_ +
                            addr_group * stride_out_ * jump_write_endl_ +
                            stride_offset * jump_write_endl_;

      for (int idx_w = 0; idx_w < real_dst_w_; idx_w++) {
        auto bank_addr =
            (bank_addr_base + idx_w * jump_write_ + idx_c * scale_) %
            bank_depth;
        auto rlt_addr = idx_h * dst_w_ * oc_ + idx_w * oc_ + idx_c * cp_;
        bank->Write(bank_addr, line_size, rlt_dtype_.data() + rlt_addr);
      }
    }
  }
}

template <>
void DptWise<DPUVersion::DPU4F>::save() {
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_)->GetH();
  auto bank_id_out_offset = bank_id_out_ % pp_conv_;
  auto bank_id_out_base = bank_id_out_ - bank_id_out_ % pp_conv_;
  for (auto idx_h = 0; idx_h < real_dst_h_; idx_h++) {
    // bank change, when (idx_h + stride_offset_out) > stride_out
    auto bank_offset = static_cast<uint32_t>(
        floor((float)(idx_h + stride_offset_out_) / stride_out_));
    // bank group inner idx increase
    auto bank_group_offset = static_cast<uint32_t>(
        floor((float)(bank_id_out_offset + bank_offset) / pp_conv_));
    auto bank_id_this_row =
        bank_id_out_base + (bank_id_out_offset + bank_offset) % pp_conv_;
    auto bank_addr_this_row =
        bank_addr_out_                                        // base_addr
        + bank_group_offset * stride_out_ * jump_write_endl_  // change bank
        + ((idx_h + stride_offset_out_) % stride_out_) *
              jump_write_endl_  // change stride offset
        ;
    auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_this_row);
    for (int idx_w = 0; idx_w < real_dst_w_; idx_w++) {
      auto bank_addr_this_pixel =
          (bank_addr_this_row + idx_w * jump_write_) % bank_depth;
      auto rlt_addr_this_pixel = idx_h * dst_w_ * oc_ + idx_w * oc_;
      BankShell::write(true, bank_id_this_row, oc_, bank_addr_this_pixel,
                       rlt_dtype_.data() + rlt_addr_this_pixel);
    }
  }
}

template <DPUVersion dv>
void DptWise<dv>::debug_input_fmap(std::string prefix) {
  if (!InstBase::debug_) return;

  // save img data
  auto img = std::vector<DPU_DATA_TYPE>(ic_ * src_h_ * src_w_);
  Calc::HWC2CHW(src_h_, src_w_, ic_, img_.data(), img.data());
  auto ofi = std::ofstream(InstBase::debug_path_ + "img_" + prefix + "_fmap." +
                               std::to_string(InstBase::instid_),
                           std::ios::trunc);
  for (int i = 0; i < ic_; i++) {
    ofi << "channel_" << i << ":" << std::endl;
    Util::SaveDec(ofi, img.data() + i * src_h_ * src_w_, src_h_ * src_w_,
                  src_w_);
  }

  // save weights data
  auto weights = std::vector<DPU_DATA_TYPE>(ic_ * kernel_h_ * kernel_w_);
  Calc::HWC2CHW(kernel_h_, kernel_w_, ic_, weights_.data(), weights.data());
  auto ofw = std::ofstream(InstBase::debug_path_ + "weights_" + prefix +
                               "_fmap." + std::to_string(InstBase::instid_),
                           std::ios::trunc);
  for (int i = 0; i < ic_; i++) {
    ofw << "input_channel_" << i << ":" << std::endl;
    Util::SaveDec(ofw, weights.data() + i * kernel_h_ * kernel_w_,
                  kernel_h_ * kernel_w_, kernel_w_);
  }

  // save bias data
  auto ofb = std::ofstream(InstBase::debug_path_ + "bias_" + prefix + "_fmap." +
                               std::to_string(InstBase::instid_),
                           std::ios::trunc);
  for (int i = 0; i < oc_; i++) {
    ofb << "output_channel_" << i << ":" << std::endl;
    Util::SaveDec(ofb, bias_.data() + i, 1, 1);
  }
}

template <>
void DptWise<DPUVersion::DPU4F>::debug_input_fmap(string prefix) {
  if (!debug_) return;

  string head;
  string save_name_img;
  string save_name_weights;
  string save_name_bias;
  string save_name_shift_bias;
  string save_name_shift_cut;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "img_" << prefix << "_fmap." << instid_;
  ss >> save_name_img;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "weights_" << prefix << "_fmap." << instid_;
  ss >> save_name_weights;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "bias_" << prefix << "_fmap." << instid_;
  ss >> save_name_bias;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "shift_bias_" << prefix << "_fmap." << instid_;
  ss >> save_name_shift_bias;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "shift_cut_" << prefix << "_fmap." << instid_;
  ss >> save_name_shift_cut;

  // save img data
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

  // save weights data
  vector<DPU_DATA_TYPE> weights(ic_ * kernel_h_ * kernel_w_);
  Calc::HWC2CHW(kernel_h_, kernel_w_, ic_, weights_.data(), weights.data());
  for (int j = 0; j < ic_; j++) {
    // save input_channel title to file
    ss.clear();
    ss.str("");
    ss << "input_channel_" << j << ":";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_weights, head.data(), head.size(), SM_APPEND);
    for (int k = 0; k < kernel_h_; k++) {
      int addr = j * kernel_h_ * kernel_w_ + k * kernel_w_;
      Util::SaveDec(save_name_weights, weights.data() + addr, kernel_w_,
                    kernel_w_, SM_APPEND);
    }
  }

  // save bias data
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

  // save shift_bias data
  for (int i = 0; i < oc_; i++) {
    // save output_channel title to file
    ss.clear();
    ss.str("");
    ss << "output_channel_" << i << "-------------------------------------";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_shift_bias, head.data(), head.size(), SM_APPEND);
    Util::SaveDec(save_name_shift_bias, shift_bias_buf_.data() + i, 1, 1,
                  SM_APPEND);
  }

  // save shift_cut data
  for (int i = 0; i < oc_; i++) {
    // save output_channel title to file
    ss.clear();
    ss.str("");
    ss << "output_channel_" << i << "-------------------------------------";
    ss >> head;
    head += "\n";
    Util::SaveBin(save_name_shift_cut, head.data(), head.size(), SM_APPEND);
    Util::SaveDec(save_name_shift_cut, shift_cut_buf_.data() + i, 1, 1,
                  SM_APPEND);
  }
}

template <DPUVersion T>
void DptWise<T>::debug_read_pos_tick() {
  if (!debug_) {
    return;
  }
  string save_name;
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "dwconv_read_pos_tick." << instid_;
  ss >> save_name;
  ofstream fout(save_name);
  UNI_LOG_CHECK(stride_offset_in_ < stride_h_, SIM_PARAMETER_FAILED)
      << "invalid stride_offset_in";
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_in_)->GetH();
  auto bank_id_group_base = (bank_id_in_ / pp_conv_) * pp_conv_;
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
              pp_conv_;
          auto addr_group_pixel_base =
              (bank_id_kernel_base +
               (stride_offset_in_ + idx_kernel_h) / stride_h_) /
              pp_conv_;
          for (auto idx_pixel_parallel = 0; idx_pixel_parallel < pp_;
               idx_pixel_parallel++) {
            auto bank_id = bank_id_group_base +
                           (bank_id_pixel_base + idx_pixel_parallel) % pp_conv_;
            auto addr_group =
                addr_group_pixel_base +
                (bank_id_pixel_base + idx_pixel_parallel) / pp_conv_;
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

template <>
void DptWise<DPUVersion::XVDPU>::debug_input_tick() {
  if (!debug_) return;

  string head;
  string save_name_img;
  string save_name_wgt;
  string save_name_bias;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "read_img_tick." << instid_;
  ss >> save_name_img;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "read_wgt_tick." << instid_;
  ss >> save_name_wgt;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "read_bias_tick." << instid_;
  ss >> save_name_bias;

  auto addr_bias = bank_addr_bias_;
  auto bias_bank_group =
      ArchCfg::Instance().get_param().dwconv_engine().bias_bank();
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

  auto bank_id_group_base = (bank_id_in_ / pp_conv_) * pp_conv_;
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
            auto bank_id_img =
                bank_id_group_base +
                (bank_id_offset_base + bank_id_offset) % pp_conv_;
            auto wrap_back = (bank_id_offset_base + bank_id_offset) / pp_conv_;
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
          // weights data order ohwi: cp_/kernel_h_/kernel_w_/cg_*cp_
          auto wgt_bank_group =
              ArchCfg::Instance().get_param().dwconv_engine().weight_bank();
          auto bank_wgt =
              Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0);
          auto bank_id_weights = bank_wgt->GetID();
          auto bank_depth_wgt = bank_wgt->GetID();
          auto bank_addr_weights_base =
              bank_addr_weights_ + idx_kernel_h * cg_ +
              idx_kernel_w * kernel_h_ * cg_ + idx_channel_group;
          auto bank_addr_weights = bank_addr_weights_base % bank_depth_wgt;

          auto ori_weights_addr = idx_kernel_h * kernel_w_ * cg_ * cp_ +
                                  idx_kernel_w * cg_ * cp_ +
                                  idx_channel_group * cp_;
          Util::SaveHexContSmallEndBankAddr(
              save_name_wgt,
              reinterpret_cast<const char*>(weights_.data() + ori_weights_addr),
              cp_, cp_, bank_depth_wgt, bank_id_weights, bank_addr_weights,
              SM_APPEND);
        }
      }
    }
  }
}

template <DPUVersion dv>
void DptWise<dv>::debug_output_fmap(int type, string save_name) {
  if (!InstBase::debug_) return;

  auto rlt_s64 = std::vector<int64_t>(oc_ * dst_h_ * dst_w_);
  auto rlt_dtype = std::vector<DPU_DATA_TYPE>(oc_ * dst_h_ * dst_w_);
  if (type == 0) {
    Calc::HWC2CHW(dst_h_, dst_w_, oc_, rlt_s64_.data(), rlt_s64.data());
  } else {
    Calc::HWC2CHW(dst_h_, dst_w_, oc_, rlt_dtype_.data(), rlt_dtype.data());
  }
  auto of = std::ofstream(InstBase::debug_path_ + save_name + "." +
                              std::to_string(InstBase::instid_),
                          std::ios::trunc);
  for (int i = 0; i < ic_; i++) {
    of << "channel_" << i << ":" << std::endl;
    if (type == 0) {
      Util::SaveDec(of, rlt_s64.data() + i * dst_h_ * dst_w_, dst_h_ * dst_w_,
                    dst_w_);
    } else {
      Util::SaveDec(of, rlt_dtype.data() + i * dst_h_ * dst_w_, dst_h_ * dst_w_,
                    dst_w_);
    }
  }
}

template <DPUVersion T>
void DptWise<T>::debug_output_tick() {
  if (!debug_) return;

  // make save file name
  std::string save_name =
      debug_path_ + "write_result_tick." + std::to_string(instid_);

  vector<DPU_DATA_TYPE> rlt_tmp(rlt_dtype_.size());

  // change data order from hwc to wch
  for (int i = 0; i < length_; i++) {
    for (int j = 0; j < cp_; j++) {
      for (int k = 0; k < pp_; k++) {
        int addr_tmp = i * cp_ * pp_ + j * pp_ + k;
        int addr_ori = k * length_ * cp_ + i * cp_ + j;

        rlt_tmp[addr_tmp] = rlt_dtype_[addr_ori];
      }
    }
  }

  // save data using upper data order
  /* auto *p = reinterpret_cast<const char *>(rlt_tmp.data());
   int line_size = cp_ * pp_ * sizeof(DPU_DATA_TYPE);
   for (int i = 0; i < length_; i++) {
       int offset = i * line_size;
       Util::SaveHexContBigEnd(save_name, p + offset,
               line_size, line_size, SM_APPEND);
   }*/
  // save img
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_)->GetH();
  auto* p = reinterpret_cast<const char*>(rlt_dtype_.data());
  auto line_size = cp_ * sizeof(DPU_DATA_TYPE);
  // bank id of the first bank in this bank_group
  auto bank_id_out_offset = bank_id_out_ % pp_conv_;
  auto bank_id_out_base = bank_id_out_ - bank_id_out_ % pp_conv_;
  for (int idx_w = 0; idx_w < length_; idx_w++) {
    for (auto idx_h = 0; idx_h < real_dst_h_; idx_h++) {
      auto bank_offset = static_cast<uint32_t>(
          floor((float)(idx_h + stride_offset_out_) / stride_out_));
      auto bank_group_offset = static_cast<uint32_t>(
          floor((float)(bank_id_out_offset + bank_offset) / pp_conv_));
      auto bank_id_this_row =
          bank_id_out_base + (bank_id_out_offset + bank_offset) % pp_conv_;
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

template <DPUVersion T>
void DptWise<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}
template <DPUVersion T>
void DptWise<T>::debug_input_tick() {
  if (!debug_) return;

  string head;
  string save_name_img;
  string save_name_wgt;
  string save_name_bias;
  string save_name_shift_bias;
  string save_name_shift_cut;

  // make save file name
  stringstream ss;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "read_img_tick." << instid_;
  ss >> save_name_img;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "read_wgt_tick." << instid_;
  ss >> save_name_wgt;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "read_bias_tick." << instid_;
  ss >> save_name_bias;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "shift_bias_read_tick." << instid_classify_;
  ss >> save_name_shift_bias;
  ss.clear();
  ss.str("");
  ss << debug_path_ << "shift_cut_read_tick." << instid_classify_;
  ss >> save_name_shift_cut;

  auto addr_bias = bank_addr_bias_;
  auto bias_bank_group =
      ArchCfg::Instance().get_param().dwconv_engine().bias_bank();
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
    // save shift_bias data in smallend mode
    Util::SaveHexContSmallEndBankAddr(
        save_name_shift_bias,
        reinterpret_cast<const char*>(shift_bias_buf_.data() + idx_cg * cp_),
        cp_ * sizeof(DPU_DATA_TYPE), cp_ * sizeof(DPU_DATA_TYPE),
        bank_depth_bias, id_bias, addr_bias++, SM_APPEND);
    // save shift_cut data in smallend mode
    Util::SaveHexContSmallEndBankAddr(
        save_name_shift_cut,
        reinterpret_cast<const char*>(shift_cut_buf_.data() + idx_cg * cp_),
        cp_ * sizeof(DPU_DATA_TYPE), cp_ * sizeof(DPU_DATA_TYPE),
        bank_depth_bias, id_bias, addr_bias++, SM_APPEND);
  }

  auto bank_id_group_base = (bank_id_in_ / pp_conv_) * pp_conv_;
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
            auto bank_id_img =
                bank_id_group_base +
                (bank_id_offset_base + bank_id_offset) % pp_conv_;
            auto wrap_back = (bank_id_offset_base + bank_id_offset) / pp_conv_;
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
          // weights data order ohwi: cp_/kernel_h_/kernel_w_/cg_*cp_
          auto wgt_bank_group =
              ArchCfg::Instance().get_param().dwconv_engine().weight_bank();
          auto bank_wgt =
              Buffer<DPU_DATA_TYPE>::Instance().GetBank(wgt_bank_group, 0);
          auto bank_id_weights = bank_wgt->GetID();
          auto bank_depth_wgt = bank_wgt->GetID();
          auto bank_addr_weights_base =
              bank_addr_weights_ + idx_kernel_h * cg_ +
              idx_kernel_w * kernel_h_ * cg_ + idx_channel_group;
          auto bank_addr_weights = bank_addr_weights_base % bank_depth_wgt;

          auto ori_weights_addr = idx_kernel_h * kernel_w_ * cg_ * cp_ +
                                  idx_kernel_w * cg_ * cp_ +
                                  idx_channel_group * cp_;
          Util::SaveHexContSmallEndBankAddr(
              save_name_wgt,
              reinterpret_cast<const char*>(weights_.data() + ori_weights_addr),
              cp_, cp_, bank_depth_wgt, bank_id_weights, bank_addr_weights,
              SM_APPEND);
        }
      }
    }
  }
}

template <>
void DptWise<DPUVersion::DPUV4E>::debug_input_tick() {
  if (!debug_) return;

  // dwc img tick 1.2
  auto save_name_img = debug_path_ + "read_img_tick." + std::to_string(instid_);

  auto AIE_HP = 2;
  auto AIE_H = 4;
  auto AIE_W = 8;
  auto Hn1 = (AIE_H - 1) * stride_h_ + kernel_h_;
  auto Wn = (((AIE_W - 1) * stride_w_ + kernel_w_ + 1) / 2) * 2;
  auto tick = std::vector<DPU_DATA_TYPE>(Hn1                             // H
                                         * div_ceil(dst_w_, AIE_W) * Wn  // W
                                         * ic_                           // C
                                         * AIE_HP);
  auto offset_src = 0;
  auto offset_dst = 0;

  std::set<std::int32_t> visited{};
  for (auto idx_hg = 0; idx_hg < div_ceil(dst_h_, AIE_H * AIE_HP); idx_hg++) {
    for (auto idx_hgp = 0; idx_hgp < AIE_HP; idx_hgp++) {
      for (auto idx_hb = 0; idx_hb < AIE_H; idx_hb++) {
        auto idx_h = idx_hg * AIE_HP * AIE_H + idx_hgp * AIE_H + idx_hb;
        if (idx_h < dst_h_) {
          for (auto idx_kh = 0; idx_kh < kernel_h_; idx_kh++) {
            auto idx_ih = idx_h * stride_h_ + idx_kh;
            for (auto idx_wg = 0; idx_wg < div_ceil(dst_w_, AIE_W); idx_wg++) {
              for (auto idx_ibw = 0; idx_ibw < Wn; idx_ibw++) {
                auto idx_iw = idx_wg * AIE_W * stride_w_ + idx_ibw;
                if (idx_iw < src_w_) {
                  for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
                    offset_src =
                        idx_ih * src_w_ * ic_ + idx_iw * ic_ + idx_cg * cp_;
                    offset_dst =
                        AIE_HP * (idx_cg * Wn * div_ceil(dst_w_, AIE_W) * Hn1 *
                                      div_ceil(dst_h_, AIE_H * AIE_HP) *
                                      cp_  // cg jump
                                  + idx_hg * Wn * div_ceil(dst_w_, AIE_W) *
                                        Hn1 * cp_            // hg jump
                                  + idx_wg * Wn * Hn1 * cp_  // wg jump
                                  + Wn *
                                        (idx_ih -
                                         (idx_hgp ? (AIE_H * stride_h_) : 0)) *
                                        cp_        // hb jump
                                  + idx_ibw * cp_  // wb jump
                                  ) +
                        idx_hgp * cp_;
                    UNI_LOG_DEBUG_CHECK(static_cast<std::size_t>(
                                            offset_src + cp_) <= img_.size(),
                                        SIM_OUT_OF_RANGE);
                    UNI_LOG_DEBUG_CHECK(static_cast<std::size_t>(
                                            offset_dst + cp_) <= tick.size(),
                                        SIM_OUT_OF_RANGE);
                    // UNI_LOG_DEBUG_CHECK(visited.count(offset_dst) == 0,
                    //                    SIM_OUT_OF_RANGE)
                    //    << "second time to " << offset_dst;
                    std::copy_n(img_.data() + offset_src, cp_,
                                tick.data() + offset_dst);
                  }
                } else {
                  continue;
                }
              }
            }
          }
        } else {
          continue;
        }
      }
    }
  }

  Util::SaveHexContSmallEnd(save_name_img,
                            reinterpret_cast<const char*>(tick.data()),
                            tick.size() * sizeof(DPU_DATA_TYPE),
                            cp_ * 2 * sizeof(DPU_DATA_TYPE), SM_TRUNC);

  /* weights tick order
     package header A
     loop range(channel_group)                                 // cg
       loop range(ceil(output_width/8))                        // wg
         package header B
         shuffled weights for kwp*cp
         refer DptWise<DPUV4E>::read() for more details
     package header A: set_input_window_size(4*32B)
     package header B: start_queue(4*16B) + data header(4*16B)
  */
  auto save_name_wgt = debug_path_ + "read_wgt_tick." + std::to_string(instid_);
  auto shuffled_weights =
      std::vector<DPU_DATA_TYPE>(kernel_h_ * kernel_w_ * ic_);
  Buffer<DPU_DATA_TYPE>::Instance()
      .GetBank(ArchCfg::Instance().get_param().dwconv_engine().weight_bank(), 0)
      ->Read(bank_addr_weights_, shuffled_weights.size(),
             shuffled_weights.data());
  auto block_size = 4 * (6 * 8  // package header B
                         + kernel_h_ * kernel_w_ * 2 * 8);
  auto group_size = block_size;
  auto tick_weights =
      std::vector<DPU_DATA_TYPE>(4 * 8 * 8 +  // package header A
                                 cg_ * div_ceil(dst_w_, AIE_W) * group_size);
  auto header_a = reinterpret_cast<uint64_t*>(tick_weights.data());
  auto PORT_NUM = 4;
  static bool pi_po = 0;  // 0 for pi, 1 for po
  std::uint32_t img_block_w =
      div_ceil((AIE_W - 1) * (stride_w_) + (kernel_w_), 2) * 2;
  std::uint32_t img_block_h = (AIE_H - 1) * (stride_h_) + kernel_h_;
  std::uint32_t img_window_size = img_block_w * img_block_h * 4 - 1;
  std::uint32_t img_ctr = 0x80000000 + img_window_size;
  std::uint32_t wgt_window_size = (kernel_w_) * (kernel_h_)*4 + 3 - 1;
  std::uint32_t wgt_ctr = 0x80000000 + wgt_window_size;
  for (auto idx = 0; idx < PORT_NUM; idx++) {
    // [img-header_pi] [Cpkt-header]
    header_a[idx + 0 * PORT_NUM] = 0x0005d01480000000;
    // [img_ctr_pi]    [Ctr_in0]
    header_a[idx + 1 * PORT_NUM] = static_cast<std::uint64_t>(img_ctr) << 32;
    // [wgt-header-pi] [Cpkt-header]
    header_a[idx + 2 * PORT_NUM] = 0x8005d05480000000;
    // [wgt_ctr_pi]    [Ctr_in0]
    header_a[idx + 3 * PORT_NUM] = static_cast<std::uint64_t>(wgt_ctr) << 32;
    // [img-header-po] [Cpkt-header]
    header_a[idx + 4 * PORT_NUM] = 0x8005d03480000000;
    // [Img_ctr_po]    [Ctr_in0]
    header_a[idx + 5 * PORT_NUM] = static_cast<std::uint64_t>(img_ctr) << 32;
    // [wgt-header-po] [Cpkt-header]
    header_a[idx + 6 * PORT_NUM] = 0x0005d07480000000;
    // [wgt_ctr_po]    [Ctr_in0]
    header_a[idx + 7 * PORT_NUM] = static_cast<std::uint64_t>(wgt_ctr) << 32;
  }
  for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
    for (auto idx_wg = 0; idx_wg < div_ceil(dst_w_, AIE_W); idx_wg++) {
      auto offset_group =
          8 * 32 + (idx_cg * div_ceil(dst_w_, AIE_W) + idx_wg) * group_size;
      auto offset_block = offset_group;
      // generate one block
      auto header_b =
          reinterpret_cast<uint64_t*>(tick_weights.data() + offset_block);
      for (auto idx = 0; idx < PORT_NUM; idx++) {
        // [img-header_sq] [Cpkt-header]
        header_b[idx + 0 * PORT_NUM] = 0x8001de0480000000;
        // [0]             [Start-img-bd]
        header_b[idx + 1 * PORT_NUM] =
            pi_po ? 0x0000000000000001 : 0x0000000000000000;
        // [wgt-header_sq] [Cpkt-header]
        header_b[idx + 2 * PORT_NUM] = 0x0001de0c80000000;
        // [0]             [Start-wgt-bd]
        header_b[idx + 3 * PORT_NUM] =
            pi_po ? 0x0000000000000003 : 0x0000000000000002;
        // [stride_w]      [W-header]
        header_b[idx + 4 * PORT_NUM] =
            (static_cast<uint64_t>(stride_w_) << 32) + 1;
        // [stride_h]      [Ker_w]
        header_b[idx + 5 * PORT_NUM] =
            (static_cast<uint64_t>(stride_h_) << 32) +
            (static_cast<uint64_t>(kernel_h_));
      }
      auto offset_wgt = offset_block + 6 * 32;
      for (auto kernel_size_idx = 0; kernel_size_idx < kernel_h_ * kernel_w_;
           kernel_size_idx++) {
        auto offset_src = kernel_size_idx * ic_ + idx_cg * cp_;
        auto offset_dst = offset_wgt + kernel_size_idx * cp_;
        UNI_LOG_DEBUG_CHECK(static_cast<std::size_t>(offset_src + cp_)  //
                                <= shuffled_weights.size(),
                            SIM_OUT_OF_RANGE);
        UNI_LOG_DEBUG_CHECK(static_cast<std::size_t>(offset_dst + cp_)  //
                                <= tick_weights.size(),
                            SIM_OUT_OF_RANGE);
        std::copy_n(shuffled_weights.begin() + offset_src, cp_,
                    tick_weights.begin() + offset_dst);
      }
      // reverse the pi po
      pi_po = !pi_po;
    }
  }
  Util::SaveHexContSmallEnd(save_name_wgt,
                            reinterpret_cast<const char*>(tick_weights.data()),
                            tick_weights.size() * sizeof(DPU_DATA_TYPE),
                            32 * sizeof(DPU_DATA_TYPE), SM_TRUNC);

  /* bias tick order
     loop range(channel_group)                    // cg
       loop range(ceil(output_width/8))           // wg
           map 10*8*32bit                         // cp=64
            -----------------------------------------------
           |  p1 |  p0 |  p1 |  p0 |  p1 |  p0 |  p1 |  p0 |
            -----------------------------------------------
           |  p3 |  p2 |  p3 |  p2 |  p3 |  p2 |  p3 |  p2 |
            -----------------------------------------------
           | c49 | c48 | c33 | c32 | c17 | c16 | c01 | c00 |
            -----------------------------------------------
           | c51 | c50 | c35 | c34 | c19 | c18 | c03 | c02 |
            -----------------------------------------------
           | c53 | c52 | ... | ... | ... | ... | ... | ... |
            -----------------------------------------------
           | c55 | c54 | ... | ... | ... | ... | ... | ... |
            -----------------------------------------------
           | c57 | c56 | ... | ... | ... | ... | ... | ... |
            -----------------------------------------------
           | c59 | c58 | ... | ... | ... | ... | ... | ... |
            -----------------------------------------------
           | c61 | c60 | ... | ... | ... | ... | ... | ... |
            -----------------------------------------------
           | c63 | c62 | ... | ... | ... | ... | ... | ... |
            -----------------------------------------------
     p0 = kwg * kh
     p1 = shift_cut
     p2 = act_type
     p3 = stride_w
   */
  auto save_name_bias =
      debug_path_ + "read_bias_tick." + std::to_string(instid_);
  auto tick_bias = std::vector<uint32_t>(cg_ * dst_w_ * 10);
  auto param = std::array<uint32_t, 4>{
      static_cast<uint32_t>(div_ceil(kernel_w_, 2) * kernel_h_),  //
      static_cast<uint32_t>(shift_cut_),                          //
      static_cast<uint32_t>(act_type_),                           //
      static_cast<uint32_t>(stride_w_)};                          //
  for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
    auto offset = idx_cg * dst_w_ * 10;
    // control packet
    for (auto idx = 0; idx < 16; idx++) {
      tick_bias[offset + idx] = param[div_floor(idx, 8) * 2 + idx % 2];
    }
    // generate cp=64 pre-shifted bias
    for (auto idx_cp = 0; idx_cp < cp_; idx_cp++) {
      tick_bias[offset + 16 +                    //
                div_floor(idx_cp % 16, 2) * 8 +  //
                div_floor(idx_cp, 16) * 2 +      //
                idx_cp % 2] =                    //
          shift_bias_ >= 32 ? bias_[idx_cg * cp_ + idx_cp] >> (shift_bias_ - 32)
                            : bias_[idx_cg * cp_ + idx_cp] << shift_bias_;
    }
    // repeat for wg times
    for (auto idx_wg = 1; idx_wg < dst_w_ / 8; idx_wg++) {
      std::copy_n(tick_bias.data() + offset, 80,
                  tick_bias.data() + offset + idx_wg * 80);
    }
  }
  Util::SaveHexContSmallEnd(
      save_name_bias, reinterpret_cast<const char*>(tick_bias.data()),
      tick_bias.size() * sizeof(uint32_t), 8 * sizeof(uint32_t), SM_TRUNC);
}

template <>
void DptWise<DPUVersion::DPUV4E>::debug_output_tick() {
  if (!debug_) return;

  auto save_name = debug_path_ + "write_result_tick." + std::to_string(instid_);

  auto AIE_HP = 2;
  auto AIE_H = 4;
  auto AIE_W = 8;
  auto offset_src = 0;
  auto offset_dst = 0;
  auto tick = std::vector<DPU_DATA_TYPE>(
      ic_ * div_ceil(dst_h_, AIE_H * AIE_HP) * AIE_H * AIE_HP *
      div_ceil(dst_w_, AIE_W) * AIE_W);
  for (auto idx_hg = 0; idx_hg < div_ceil(dst_h_, AIE_HP * AIE_H); idx_hg++) {
    for (auto idx_hgp = 0; idx_hgp < AIE_HP; idx_hgp++) {
      for (auto idx_hgb = 0; idx_hgb < AIE_H; idx_hgb++) {
        auto idx_h = idx_hg * AIE_HP * AIE_H + idx_hgp * AIE_H + idx_hgb;
        for (auto idx_wg = 0; idx_wg < div_ceil(dst_w_, AIE_W); idx_wg++) {
          for (auto idx_wb = 0; idx_wb < AIE_W; idx_wb++) {
            for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
              offset_src = idx_h * dst_w_ * oc_ + idx_wg * AIE_W * oc_ +
                           idx_wb * oc_ + idx_cg * cp_;
              offset_dst =
                  AIE_HP *
                      (idx_cg * AIE_W * div_ceil(dst_w_, AIE_W) * AIE_H *
                           div_ceil(dst_h_, AIE_H * AIE_HP) * cp_  // cg jump
                       + idx_wg * AIE_W * AIE_H *
                             div_ceil(dst_h_, AIE_H * AIE_HP) * cp_  // wg jump
                       + idx_wb * AIE_H * cp_                        // wb jump
                       + (idx_h - (idx_hgp ? AIE_H : 0)) * cp_       // hb jump
                       ) +
                  idx_hgp * cp_;
              ;
              UNI_LOG_DEBUG_CHECK(static_cast<std::size_t>(offset_src + cp_)  //
                                      <= rlt_dtype_.size(),
                                  SIM_OUT_OF_RANGE);
              UNI_LOG_DEBUG_CHECK(
                  static_cast<std::size_t>(offset_dst + cp_) <= tick.size(),
                  SIM_OUT_OF_RANGE);
              std::copy_n(rlt_dtype_.data() + offset_src, cp_,
                          tick.data() + offset_dst);
            }
          }
        }
      }
    }
  }
  Util::SaveHexContSmallEnd(save_name,
                            reinterpret_cast<const char*>(tick.data()),
                            tick.size() * sizeof(DPU_DATA_TYPE),
                            cp_ * 2 * sizeof(DPU_DATA_TYPE), SM_TRUNC);
  auto save_name_pos =
      debug_path_ + "write_result_pos_tick." + std::to_string(instid_);
  std::ofstream pos(save_name_pos, std::ios::trunc);
  auto bank_depth =
      Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_out_)->GetH();
  auto bank_id_group_base = ArchCfg::Instance().get_base_bank_id(bank_id_out_);
  for (auto idx_cg = 0; idx_cg < cg_; idx_cg++) {
    for (auto idx_w = 0; idx_w < dst_w_; idx_w++) {
      for (auto idx_h = 0; idx_h < dst_h_; idx_h++) {
        auto bank_id_offset =
            bank_id_out_ - bank_id_group_base +
            div_floor(stride_offset_out_ + idx_h, stride_out_);
        auto bank_id = bank_id_group_base + bank_id_offset % pp_conv_;
        auto bank_addr =
            (bank_addr_out_ +
             div_floor(bank_id_offset, pp_conv_) * stride_out_ *
                 jump_write_endl_ +
             ((stride_offset_out_ + idx_h) % stride_out_) * jump_write_endl_ +
             idx_w * jump_write_ + idx_cg) %
            bank_depth;
        if (idx_w >= real_dst_w_ || idx_h >= real_dst_h_) {
          pos << "xx-xxxx ";

        } else {
          pos << std::setfill('0') << std::setw(2) << bank_id << "-"
              << std::setw(4) << bank_addr << " ";
        }
        if (idx_h % 2 == 1) {
          pos << std::endl;
        }
      }
    }
  }
}
