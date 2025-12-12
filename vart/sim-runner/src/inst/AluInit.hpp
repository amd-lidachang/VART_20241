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
#ifndef __ALUINIT_HPP__
#define __ALUINIT_HPP__

#include "pub/InstBase.hpp"

#define MAX_ELTWISE_OPERAND_NUM 4

template <DPUVersion dv>
class AluInit : public InstBase {
 public:
  explicit AluInit(int inst_type, int instid, vector<string>& inst_str,
                   vector<uint32_t>& inst_val);
  ~AluInit();

  static AluInit* CUROBJ;
  static int32_t left_alu_num;

 public:
  // excute load instruction
  void Exec() override final;
  void GenInstStr(int inst_fmt) override final;

 public:
  int32_t get_jump_read_endl() { return jump_read_endl_; }
  int32_t get_jump_read_weights() { return jump_read_weights_; }
  int32_t get_shift_cut() { return shift_cut_; }

  int32_t get_jump_read() { return jump_read_; }
  int32_t get_shift_bias() { return shift_bias_; }
  int32_t get_act_type() { return act_type_; }
  int32_t get_exec_mode() { return exec_mode_; }
  int32_t get_stride_h() { return stride_h_; }
  int32_t get_stride_w() { return stride_w_; }

  int32_t get_jump_write_endl() { return jump_write_endl_; }
  int32_t get_stride_offset_out() { return stride_offset_out_; }
  int32_t get_stride_offset_in() { return stride_offset_in_; }
  int32_t get_channel_offset() { return channel_offset_; }
  int32_t get_channel_group() { return channel_group_; }

  int32_t get_jump_write() { return jump_write_; }
  int32_t get_length() { return length_; }
  int32_t get_stride_out() { return stride_out_; }
  int32_t get_multi_factor() { return multi_factor_; }

  int32_t get_kernel_h() { return kernel_h_; }
  int32_t get_kernel_w() { return kernel_w_; }
  int32_t get_shift_prelu_p() { return shift_prelu_p_; }
  int32_t get_shift_prelu_n() { return shift_prelu_n_; }

  int32_t get_b_mode() { return b_mode_; }
  int32_t get_num() { return num_; }
  int32_t get_share_channel_group() { return share_channel_group_; }
  int32_t get_share_kernel() { return share_kernel_; }
  int32_t get_alu_num() { return alu_num_; }
  int32_t get_tile_ohg() { return tile_ohg_; }
  int32_t get_tile_cg() { return tile_cg_; }
  int32_t get_tile_owg() { return tile_owg_; }
  int32_t get_pad_type() { return pad_type_; }
  int32_t get_ow_iter() { return ow_iter_; }
  int32_t get_ow_offset() { return ow_offset_; }
  int32_t get_shift_hswish() { return shift_hswish_; }
  int32_t get_shift_hsigmoid() { return shift_hsigmoid_; }
  int32_t get_hsigmoid_in() { return hsigmoid_in_; }
  int32_t get_share_pp(int32_t id) { return share_pp_list_.at(id); }
  int32_t get_shift_read(int32_t id) { return shift_read_list_.at(id); }
  int32_t get_weights_lines() { return weights_lines_; }
  int32_t get_compression_weights() { return compression_weights_; }
  int32_t get_kh_iter() { return kh_iter_; }
  int32_t get_kw_iter() { return kw_iter_; }

  void set_jump_read(int32_t jump_read) { jump_read_ = jump_read; }
  void set_jump_read_endl(int32_t jump_read_endl) {
    jump_read_endl_ = jump_read_endl;
  }
  void set_jump_read_weights(int32_t jump_read_weights) {
    jump_read_weights_ = jump_read_weights;
  }
  void set_jump_write(int32_t jump_write) { jump_write_ = jump_write; }
  void set_jump_write_endl(int32_t jump_write_endl) {
    jump_write_endl_ = jump_write_endl;
  }
  void set_share_pp(unsigned idx, bool value = true) {
    UNI_LOG_CHECK(idx < MAX_ELTWISE_OPERAND_NUM, SIM_PARAMETER_FAILED);
    share_pp_list_[idx] = value;
  }

 private:
  int32_t jump_read_endl_{0};
  int32_t jump_read_weights_{0};
  int32_t shift_cut_{0};

  int32_t jump_read_{0};
  int32_t shift_bias_;
  int32_t act_type_;
  int32_t exec_mode_;
  int32_t stride_h_;
  int32_t stride_w_;

  int32_t jump_write_endl_;
  int32_t stride_offset_out_;
  int32_t stride_offset_in_;
  int32_t channel_offset_;
  int32_t channel_group_;

  int32_t jump_write_;
  int32_t length_;
  int32_t stride_out_;
  int32_t multi_factor_;

  int32_t kernel_h_;
  int32_t kernel_w_;
  int32_t shift_prelu_p_;
  int32_t shift_prelu_n_;

  int32_t b_mode_;
  int32_t num_;
  int32_t share_channel_group_;
  int32_t share_kernel_;
  int32_t alu_num_;

  int32_t tile_ohg_;
  int32_t tile_cg_;
  int32_t tile_owg_;
  int32_t ow_iter_;
  int32_t ow_offset_;

  int32_t pad_type_;

  int32_t shift_hswish_;
  int32_t shift_hsigmoid_;
  int32_t hsigmoid_in_;
  int32_t weights_lines_;

  int32_t compression_weights_;
  int32_t total_tile_;
  int32_t one_height_;
  int32_t one_width_;

  int32_t kh_iter_;
  int32_t kw_iter_;

  int32_t incAO3_;
  int32_t incAO2_;

  std::array<int32_t, MAX_ELTWISE_OPERAND_NUM> share_pp_list_;
  std::array<int32_t, MAX_ELTWISE_OPERAND_NUM> shift_read_list_;
};

template <>
AluInit<DPUVersion::XVDPU>::AluInit(int inst_type, int instid,
                                    vector<string>& inst_str,
                                    vector<uint32_t>& inst_val);
template <>
AluInit<DPUVersion::DPUV2>::AluInit(int inst_type, int instid,
                                    vector<string>& inst_str,
                                    vector<uint32_t>& inst_val);

#endif /* __ALUINIT_HPP__ */
