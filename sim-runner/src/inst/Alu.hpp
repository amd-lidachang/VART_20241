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
#ifndef __ALU_HPP__
#define __ALU_HPP__

#include <map>
#include "InstBase.hpp"

template <DPUVersion dv>
class Alu : public InstBase {
 public:
  const static DPU_DATA_TYPE MIN_DTYPE_VALUE;

 public:
 public:
  explicit Alu(int inst_type, int instid, vector<string>& inst_str,
               vector<uint32_t>& inst_val);
  ~Alu();

 public:
  // excute load instruction
  void Exec() override final;
  void GenInstStr(int inst_fmt) override final;

 private:
  // get the group base address
  int get_group_base_addr(int grp);
  // read img, weights, bias data from bank
  void read();
  // padding for img data
  void pad();
  // do dwconv operation of the whole feature map
  void dw_conv();
  // do one dwconv kernel's dwconv operation
  void dw_conv_one(int idx_src_h, int idx_src_w, int idx_oc);
  // do elementwise add or mul
  void eltwise() {}
  // add bias to dwconv result
  void bias();
  // do shift, trunc operation
  void transform();
  // save result
  void save();

  // debug funcs for debug
  void gen_debug_instr_ac(string file_name);
  // debug img/weights/bias input value
  void debug_input_fmap_macc_comp(string prefix) {}
  void debug_input_fmap_elew(string prefix) {}
  void debug_input_fmap(string prefix);
  void debug_read_pos_tick();
  void debug_input_tick_macc_comp(string prefix) {}
  void debug_input_tick_elew(string prefix) {}
  void debug_input_tick(string prefix);
  // if type is 1, debug rlt_dtype_
  // if type is 0, debug rlt_s64_
  void debug_output_fmap_elew() {}
  void debug_output_fmap(int type, string save_name);
  void debug_output_tick_elew() {}
  void debug_output_tick(string save_name);

  int translate_shift_cut();
  void maxpool();
  void avepool_one(int idx_src_h, int idx_src_w);
  void maxpool_one(int idx_src_h, int idx_src_w);

  void target_check();

  void alu_genToken_PL_macc_comp() const {}
  void alu_genToken_PL_elew() const {}
  void alu_genToken_PL() const {}
  void alu_gen_ctrlpkg() const {}
  void alu_config_mt_transmit_to_stream() const {}
  void alu_config_layer_parameter() const {}
  void alu_config_DMA_BDs() const {}
  void alu_config_start_DMA() const {}
  void alu_config_mt_receive_from_stream() const {}

 private:
  int32_t bank_addr_out_;
  int32_t bank_id_out_;

  int32_t bank_addr_in_;
  int32_t bank_id_in_;

  int32_t reuse_;

  int32_t bank_addr_in_1_;
  int32_t pad_bottom_;
  int32_t pad_right_;
  int32_t pad_left_;
  int32_t pad_top_;

  int32_t bank_addr_in_2_;
  int32_t bank_addr_weights_;
  int32_t valid_pixel_parallel_;

  int32_t bank_addr_in_3_;
  int32_t bank_addr_bias_;

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
  int32_t kernel_d_;
  int32_t shift_prelu_p_;
  int32_t shift_prelu_n_;

  int32_t num_;
  int32_t share_channel_group_;
  int32_t share_kernel_;
  int32_t shift_write_;
  int32_t shift_write_bit_width_;
  int32_t oh_offset_{0};
  int32_t weights_lines_;

  // hardware config value
  int pp_;
  int pp_alu_;
  int cp_;
  int owp_;

  // self defined
  int vpp_;
  int cg_;
  int co_;
  int ic_;  // input_channel is channel_parallel
  int oc_;  // output_channel is channel_parallel
  int src_h_;
  int real_src_h_;
  int src_w_;
  int dst_h_;
  int real_dst_h_;
  int dst_w_;
  int src_size_;
  int group_id_in_;
  int group_id_out_;
  int shift_p_p;
  int shift_p_n;

  int32_t cg_offset_;
  int32_t macc_cont_;

  int32_t b_mode_{0};
  int32_t pad_type_{};
  int32_t tile_ohg_;
  int32_t tile_cg_;
  int32_t tile_owg_;
  int32_t ow_iter_;
  int32_t ow_offset_;

  int32_t shift_hswish_;
  int32_t shift_hsigmoid_;
  int32_t hsigmoid_in_;

  using h_info_list = std::vector<int32_t>;

  std::map<int32_t, h_info_list> mt_addr_in_;
  std::map<int32_t, h_info_list> aluaddr_jump_read_;
  std::map<int32_t, int32_t> aluaddr_shift_read_;
  h_info_list mt_addr_out_;
  h_info_list aluaddr_jump_write_;

  std::map<int32_t, h_info_list> in_h_num_map_;
  std::vector<int32_t> out_h_num_list_;

  // img, weights, bias
  vector<DPU_DATA_TYPE> img_;        // src_h_ * num_ * src_w_ * ic_
  vector<DPU_DATA_TYPE> weights_;    // oc_ * kernel_h_ * kernel_w_
  vector<int16_t> weights_16bits_;   // oc_ * kernel_h_ * kernel_w_
  vector<DPU_DATA_TYPE> bias_;       // oc_
  vector<DPU_DATA_TYPE> prelu_;      // oc_
  vector<int64_t> rlt_s64_;          // dst_h_ * dst_w_ * oc_
  vector<DPU_DATA_TYPE> rlt_dtype_;  // dst_h_ * dst_w_ * oc_

  int32_t weights_data_size_{1};
};

template <>
Alu<DPUVersion::DPUV2>::Alu(int inst_type, int instid, vector<string>& inst_str,
                            vector<uint32_t>& inst_val);
template <>
void Alu<DPUVersion::DPUV2>::Exec();

template <>
Alu<DPUVersion::XVDPU>::Alu(int inst_type, int instid, vector<string>& inst_str,
                            vector<uint32_t>& inst_val);
template <>
void Alu<DPUVersion::XVDPU>::target_check();
template <>
void Alu<DPUVersion::XVDPU>::Exec();

template <>
void Alu<DPUVersion::XV2DPU>::Exec();
template <>
void Alu<DPUVersion::XV2DPU>::target_check();
template <>
void Alu<DPUVersion::XV2DPU>::read();
template <>
void Alu<DPUVersion::XV2DPU>::pad();
template <>
void Alu<DPUVersion::XV2DPU>::eltwise();
template <>
void Alu<DPUVersion::XV2DPU>::transform();
template <>
void Alu<DPUVersion::XV2DPU>::save();
template <>
void Alu<DPUVersion::XV2DPU>::debug_input_fmap_macc_comp(string prefix);
template <>
void Alu<DPUVersion::XV2DPU>::debug_input_fmap_elew(string prefix);
template <>
void Alu<DPUVersion::XV2DPU>::debug_input_fmap(string prefix);
template <>
void Alu<DPUVersion::XV2DPU>::debug_input_tick_macc_comp(string prefix);
template <>
void Alu<DPUVersion::XV2DPU>::debug_input_tick_elew(string prefix);
template <>
void Alu<DPUVersion::XV2DPU>::debug_input_tick(string prefix);
template <>
void Alu<DPUVersion::XV2DPU>::debug_output_fmap_elew();
template <>
void Alu<DPUVersion::XV2DPU>::debug_output_tick_elew();
template <>
void Alu<DPUVersion::XV2DPU>::debug_output_tick(string save_name);

#endif /* __ALU_HPP__ */
