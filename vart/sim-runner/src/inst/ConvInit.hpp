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
#ifndef __CONVINITL_HPP__
#define __CONVINITL_HPP__

#include "pub/InstBase.hpp"

template <DPUVersion T>
class ConvInit : public InstBase {
 public:
  explicit ConvInit(int inst_type, int instid, vector<string>& inst_str,
                    vector<uint32_t>& inst_val);
  ~ConvInit();

  static ConvInit* CUROBJ;

 public:
  // excute load instruction
  void Exec() override final{};
  void GenInstStr(int inst_fmt) override final;

 public:
  int get_kernel_w() { return kernel_w_; }
  int get_kernel_h() { return kernel_h_; }
  int get_shift_bias() { return shift_bias_; }
  int get_shift_cut() { return shift_cut_; }
  int get_hsigmoid_in() { return hsigmoid_in_; }
  int get_shift_hsigmoid() { return shift_hsigmoid_; }
  int get_shift_hswish() { return shift_hswish_; }
  int get_act_type() { return act_type_; }
  int get_stride_w() { return stride_w_; }
  int get_stride_h() { return stride_h_; }
  int get_stride_out() { return stride_out_; }
  int get_jump_read() { return jump_read_; }
  int get_jump_read_endl() { return jump_read_endl_; }
  int get_jump_write() { return jump_write_; }
  int get_jump_write_endl() { return jump_write_endl_; }
  int get_stride_offset_in() { return stride_offset_in_; }
  int get_stride_offset_out() { return stride_offset_out_; }
  int get_valid_pixel_parallel() { return valid_pixel_parallel_; }

  int get_shift_relusix() { return shift_relusix_; }
  int get_stride_out_cb() { return stride_out_cb_; }
  int get_stride_out_hbm() { return stride_out_hbm_; }
  int get_jump_write_cb() { return jump_write_cb_; }
  int get_jump_write_endl_cb() { return jump_write_endl_cb_; }
  int get_jump_write_hbm() { return jump_write_hbm_; }
  int get_jump_write_endl_hbm() { return jump_write_endl_hbm_; }
  int get_stride_offset_out_cb() { return stride_offset_out_cb_; }
  int get_stride_offset_out_hbm() { return stride_offset_out_hbm_; }

  int get_conv_num() { return conv_num_; }
  int get_tile_icg() { return tile_icg_; }
  int get_ic_iter() { return ic_iter_; }
  int get_icg_offset() { return icg_offset_; }
  int get_channel_offset() { return channel_offset_; }
  int get_channel_group() { return channel_group_; }
  int get_length() { return length_; }
  int get_tile_ocg() { return tile_ocg_; }
  int get_tile_owg() { return tile_owg_; }
  int get_tile_ohg() { return tile_ohg_; }
  int get_ow_iter() { return ow_iter_; }
  int get_ow_offset() { return ow_offset_; }
  int get_jump_read_weights_endl() { return jump_read_weights_endl_; }
  int get_jump_read_weights() { return jump_read_weights_; }

  int get_quant_lth() { return quant_lth_; }
  int get_calt_mode() { return calt_mode_; }

  int get_batch_num() { return batch_num_; }
  int get_stride_batch() { return stride_batch_; }

  int get_tile_en() { return tile_en_; }

  int get_total_in_() { return total_in_; }
  int get_one_width_() { return one_width_; }
  int get_one_height_() { return one_height_; }
  int get_one_size_() { return one_size_; }
  int get_one_cube_() { return one_cube_; }
  int get_total_tile_() { return total_tile_; }
  int get_exec_mode() { return exec_mode_; }
  int get_compression() { return compression_; }
  int get_aie_mode() { return aie_mode_; }

  int get_kw_iter() { return kw_iter_; }
  int get_kh_iter() { return kh_iter_; }

 private:
  int kernel_w_;
  int kernel_h_;
  int shift_bias_;
  int shift_cut_;
  int hsigmoid_in_;
  int shift_hsigmoid_;
  int shift_hswish_;
  int quant_lth_;
  int calt_mode_;
  int act_type_;
  int stride_w_;
  int stride_h_;
  int stride_out_;
  int valid_pixel_parallel_;
  int jump_read_;
  int jump_read_endl_;
  int jump_write_;
  int jump_write_endl_;
  int stride_offset_in_;
  int stride_offset_out_;

  int jump_write_cb_;
  int stride_offset_out_cb_;
  int stride_out_cb_;
  int shift_relusix_;
  int jump_write_endl_cb_;
  int jump_write_hbm_;
  int jump_write_endl_hbm_;
  int stride_out_hbm_;
  int stride_offset_out_hbm_;

  int conv_num_;
  int tile_icg_;
  int ic_iter_;
  int icg_offset_;
  int channel_offset_;
  int channel_group_;
  int length_;
  int tile_ocg_;
  int tile_owg_;
  int tile_ohg_;
  int ow_iter_;
  int ow_offset_;
  int jump_read_weights_endl_;

  int kh_iter_;
  int kw_iter_;

 public:
  void set_jump_read(int jump_read) { jump_read_ = jump_read; }
  void set_jump_read_endl(int jump_read_endl) {
    jump_read_endl_ = jump_read_endl;
  }
  void set_jump_write(int jump_write) { jump_write_ = jump_write; }
  void set_jump_write_endl(int jump_write_endl) {
    jump_write_endl_ = jump_write_endl;
  }
  void set_jump_read_weights_endl(int jump_read_weights_endl) {
    jump_read_weights_endl_ = jump_read_weights_endl;
  }
  void set_jump_read_weights(int jump_read_weights) {
    jump_read_weights_ = jump_read_weights;
  }
  void set_jump_read_elew(int jump_read_elew) {
    jump_read_elew_ = jump_read_elew;
  }
  void set_jump_read_elew_endl(int jump_read_elew_endl) {
    jump_read_elew_endl_ = jump_read_elew_endl;
  }

 private:
  int jump_read_weights_;

  int batch_num_;
  int stride_batch_;

  int tile_en_;

  int total_in_;
  int one_width_;
  int one_height_;
  int one_size_;
  int one_cube_;
  int total_tile_;

  int exec_mode_;
  int compression_;
  int aie_mode_;

  int jump_read_elew_;
  int jump_read_elew_endl_;
  int shift_elew0_;
  int shift_elew1_;
  int elew_cut_;
  int elew_act_type_;
};

template <DPUVersion T>
ConvInit<T>* ConvInit<T>::CUROBJ = nullptr;
template <DPUVersion T>
ConvInit<T>::~ConvInit() {}

#endif /* __CONVINITL_HPP__ */
