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

#include "ConvAddr.hpp"

#include "Calc.hpp"
#include "ConvInit.hpp"

template <DPUVersion dv>
int32_t ConvAddr<dv>::num_ = 0;

template <DPUVersion dv>
std::vector<conv_addr_t> ConvAddr<dv>::convaddr_info{};

template <DPUVersion T>
ConvAddr<T>::ConvAddr(int inst_type, int instid, vector<string>& inst_str,
                      vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_CONV);
  Helper<DPUVersion::XV2DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  // get convaddr instruction field value
  using TABLE = TableInterface<DPUVersion::XV2DPU>::inst_table;
  local_type_ = field_val_[TABLE::CONVADDR_FIELD_TYPE];
  local_last_ = field_val_[TABLE::CONVADDR_FIELD_LAST];
  local_compression_ = field_val_[TABLE::CONVADDR_FIELD_COMPRESSION];
  local_invalid_ = field_val_[TABLE::CONVADDR_FIELD_INVALID];
  local_h_num_ = field_val_[TABLE::CONVADDR_FIELD_H_NUM];

  local_mt_addr_ = field_val_[TABLE::CONVADDR_FIELD_MT_ADDR];
  local_jump_ = field_val_[TABLE::CONVADDR_FIELD_JUMP];
  local_jump_endl_ = field_val_[TABLE::CONVADDR_FIELD_JUMP_ENDL];

  ++num_;
  convaddr_info.push_back({local_type_, local_h_num_, local_mt_addr_,
                           local_invalid_, local_jump_, local_jump_endl_});
  auto convaddr_type = (conv_addr_type_t)local_type_;
  auto convinit_ptr = ConvInit<DPUVersion::XV2DPU>::CUROBJ;
  switch (convaddr_type) {
    case conv_addr_type_t::CONV_ADDR_IFM:
      convinit_ptr->set_jump_read(local_jump_);
      convinit_ptr->set_jump_read_endl(local_jump_endl_);
      break;
    case conv_addr_type_t::CONV_ADDR_WGT:
      convinit_ptr->set_jump_read_weights(local_jump_);
      convinit_ptr->set_jump_read_weights_endl(local_jump_endl_);
      break;
    case conv_addr_type_t::CONV_ADDR_BIAS:
      /* bank_bias_mt_addr is at CONV inst */
      break;
    case conv_addr_type_t::CONV_ADDR_OFM:
      convinit_ptr->set_jump_write(local_jump_);
      convinit_ptr->set_jump_write_endl(local_jump_endl_);
      break;
    case conv_addr_type_t::CONV_ADDR_IFM_ELEW:
      convinit_ptr->set_jump_read_elew(local_jump_);
      convinit_ptr->set_jump_read_elew_endl(local_jump_endl_);
      UNI_LOG_CHECK(
          convinit_ptr->get_exec_mode() == Calc::CONV_MODE_CONV_ELEWADD_FUSION,
          SIM_PARAMETER_MISMATCH)
          << "conv only receive elew input fmap when exec_mode is "
             "conv_elewadd_fusion";
      break;
  }

  CUROBJ = this;
}

template <DPUVersion T>
void ConvAddr<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}

template <DPUVersion T>
int32_t ConvAddr<T>::get_num() {
  return num_;
}

template <DPUVersion T>
int32_t ConvAddr<T>::get_direction(int32_t idx) {
  auto convaddr_type = (conv_addr_type_t)convaddr_info[idx].type;
  switch (convaddr_type) {
    case conv_addr_type_t::CONV_ADDR_IFM:
    case conv_addr_type_t::CONV_ADDR_IFM_ELEW:
    case conv_addr_type_t::CONV_ADDR_WGT:
    case conv_addr_type_t::CONV_ADDR_BIAS:
      return 1;
    case conv_addr_type_t::CONV_ADDR_OFM:
      return 0;
  }
  return -1;  // untouchable.
}

template <DPUVersion T>
int32_t ConvAddr<T>::get_h_num(int32_t idx) {
  return convaddr_info[idx].h_num;
}

template <DPUVersion T>
int32_t ConvAddr<T>::get_mt_addr(int32_t idx) {
  return convaddr_info[idx].mt_addr;
}

template <DPUVersion T>
int32_t ConvAddr<T>::get_invalid(int32_t idx) {
  return convaddr_info[idx].invalid;
}

template <DPUVersion T>
conv_addr_t& ConvAddr<T>::get_convaddr_info(int32_t idx) {
  return convaddr_info[idx];
}

template <DPUVersion T>
void ConvAddr<T>::clear() {
  num_ = 0;
  convaddr_info.clear();
}

template <DPUVersion T>
conv_addr_type_t ConvAddr<T>::get_type(int32_t idx) {
  return (conv_addr_type_t)convaddr_info[idx].type;
}

template class ConvAddr<DPUVersion::XV2DPU>;
