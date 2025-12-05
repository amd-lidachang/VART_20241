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
#include "AluAddr.hpp"

#include "AluInit.hpp"

template <DPUVersion dv>
int32_t AluAddr<dv>::num_ = 0;
template <DPUVersion dv>
std::vector<alu_addr_t> AluAddr<dv>::aluaddr_info;

template <DPUVersion T>
AluAddr<T>::AluAddr(int inst_type, int instid, vector<string>& inst_str,
                    vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_ALU);
  Helper<DPUVersion::XV2DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XV2DPU>::inst_table;

  local_type_ = field_val_[TABLE::ALUADDR_FIELD_TYPE];
  local_h_num_ = field_val_[TABLE::ALUADDR_FIELD_H_NUM];
  local_mt_addr_ = field_val_[TABLE::ALUADDR_FIELD_MT_ADDR];
  local_macc_dim_ = field_val_[TABLE::ALUADDR_FIELD_MACC_DIM];
  local_id_ = field_val_[TABLE::ALUADDR_FIELD_ID];
  local_jump_ = field_val_[TABLE::ALUADDR_FIELD_JUMP];
  local_jump_endl_ = field_val_[TABLE::ALUADDR_FIELD_JUMP_ENDL];
  local_invalid_ = field_val_[TABLE::ALUADDR_FIELD_INVALID];
  local_last_ = field_val_[TABLE::ALUADDR_FIELD_LAST];
  local_compression_ = field_val_[TABLE::ALUADDR_FIELD_COMPRESSION];

  ++num_;
  aluaddr_info.push_back({
      local_type_,
      local_last_,
      local_compression_,
      local_invalid_,
      local_h_num_,
      local_macc_dim_,
      local_mt_addr_,
      local_id_,
      local_jump_,
      local_jump_endl_,
  });

  auto aluinit_ptr = AluInit<DPUVersion::XV2DPU>::CUROBJ;
  auto alu_addr_type = alu_addr_type_t(local_type_);
  switch (alu_addr_type) {
    case alu_addr_type_t::ALU_ADDR_IFM:
      aluinit_ptr->set_jump_read(local_jump_);
      aluinit_ptr->set_jump_read_endl(local_jump_endl_);
      if (local_jump_endl_ == 0) {
        aluinit_ptr->set_share_pp(local_id_);
      }
      assert(local_id_ >= 0 && local_id_ < 4);
      break;
    case alu_addr_type_t::ALU_ADDR_WGT:
      aluinit_ptr->set_jump_read_weights(local_jump_);
      assert(local_id_ == 0);
      break;
    case alu_addr_type_t::ALU_ADDR_BIAS:
      assert(local_id_ == 0);
      break;
    case alu_addr_type_t::ALU_ADDR_OFM:
      aluinit_ptr->set_jump_write(local_jump_);
      aluinit_ptr->set_jump_write_endl(local_jump_endl_);
      assert(local_id_ == 0);
      break;
  }

  CUROBJ = this;
}

template <DPUVersion T>
void AluAddr<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}

template <DPUVersion T>
int32_t AluAddr<T>::get_num() {
  return num_;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_direction(int32_t idx) {
  auto aluaddr_type = alu_addr_type_t(aluaddr_info[idx].type);
  switch (aluaddr_type) {
    case alu_addr_type_t::ALU_ADDR_IFM:
    case alu_addr_type_t::ALU_ADDR_WGT:
    case alu_addr_type_t::ALU_ADDR_BIAS:
      return 1;
    case alu_addr_type_t::ALU_ADDR_OFM:
      return 0;
  }
  return -1;  // untouchable.
}

template <DPUVersion T>
int32_t AluAddr<T>::get_h_num(int32_t idx) {
  return aluaddr_info[idx].h_num;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_mt_addr(int32_t idx) {
  return aluaddr_info[idx].mt_addr;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_id(int32_t idx) {
  return aluaddr_info[idx].id;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_jump(int32_t idx) {
  return aluaddr_info[idx].jump;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_jump_endl(int32_t idx) {
  return aluaddr_info[idx].jump_endl;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_invalid(int32_t idx) {
  return aluaddr_info[idx].invalid;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_last(int32_t idx) {
  return aluaddr_info[idx].last;
}

template <DPUVersion T>
int32_t AluAddr<T>::get_compression(int32_t idx) {
  return aluaddr_info[idx].compression;
}

template <DPUVersion T>
alu_addr_type_t AluAddr<T>::get_type(int32_t idx) {
  return (alu_addr_type_t)aluaddr_info[idx].type;
}

template <DPUVersion T>
void AluAddr<T>::clear() {
  aluaddr_info.clear();
  num_ = 0;
}

template <DPUVersion T>
bool AluAddr<T>::check() {
  std::map<std::string, int> jump_map;  // {"dir_id_invalid", value}
  std::map<std::string, int> jump_endl_map;
  std::map<std::string, std::vector<int>> last_map;  // {"dir_id", value}

  for (int idx = 0; idx < num_; ++idx) {
    auto id = get_id(idx);
    auto type = get_type(idx);
    auto invalid = get_invalid(idx);
    auto jump = get_jump(idx);
    auto jump_endl = get_jump_endl(idx);
    auto last = get_last(idx);
    std::string dir_id = std::to_string((int)type) + "_" + std::to_string(id);
    std::string dir_id_invalid = dir_id + "_" + std::to_string(invalid);

    jump_map.insert({dir_id_invalid, jump});
    UNI_LOG_CHECK(jump_map[dir_id_invalid] == jump, VART_INVALID_VALUE)
        << "jump in different aluaddr with same id has different "
           "value";

    jump_endl_map.insert({dir_id_invalid, jump_endl});
    UNI_LOG_CHECK(jump_endl_map[dir_id_invalid] == jump_endl,
                  VART_INVALID_VALUE)
        << "jump_endl in different aluaddr with same id has different "
           "value";

    last_map[dir_id].push_back(last);
  }
  for (auto iter = last_map.begin(); iter != last_map.end(); ++iter) {
    auto num_last_1 = std::count(iter->second.begin(), iter->second.end(), 1);
    UNI_LOG_CHECK((0 == num_last_1) || (1 == num_last_1), VART_INVALID_VALUE)
        << "there must be no more than one last_1 in all aluaddrs with same id "
           "for input or "
           "output, but num_last_1 = "
        << num_last_1 << " is given by dir_id: " << iter->first;
  }

  return true;
}
template <DPUVersion T>
alu_addr_t &AluAddr<T>::get_alu_addr_info(int32_t idx) {
  return aluaddr_info[idx];
}

template class AluAddr<DPUVersion::XV2DPU>;
