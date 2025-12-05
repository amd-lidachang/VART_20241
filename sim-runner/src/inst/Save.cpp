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
#include "Save.hpp"

#include <memory.h>

#include "SimCfg.hpp"
#include "pub/DumpBank.hpp"
#include "xv2dpu/simCfg.hpp"

template <>
Save<DPUVersion::DPUV2>::Save(int inst_type, int instid,
                              vector<string>& inst_str,
                              vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV2>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV2>::inst_table;
  // get save instruction field value
  hp_id_ = field_val_[TABLE::SAVE_FIELD_HP_ID];
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
  const_en_ = field_val_[TABLE::SAVE_FIELD_CONST_EN];
  const_value_ = field_val_[TABLE::SAVE_FIELD_CONST_VALUE];
  argmax_ = field_val_[TABLE::SAVE_FIELD_ARGMAX];
}

template <>
Save<DPUVersion::DPUV3E>::Save(int inst_type, int instid,
                               vector<string>& inst_str,
                               vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV3E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV3E>::inst_table;
  // get save instruction field value
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <>
Save<DPUVersion::DPUV4E>::Save(int inst_type, int instid,
                               vector<string>& inst_str,
                               vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV4E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV4E>::inst_table;
  // get save instruction field value
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <>
Save<DPUVersion::DPUV3ME>::Save(int inst_type, int instid,
                                vector<string>& inst_str,
                                vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV3ME>::InstInit(inst_type_, root_debug_path_, debug_,
                                        opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV3ME>::inst_table;
  // get save instruction field value
  hp_id_ = field_val_[TABLE::SAVE_FIELD_HP_ID];
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <>
Save<DPUVersion::XVDPU>::Save(int inst_type, int instid,
                              vector<string>& inst_str,
                              vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::XVDPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XVDPU>::inst_table;
  // get save instruction field value
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];

  argmax_ = field_val_[TABLE::SAVE_FIELD_ARGMAX];

  const_en_ = field_val_[TABLE::SAVE_FIELD_CONST_EN];
  const_value_ = field_val_[TABLE::SAVE_FIELD_CONST_VALUE];
}

template <>
Save<DPUVersion::XV2DPU>::Save(int inst_type, int instid,
                               vector<string>& inst_str,
                               vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::XV2DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XV2DPU>::inst_table;
  // get save instruction field value
  const_en_ = field_val_[TABLE::SAVE_FIELD_CONST_EN];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_MT_ADDR];

  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  const_value_ = field_val_[TABLE::SAVE_FIELD_CONST_VALUE];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];

  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];

  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];

  bank_id_ = 0;
  argmax_ = field_val_[TABLE::SAVE_FIELD_ARGMAX];
}

template <>
Save<DPUVersion::DPU4F>::Save(int inst_type, int instid,
                              vector<string>& inst_str,
                              vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPU4F>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPU4F>::inst_table;
  // get save instruction field value
  hp_id_ = field_val_[TABLE::SAVE_FIELD_HP_ID];
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  quant_lth_ = field_val_[TABLE::SAVE_FIELD_QUANT_LTH];  // 0: 4 bit 1: 8 bit
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <DPUVersion T>
Save<T>::~Save() {}

// public funcs
template <DPUVersion T>
void Save<T>::Exec() {
  target_check();
  size_t ddr_size = 0;
  auto* ddr_img = DDR::Instance().GetAddr(reg_id_, 0, &ddr_size);
  UNI_LOG_CHECK((int32_t)(ddr_addr_ + (length_ - 1) * jump_write_ + channel_) <=
                    static_cast<int32_t>(DDR::Instance().GetSize(reg_id_)),
                SIM_OUT_OF_RANGE);
  UNI_LOG_CHECK(argmax_ >= 0 && argmax_ <= 3, SIM_OUT_OF_RANGE);
  // decode argmax
  auto if_argmax = (this->argmax_ >> 1);
  std::string argmax_rlt =
      (this->argmax_ & 0x1) ? std::string{"data"} : std::string{"index"};

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  for (auto pixel = 0; pixel < length_; pixel++) {
    auto ddr_addr = ddr_addr_ + pixel * jump_write_;

    if (const_en_) {  // write const
      memset(ddr_img + ddr_addr, const_value_, channel_);
    } else if (if_argmax) {  // enable argmax
      auto bank_addr = (bank_addr_ + pixel * jump_read_) % bank_depth;
      auto data = vector<DPU_DATA_TYPE>(channel_);
      bank->Read(bank_addr, channel_, data.data());
      auto max_index = argmax(data);
      ddr_img[ddr_addr] =
          argmax_rlt == std::string{"index"} ? max_index : data.at(max_index);
    } else {  // normal save
      auto bank_addr = (bank_addr_ + pixel * jump_read_) % bank_depth;
      UNI_LOG_CHECK_LT(ddr_addr, ddr_size, SIM_ADDR_OVERFLOW);
      bank->Read(bank_addr, channel_,
                 reinterpret_cast<DPU_DATA_TYPE*>(ddr_img + ddr_addr));
      if (SimCfg::Instance().get_memory_doubleWrite_check()) {
        DDRblock ddrBlock(reg_id_, ddr_addr, channel_);
        DDRblock::insert(debug_path_, ddrBlock);
      }
    }
  }

  // record ddr used
  int hp_width = SimCfg::Instance().get_hp_width();
  for (auto pixel = 0; pixel < length_; pixel++) {
    auto addr_base = ddr_addr_ + pixel * jump_write_;
    int32_t num_bytes = hp_width - (addr_base % hp_width);
    int num_entry = ceil((double)channel_ / hp_width);
    if (num_bytes != 0) {
      if (channel_ <= num_bytes)
        num_entry = 1;
      else
        num_entry = ceil((double)(channel_ - num_bytes) / hp_width) + 1;
    }

    for (auto i = 0; i < num_entry; i++) {
      auto addr = addr_base + i * hp_width;
      DDR::Instance().set_addr_used(reg_id_, addr);
    }
  }

  debug_tick();

  save_ofm_PL();
  save_gen_ctrlpkg();
}

// public funcs
template <>
void Save<DPUVersion::DPU4F>::Exec() {
  target_check();
  auto* ddr_img = DDR::Instance().GetAddr(reg_id_, 0);
  UNI_LOG_CHECK(length_ * jump_write_ <=
                    static_cast<int32_t>(DDR::Instance().GetSize(reg_id_)),
                SIM_OUT_OF_RANGE);
  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();

  for (auto pixel = 0; pixel < length_; pixel++) {
    auto ddr_addr = ddr_addr_ + pixel * jump_write_;
    auto bank_addr = (bank_addr_ + pixel * jump_read_) % bank_depth;
    vector<DPU_DATA_TYPE> data_buf(channel_);
    BankShell::read(quant_lth_, bank_id_, channel_, bank_addr, data_buf.data());
    combine(quant_lth_, data_buf,
            reinterpret_cast<DPU_DATA_TYPE*>(ddr_img + ddr_addr));
  }
  debug_tick();
}

template <DPUVersion T>
void Save<T>::combine(bool is_8_bit, vector<DPU_DATA_TYPE> data_in,
                      DPU_DATA_TYPE* data_out) {
  auto num =
      is_8_bit ? data_in.size() : (data_in.size() / 2 + data_in.size() % 2);
  for (auto idx_data = 0U; idx_data < num; idx_data++) {
    if (is_8_bit)
      data_out[idx_data] = data_in.at(idx_data);
    else {
      auto lo = data_in.at(2 * idx_data);
      auto hi = static_cast<DPU_DATA_TYPE>(
          ((idx_data < num - 1) || (data_in.size() % 2 == 0))
              ? data_in.at(2 * idx_data + 1)
              : 0);
      data_out[idx_data] = static_cast<DPU_DATA_TYPE>((hi << 4) | (0x0F & lo));
    }
  }
}

template <DPUVersion T>
DPU_DATA_TYPE Save<T>::argmax(const vector<DPU_DATA_TYPE>& data_in) {
  auto num = data_in.size();

  DPU_DATA_TYPE pos = 0;
  auto max = data_in[pos];
  for (auto idx_data = 1U; idx_data < num; ++idx_data) {
    if (data_in[idx_data] > max) {
      max = data_in[idx_data];
      pos = idx_data;
    }
  }
  return pos;
}

template <DPUVersion T>
void Save<T>::debug_tick() {
  if (!debug_) return;

  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  int cg = ceil((double)channel_ / bank_width);
  int pixel_channel = cg * bank_width;
  vector<DPU_DATA_TYPE> buf(pixel_channel);

  for (int i = 0; i < length_; i++) {
    int bank_addr = bank_addr_ + i * jump_read_;

    bank->Read(bank_addr, pixel_channel, buf.data());
    Util::SaveHexContBigEndBankAddr(
        save_name, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_width * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);
  }
}

template <>
void Save<DPUVersion::DPUV4E>::debug_tick() {
  if (!debug_) return;

  string save_name_input = debug_path_ + "input_inst_" + to_string(instid_) +
                           "tick." + to_string(instid_classify_);
  string save_name_output = debug_path_ + "output_inst_" + to_string(instid_) +
                            "tick." + to_string(instid_classify_);

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  int cg = ceil((double)channel_ / bank_width);
  int pixel_channel = cg * bank_width;
  vector<DPU_DATA_TYPE> buf(pixel_channel);

  auto* ddr_buf = reinterpret_cast<DPU_DATA_TYPE*>(
      DDR::Instance().GetAddr(reg_id_, ddr_addr_));

  for (auto idx_length = 0; idx_length < length_; idx_length++) {
    auto bank_addr = idx_length * jump_read_;

    bank->Read(bank_addr, pixel_channel, buf.data());
    Util::SaveHexContSmallEndBankAddr(
        save_name_input, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_width * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);

    auto ddr_offset = ddr_addr_ + idx_length * jump_write_;
    Util::SaveHexContSmallEndDDRAddr(save_name_output, &ddr_buf[ddr_offset],
                                     channel_ * sizeof(DPU_DATA_TYPE),
                                     hp_width_, ddr_addr_ + ddr_offset, reg_id_,
                                     SM_APPEND);
  }
}
template <>
void Save<DPUVersion::DPU4F>::debug_tick() {
  if (!debug_) return;

  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  // how many DPU_DATA_TYPE element save engine can write to DDR at once
  auto cp = ArchCfg::Instance().get_param().save_engine().channel_parallel();
  auto bank_w = quant_lth_ ? bank_width : bank_width / 2;
  cp = quant_lth_ ? cp : 2 * cp;
  int cg = ceil((double)channel_ / cp);
  int pixel_channel = cg * cp;
  vector<DPU_DATA_TYPE> inter_buf(pixel_channel);
  auto data_size = quant_lth_ ? pixel_channel : pixel_channel / 2;
  vector<DPU_DATA_TYPE> buf(data_size);

  for (int i = 0; i < length_; i++) {
    int bank_addr = (bank_addr_ + i * jump_read_) % bank_depth;

    BankShell::read(quant_lth_, bank_id_, pixel_channel, bank_addr,
                    inter_buf.data());
    combine(quant_lth_, inter_buf, buf.data());
    Util::SaveHexContSmallEndBankAddr(
        save_name, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_w * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);
  }
}

template <DPUVersion T>
void Save<T>::target_check() {
  // bank id check
  auto white_list = ArchCfg::Instance().get_bank_access_white_list("save-in");
  UNI_LOG_CHECK((white_list.count(bank_id_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id (= " << bank_id_ << ") does not match target!" << endl;
}

template <DPUVersion T>
void Save<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}
