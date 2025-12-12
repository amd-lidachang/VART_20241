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
#include "buffer/DDR.hpp"
#include "SimCfg.hpp"
#include "UniLog/UniLog.hpp"
#include "util/Util.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/tensor/tensor.hpp"

DDR::DDR() {
  data_reg_num_ =
      (ArchCfg::Instance().get_param().type() == "DPUCV2DX8G") ? 256 : 8;
  code_reg_id_ = (ArchCfg::Instance().get_param().type() == "DPUCV2DX8G")
                     ? data_reg_num_ + 1
                     : data_reg_num_;
}

char* DDR::GetAddr(int reg_id, uint64_t offset, size_t *data_size) {
  UNI_LOG_CHECK(ddr_buf_.find(reg_id) != ddr_buf_.end(), SIM_OUT_OF_RANGE)
      << "ddr space for reg_id: " << reg_id << "is not allocated!";

  auto& reg = ddr_buf_.at(reg_id);
  UNI_LOG_CHECK(offset < reg.size, SIM_OUT_OF_RANGE)
      << "ddr reg id: " << reg_id << ", offset: " << offset;
  if (data_size) {
    *data_size = reg.data.size();
  }
  return ((reg.data).data() + offset);
}

uint64_t DDR::GetSize(int reg_id) const {
  UNI_LOG_CHECK(ddr_buf_.find(reg_id) != ddr_buf_.end(), SIM_OUT_OF_RANGE)
      << "ddr space for reg_id: " << reg_id << "is not allocated!";

  return ddr_buf_.at(reg_id).size;
}

void DDR::SaveReg(const std::string file) {
  ofstream ofs(file, std::ios::app);
  Util::ChkOpen(ofs, file);

  ofs << std::endl;
  ofs << "# memory size for each reg id" << std::endl;
  for (auto& item : ddr_buf_) {
    auto& reg = item.second;
    ofs << "reg" << reg.id << " : 0x" << hex << reg.size << std::endl;
  }
  ofs << std::endl;
  ofs << "# memory type: 0-code, 1-parameter, 2-data" << std::endl;
  for (auto& reg : reg_id_to_context_type_) {
    ofs << "reg" << GetRegID(reg.first) << " : " << reg.second << std::endl;
  }
  ofs.close();
}

void DDR::SaveDDR(std::string save_name, int fmt, bool skip) {
  auto hp_width_ = SimCfg::Instance().get_hp_width();
  UNI_LOG_CHECK(hp_width_ % sizeof(DPU_DATA_TYPE) == 0, SIM_PARAMETER_FAILED)
      << "hp_width doesn't meet the requirement!";

  std::string ori_name = save_name;
  if (!SimCfg::Instance().get_ddr_dump_split()) {
    save_name += Util::GetFileNameSuffix(fmt);
    Util::ChkFile(save_name, true);
  }

  int saveDDRFast = SimCfg::Instance().get_ddr_dump_end_fast() & 0x01;
  for (auto& item : ddr_buf_) {
    auto reg_id = item.first;
    auto& reg = item.second;
    if (reg.data.size() == 0) continue;
    if (saveDDRFast && (reg.isUsed == false)) continue;
    auto buf = reg.data;

    if (skip && (reg.id >= data_reg_num_)) continue;

    if (SimCfg::Instance().get_ddr_dump_split()) {
      save_name = ori_name + "_reg_" + std::to_string(reg.id) +
                  Util::GetFileNameSuffix(fmt);
      Util::ChkFile(save_name, true);
    }

    if (fmt <= DATA_FMT_HEX_CONT_BIGEND) {
      auto* data_dtype_ptr = reinterpret_cast<DPU_DATA_TYPE*>(buf.data());
      auto data_dtype_size = buf.size() / sizeof(DPU_DATA_TYPE);
      auto line_size =
          (fmt == DATA_FMT_DEC) ? 1 : (hp_width_ / sizeof(DPU_DATA_TYPE));
      Util::SaveData(fmt, save_name, data_dtype_ptr, data_dtype_size, line_size,
                     SM_APPEND);
    } else if (fmt == DATA_FMT_HEX_CONT_SMALLEND_DDRADDR) {
      if (saveDDRFast == 1) {
        saveRegUsedBlk(save_name, reg, fmt);
        saveRegUsedBlk(save_name + ".init", ddr_buf_init_.at(reg_id), fmt);
      } else {
        Util::SaveHexContSmallEndDDRAddr(save_name, buf.data(), buf.size(),
                                         hp_width_, 0, reg.id, SM_APPEND);
      }
    } else if (fmt == DATA_FMT_HEX_CONT_BIGEND_DDRADDR) {
      if (saveDDRFast == 1) {
        saveRegUsedBlk(save_name, reg, fmt);
        saveRegUsedBlk(save_name + ".init", ddr_buf_init_.at(reg_id), fmt);
      } else {
        Util::SaveHexContBigEndDDRAddr(save_name, buf.data(), buf.size(),
                                       hp_width_, 0, reg.id, SM_APPEND);
      }
    } else {
      UNI_LOG_FATAL(SIM_OUT_OF_RANGE) << "Not support fmt: " << fmt;
    }
  }
}

void DDR::SaveDDR_cosim(std::string save_name, int fmt, bool skip) {
  auto hp_width_ = SimCfg::Instance().get_hp_width();
  UNI_LOG_CHECK(hp_width_ % sizeof(DPU_DATA_TYPE) == 0, SIM_PARAMETER_FAILED)
      << "hp_width doesn't meet the requirement!";

  std::string ori_name = save_name;
  if (!SimCfg::Instance().get_ddr_dump_split()) {
    Util::ChkFile(save_name, true);
  }

  bool if_dump_all = SimCfg::Instance().get_dump_ddr_all();
  bool saveDDRFast = (!if_dump_all) &&
                     (1 == (SimCfg::Instance().get_ddr_dump_end_fast() & 0x01));

  for (auto& item : ddr_buf_) {
    auto reg_id = item.first;
    if (if_dump_all) {
      if (256 == reg_id || 257 == reg_id || 258 == reg_id) {
        continue;
      }
    }

    auto& reg = item.second;
    if (reg.data.size() == 0) continue;
    if (saveDDRFast && (reg.isUsed == false)) continue;
    auto buf = reg.data;

    if (skip && (reg.id >= data_reg_num_)) continue;

    if (SimCfg::Instance().get_ddr_dump_split()) {
      save_name = ori_name + "_reg_" + std::to_string(reg.id) +
                  Util::GetFileNameSuffix(fmt);
      Util::ChkFile(save_name, true);
    }

    if (fmt <= DATA_FMT_HEX_CONT_BIGEND) {
      auto* data_dtype_ptr = reinterpret_cast<DPU_DATA_TYPE*>(buf.data());
      auto data_dtype_size = buf.size() / sizeof(DPU_DATA_TYPE);
      auto line_size =
          (fmt == DATA_FMT_DEC) ? 1 : (hp_width_ / sizeof(DPU_DATA_TYPE));
      Util::SaveData(fmt, save_name, data_dtype_ptr, data_dtype_size, line_size,
                     SM_APPEND);
    } else if (fmt == DATA_FMT_HEX_CONT_SMALLEND_DDRADDR) {
      if (saveDDRFast) {
        saveRegUsedBlk(save_name, reg, fmt);
        saveRegUsedBlk(save_name + ".init", ddr_buf_init_.at(reg_id), fmt);
      } else {
        Util::SaveHexContSmallEndDDRAddr(save_name, buf.data(), buf.size(),
                                         hp_width_, 0, reg.id, SM_APPEND);
      }
    } else if (fmt == DATA_FMT_HEX_CONT_BIGEND_DDRADDR) {
      if (saveDDRFast) {
        saveRegUsedBlk(save_name, reg, fmt);
        saveRegUsedBlk(save_name + ".init", ddr_buf_init_.at(reg_id), fmt);
      } else {
        Util::SaveHexContBigEndDDRAddr(save_name, buf.data(), buf.size(),
                                       hp_width_, 0, reg.id, SM_APPEND);
      }
    } else {
      UNI_LOG_FATAL(SIM_OUT_OF_RANGE) << "Not support fmt: " << fmt;
    }
  }
}


void DDR::saveRegUsedBlk(const string& save_name, const Reg& reg, int fmt) {
  UNI_LOG_CHECK((fmt == DATA_FMT_HEX_CONT_SMALLEND_DDRADDR ||
                 fmt == DATA_FMT_HEX_CONT_BIGEND_DDRADDR),
                SIM_OUT_OF_RANGE)
      << "saveRegUsedBlk mode only support ddr format :["
      << DATA_FMT_HEX_CONT_BIGEND_DDRADDR << ", "
      << DATA_FMT_HEX_CONT_SMALLEND_DDRADDR << "]";
  int64_t iter_num = reg.usedLine.size();
  auto hp_width = SimCfg::Instance().get_hp_width();
  uint64_t num_bytes = hp_width * 16;
  auto& buf = reg.data;
  for (auto i = 0; i < iter_num; i++) {
    uint64_t addr_offset = num_bytes * reg.usedLine[i];
    uint32_t blk_size = (num_bytes > (reg.size - addr_offset))
                            ? (reg.size - addr_offset)
                            : num_bytes;
    if (fmt == DATA_FMT_HEX_CONT_SMALLEND_DDRADDR) {
      Util::SaveHexContSmallEndDDRAddr(save_name, buf.data() + addr_offset,
                                       blk_size, hp_width, addr_offset, reg.id,
                                       SM_APPEND);
    } else {
      Util::SaveHexContBigEndDDRAddr(save_name, buf.data() + addr_offset,
                                     blk_size, hp_width, addr_offset, reg.id,
                                     SM_APPEND);
    }
  }
}

void DDR::SaveDDR(
    std::string save_name,
    const std::vector<std::tuple<int32_t, int32_t, int32_t>>& regs, int fmt,
    bool skip) {
  auto hp_width_ = SimCfg::Instance().get_hp_width();
  UNI_LOG_CHECK(hp_width_ % sizeof(DPU_DATA_TYPE) == 0, SIM_PARAMETER_FAILED)
      << "hp_width doesn't meet the requirement!";

  std::string ori_name = save_name;
  if (!SimCfg::Instance().get_ddr_dump_split()) {
    Util::ChkFile(save_name, true);
  }

  for (auto& entry : regs) {
    auto reg_id = std::get<0>(entry);
    auto ddr_offset = std::get<1>(entry) / hp_width_ * hp_width_;
    auto size = std::get<2>(entry);
    size += std::get<1>(entry) - ddr_offset;
    size = (size + hp_width_ - 1) / hp_width_ * hp_width_;

    UNI_LOG_CHECK(ddr_buf_.find(reg_id) != ddr_buf_.end(), SIM_OUT_OF_RANGE)
        << "ddr space for reg_id: " << reg_id << "is not allocated!";
    auto buf = ddr_buf_.at(reg_id).data;

    if (skip && (reg_id >= data_reg_num_)) continue;

    if (SimCfg::Instance().get_ddr_dump_split()) {
      save_name = ori_name + "_reg_" + std::to_string(reg_id) +
                  Util::GetFileNameSuffix(fmt);
      Util::ChkFile(save_name, true);
    }

    if (fmt <= DATA_FMT_HEX_CONT_BIGEND) {
      auto* data_dtype_ptr =
          reinterpret_cast<DPU_DATA_TYPE*>(buf.data() + ddr_offset);
      auto line_size =
          (fmt == DATA_FMT_DEC) ? 1 : (hp_width_ / sizeof(DPU_DATA_TYPE));
      Util::SaveData(fmt, save_name, data_dtype_ptr,
                     size / sizeof(DPU_DATA_TYPE), line_size, SM_APPEND);
    } else if (fmt == DATA_FMT_HEX_CONT_SMALLEND_DDRADDR) {
      Util::SaveHexContSmallEndDDRAddr(save_name, buf.data() + ddr_offset, size,
                                       hp_width_, ddr_offset, reg_id,
                                       SM_APPEND);
    } else if (fmt == DATA_FMT_HEX_CONT_BIGEND_DDRADDR) {
      Util::SaveHexContBigEndDDRAddr(save_name, buf.data() + ddr_offset, size,
                                     hp_width_, ddr_offset, reg_id, SM_APPEND);
    } else {
      UNI_LOG_FATAL(SIM_OUT_OF_RANGE) << "Not support fmt: " << fmt;
    }
  }
}

int DDR::GetRegID(std::string reg_id) {
  int id;
  if (reg_id == "REG_CODE")
    id = data_reg_num_;  // default reg id for code
  else
    id = stoi(reg_id.substr(4));  // prefix: "REG_"
  return id;
}

void DDR::Initial(const xir::Subgraph* subg) {
  UNI_LOG_INFO << "DDR initializing...";
  auto reg_to_size =
      subg->get_attr<std::map<std::string, std::int32_t>>("reg_id_to_size");

  reg_id_to_context_type_ = subg->get_attr<std::map<std::string, std::string>>(
      "reg_id_to_context_type");

  ddr_buf_.clear();
  for (auto& e : reg_to_size) {
    struct Reg ddr;
    ddr.id = GetRegID(e.first);
    ddr.size = std::ceil((double)e.second / 4096) * 4096;
    ddr.data.resize(ddr.size);
    ddr_buf_.emplace(ddr.id, ddr);
  }

  if (subg->has_attr("reg_id_to_parameter_value")) {
    auto reg_to_data = subg->get_attr<std::map<std::string, std::vector<char>>>(
        "reg_id_to_parameter_value");
    for (auto& e : reg_to_data) {
      int reg_id = GetRegID(e.first);
      std::vector<char> dat = e.second;

      UNI_LOG_CHECK(ddr_buf_.find(reg_id) != ddr_buf_.end(), SIM_OUT_OF_RANGE)
          << "ddr space for reg_id: " << reg_id << "is not allocated!";

      auto& reg = ddr_buf_.at(reg_id);
      UNI_LOG_CHECK(dat.size() <= reg.size, SIM_OUT_OF_RANGE)
          << "ddr reg id: " << reg_id << ", initial size: " << dat.size();

      if (reg_id_to_context_type_.at(e.first) == "CONST") {
        std::copy(dat.begin(), dat.end(), reg.data.begin());
      }
    }
  }
}

void DDR::Initial() {
  UNI_LOG_DEBUG_INFO << "Initial DDR with config file...";
  // get reg ID, ddr size, init type, initfile name from file
  RegConf();
  DDRConf();

  // copy ddr to ddr init for onlu dump used ddr
  if (SimCfg::Instance().get_ddr_dump_end_fast() != 0) {
    copy_ddr_buff();
  }
}

void DDR::RegConf(const string& regcfg_file) {  // get reg config value
  LoadCfg lc(regcfg_file);
  const string HEX_CHARS = "xXabcdefABCDEF";
  for (auto key : lc.GetAllKeys()) {
    auto key_prefix = key.substr(0, 3);
    if ("reg" != key_prefix) {
      UNI_LOG_WARNING << "invalid attribute <" << key << "> in RegCfg.txt";
      continue;
    }
    auto reg_id = stoi(key.substr(3));  // prefix: "reg"

    auto reg = Reg{};
    reg.id = reg_id;

    string val = lc[key];
    val.erase(val.begin());
    val.erase(val.end() - 1);
    auto v = Util::Split(val);
    // get size value
    if (v[1].find_first_of(HEX_CHARS) == string::npos) {
      reg.size = stoul(v[1], nullptr, 10);
    } else {
      reg.size = stoul(v[1], nullptr, 16);
    }
    ddr_buf_.emplace(reg_id, reg);
  }

  // get ddr config value
  for (auto& item : ddr_buf_) {
    auto& reg = item.second;
    string reg_name;
    reg_name = "REG_" + std::to_string(reg.id);
    reg_id_to_context_type_.emplace(reg_name, "DATA");  // all taken as data
  }
}

void DDR::DDRConf(const string& ddrcfg_file) {
  LoadCfg lc(ddrcfg_file);

  // only support all init
  string val = lc["all"];
  val.erase(val.begin());
  val.erase(val.end() - 1);

  auto v = Util::Split(val, ", ");

  // without init
  if (!Util::Str2Bool(v[0])) return;

  auto data_type = stoul(v[1]);

  UNI_LOG_CHECK(data_type == DATA_FMT_HEX_CONT_SMALLEND_DDRADDR ||
                    data_type >= DATA_FMT_MAX,
                SIM_OUT_OF_RANGE)
      << "unsupported ddr type!";

  // file init
  if (data_type == DATA_FMT_HEX_CONT_SMALLEND_DDRADDR) {
    init_ddr(v[2]);
  } else {
    auto m = Util::Split(v[3], "-");
    for (auto e : m) {
      auto reg_id = stoul(e);
      init_ddr(reg_id);  // init reg_id
    }
  }
}
// random
void DDR::init_ddr(const int32_t& id) {
  UNI_LOG_CHECK(ddr_buf_.find(id) != ddr_buf_.end(), SIM_OUT_OF_RANGE)
      << "error: " << id << " >= " << ddr_buf_.size();
  ddr_buf_.at(id).data.resize(ddr_buf_.at(id).size);
  Util::Random(
      ddr_buf_.at(id).data.data(), static_cast<size_t>(ddr_buf_.at(id).size),
      static_cast<char>(-16), static_cast<char>(16), 12345);  // seed = 0
}

void DDR::copy_ddr_buff() {
  // cp init
  for (auto& item : ddr_buf_) {
    auto& reg = item.second;
    Reg cpyReg;
    cpyReg.id = reg.id;
    cpyReg.size = reg.size;
    cpyReg.data.resize(reg.data.size());
    cpyReg.data = reg.data;
    cpyReg.isUsed = false;
    ddr_buf_init_.emplace(reg.id, cpyReg);
  }
}

void DDR::insertUsedLine(const int isinit, const int32_t reg_id,
                         const uint64_t offset) {
  auto& usedVec = (isinit == 1) ? ddr_buf_.at(reg_id).usedLine
                                : ddr_buf_init_.at(reg_id).usedLine;
  auto hp_width = SimCfg::Instance().get_hp_width();
  uint64_t index = offset / (hp_width * 16);
  if (usedVec.size() == 0) {
    usedVec.push_back(index);
  } else {
    auto idx = 0u;
    int pos = -1;
    while (idx < usedVec.size()) {
      if (usedVec[idx] == index) {
        pos = -1;
        break;
      } else if (usedVec[idx] < index) {
        idx++;
        pos = idx;
      } else {
        pos = idx;
        break;
      }
    }
    if (pos != -1) {
      pos = (idx == usedVec.size() - 1) ? (usedVec.size() - 1) : pos;
      usedVec.insert(usedVec.begin() + pos, index);
    }
  }
}
void DDR::set_addr_used(const int32_t reg_id, const uint64_t offset) {
  if (SimCfg::Instance().get_ddr_dump_end_fast() == 0) return;
  ddr_buf_init_.at(reg_id).isUsed = true;
  ddr_buf_.at(reg_id).isUsed = true;
  insertUsedLine(0, reg_id, offset);
  insertUsedLine(1, reg_id, offset);
}
void DDR::init_ddr(const string& init_file) {
  std::fstream f(init_file);
  Util::ChkOpen(f, init_file);

  std::for_each(ddr_buf_.begin(), ddr_buf_.end(), [](auto& item) {
    auto& reg = item.second;
    if (reg.data.size() == 0) {
      reg.data.resize(reg.size);
    }
  });

  auto hp_width = SimCfg::Instance().get_hp_width();

  // read ddr init file, put related value into DDR according to its address
  // ddr init file format is as follows:
  // 1-0x0000000000 : 00000000000000000000000000000000
  for (std::string line; std::getline(f, line);) {
    if (line.empty()) continue;

    auto pos_dash = line.find_first_of("-");
    auto reg_id = std::stoul(line.substr(0, pos_dash));
    auto pos_colon = line.find_first_of(" : ", pos_dash + 1);
    auto offset = std::stoul(
        line.substr(pos_dash + 1, pos_colon - pos_dash - 1), nullptr, 16);
    auto value = line.substr(pos_colon + 3);

    // get every value and save it into ddr_
    UNI_LOG_CHECK(value.size() == static_cast<uint32_t>(2 * hp_width),
                  SIM_PARAMETER_FAILED)
        << value.size() << " != " << 2 * hp_width;
    auto p = GetAddr(reg_id, offset);
    if (hp_width % 8 == 0) {
      auto num = hp_width / 8;
      auto ptr = reinterpret_cast<uint64_t*>(p);
      for (int j = 0; j < num; j++) {
        *ptr++ = stoull(value.substr((num - j - 1) * 16, 16), nullptr, 16);
      }
    } else {
      for (int j = 0; j < hp_width; j++) {
        int pos = (2 * hp_width - 2 * j - 2);
        p[j] = static_cast<char>(stoi(string(value, pos, 2), nullptr, 16));
      }
    }
  }
  f.close();
}
