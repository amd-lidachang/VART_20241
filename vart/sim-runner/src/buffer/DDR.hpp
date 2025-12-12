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
#pragma once

#include <string>
#include <tuple>
#include <vector>
#include "SimCfg.hpp"
#include "util/LoadCfg.hpp"

namespace xir {
class Subgraph;
}

struct Reg {
  int id;
  uint64_t size;
  std::vector<char> data;
  bool isUsed;
  // record reg data used addr, 16xhp_width per block
  std::vector<uint64_t> usedLine;
};

class DDR {
 public:
  static DDR& Instance() {
    static DDR ddr;
    return ddr;
  }
  void Initial();                           // init by config file
  void Initial(const xir::Subgraph* subg);  // init by xir subgraph

 public:
  char* GetAddr(int reg_id, uint64_t offset, size_t* data_size = nullptr);
  uint64_t GetSize(int reg_id) const;
  void SaveReg(const std::string file);
  void SaveDDR_cosim(std::string save_name, int fmt, bool skip = false);
  void SaveDDR(std::string save_name, int fmt, bool skip = false);
  void SaveDDR(std::string save_name,
               const std::vector<std::tuple<int32_t, int32_t, int32_t>>& regs,
               int fmt, bool skip = false);
  void set_addr_used(const int32_t reg_id, const uint64_t offset);

  int32_t GetDataRegNum() { return data_reg_num_; }
  int32_t GetCodeRegId() { return code_reg_id_; }

 private:
  explicit DDR();
  DDR(const DDR&) = delete;
  DDR& operator=(const DDR&) = delete;
  int GetRegID(std::string reg_id);

 private:
  void RegConf(const string& regcfg_file = "./config/RegCfg.txt");
  void DDRConf(const string& ddrcfg_file = "./config/DDRCfg.txt");
  // random
  void init_ddr(const int32_t& id);
  void init_ddr(const string& init_file);
  // copy ddr_buf_;
  void copy_ddr_buff();
  void insertUsedLine(const int isinit, const int32_t reg_id,
                      const uint64_t offset);
  void saveRegUsedBlk(const string& save_name, const Reg& reg, int fmt);

 private:
  std::map<int, Reg> ddr_buf_;
  std::map<int, Reg> ddr_buf_init_;
  std::map<std::string, std::string> reg_id_to_context_type_;
  int32_t data_reg_num_;
  int32_t code_reg_id_;
};
