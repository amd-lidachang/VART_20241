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

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <string>

#include "UniLog/UniLog.hpp"

// MSVC NOTE: we must have vitis/ai/target_factory.hpp to include definition
// TARGET_FACTORY_DLLESPEC
//
#include "vitis/ai/target_factory.hpp"

using namespace std;

using google::protobuf::io::FileInputStream;

class ArchCfg {
 public:
  using BankGroup = vitis::ai::Target::BankGroup;

  static ArchCfg& Instance() {
    static ArchCfg param;
    return param;
  }

  void set_param(vitis::ai::Target param);
  const vitis::ai::Target& get_param();
  // target related
  std::set<std::int32_t> get_bank_access_white_list(
      const std::string& access_type);
  std::set<std::int32_t> get_instr_limit_white_list(
      const std::string& limit_type);
  std::set<std::int32_t> parse_range(const std::string& range);
  void init_white_lists();
  std::unordered_set<std::int32_t> get_alu_support_list();

 private:
  ArchCfg() = default;
  ~ArchCfg() = default;

 private:
  // target related
  /// bank access
  void trans_bank_group_info(
      std::map<std::string, std::vector<std::int32_t>>& data_bank_group,
      std::map<std::string, std::vector<std::int32_t>>& param_bank_group);
  void set_access_white_list(
      //      const std::vector<std::string>& bank_groups,
      const string& bank_group_name,
      const std::map<std::string, std::vector<std::int32_t>>& bank_group_map,
      const string& type, const bool& has_virtual = false);

  /// parse range
  std::vector<std::string> tokenize(const std::string& source,
                                    const std::string& delimiter);
  std::vector<std::string> match(const std::string& str,
                                 const std::string& regex);
  void init_bank_access_lists();
  void init_instr_limit_lists();
  void init_alu_support_lists();

 private:
  vitis::ai::Target arch_param_;
  std::map<std::string, std::set<std::int32_t>> bank_access_;
  std::map<std::string, std::set<std::int32_t>> instr_limit_;
  std::unordered_set<int32_t> alu_support_list_;

 public:
  uint32_t get_base_bank_id(uint32_t bank_id);

 private:
  void init();
  void insert_bank_info(BankGroup bank_group);

  // map<bank_id, tuple<bank_name, bank_type, bank_width, bank_depth,
  // base_bank_id, bank_num>>
  std::map<uint32_t, std::tuple<std::string, std::string, uint32_t, uint32_t,
                                uint32_t, uint32_t>>
      bank_info_;
};

// Target::ArchParam ArchCfg::arch_param_;
