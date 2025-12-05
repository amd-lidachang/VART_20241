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
#include "ArchCfg.hpp"
#include <regex>

void ArchCfg::set_param(vitis::ai::Target param) {
  arch_param_ = param;
  init();
}

const vitis::ai::Target& ArchCfg::get_param() { return arch_param_; }

// bank_access
void ArchCfg::trans_bank_group_info(
    std::map<std::string, std::vector<std::int32_t>>& data_bank_group,
    std::map<std::string, std::vector<std::int32_t>>& param_bank_group) {
  for (auto& this_bank_group : arch_param_.bank_group()) {
    if (this_bank_group.type() == "Param") {
      param_bank_group[this_bank_group.name()].emplace_back(
          this_bank_group.base_id());
      param_bank_group[this_bank_group.name()].emplace_back(
          this_bank_group.bank_num());
    } else {
      auto bank_group_name{this_bank_group.name()};
      if (this_bank_group.type() == "Special") bank_group_name += "-Special";
      data_bank_group[bank_group_name].emplace_back(this_bank_group.base_id());
      data_bank_group[bank_group_name].emplace_back(this_bank_group.bank_num());
    }
  }
}

void ArchCfg::set_access_white_list(
    const string& bank_group_name,
    const std::map<std::string, std::vector<std::int32_t>>& bank_group_map,
    const string& type, const bool& has_virtual) {
  auto this_bank_group = bank_group_map.find(bank_group_name);
  if (this_bank_group == bank_group_map.end())
    this_bank_group = bank_group_map.find(bank_group_name + "-Special");
  UNI_LOG_CHECK(((this_bank_group != bank_group_map.end()) &&
                 (this_bank_group->second.size() == 2U)),
                SIM_PARAMETER_FAILED)
      << bank_group_name << " has not been initial by trans_bank_group_info()!";

  auto lower = this_bank_group->second[0];
  auto upper = lower + this_bank_group->second[1] - 1;
  for (; lower <= upper; lower++) bank_access_[type].insert(lower);
  if (!has_virtual) return;

  lower = static_cast<std::int32_t>(
      this_bank_group->second[0] + 200 +
      bank_group_map.at(arch_param_.eltwise_engine().input_bank()[0])[0]);
  upper = lower + this_bank_group->second[1] - 1;
  for (; lower <= upper; lower++) bank_access_[type].insert(lower);
}

void ArchCfg::init_bank_access_lists() {
  std::map<std::string, std::vector<std::int32_t>> data_bank_group;
  std::map<std::string, std::vector<std::int32_t>> param_bank_group;
  auto has_virtual_{false};
  // if ((arch_param_.feature_code() && 0x0000000000000400) ==
  // 0x0000000000000400)
  if (arch_param_.type() == "DPUCVDX8H") has_virtual_ = true;
  trans_bank_group_info(data_bank_group, param_bank_group);
  // init load
  if (arch_param_.has_load_engine()) {
    for (auto& bank_group_name : arch_param_.load_engine().output_bank())
      set_access_white_list(bank_group_name, data_bank_group, "load-out");
    for (auto& this_bank_group : arch_param_.bank_group()) {
      if (this_bank_group.type() == "Param") {
        set_access_white_list(this_bank_group.name(), param_bank_group,
                              "load-out");
      }
    }
    if (arch_param_.type() == "DPUCAHX8H")
      bank_access_["load-out"].insert(static_cast<std::int32_t>(63));
    if (arch_param_.type() == "DPUCVDX8H")
      bank_access_["load-out"].insert(static_cast<std::int32_t>(255));
  }

  // init save
  if (arch_param_.has_save_engine()) {
    for (auto& bank_group_name : arch_param_.save_engine().input_bank())
      set_access_white_list(bank_group_name, data_bank_group, "save-in");
  }
  // init conv
  if (arch_param_.has_conv_engine()) {
    for (auto& bank_group_name : arch_param_.conv_engine().input_bank())
      set_access_white_list(bank_group_name, data_bank_group, "conv-in");
    for (auto& bank_group_name : arch_param_.conv_engine().output_bank()) {
      set_access_white_list(bank_group_name, data_bank_group, "conv-out",
                            has_virtual_);

      if (arch_param_.type() == "DPUCAHX8L") {
        auto type =
            (data_bank_group.find(bank_group_name) == data_bank_group.end())
                ? "conv-out-cb"
                : "conv-out-hbm";
        set_access_white_list(bank_group_name, data_bank_group, type);
      }
    }
  }
  // init eltwise
  if (arch_param_.has_eltwise_engine()) {
    for (auto& bank_group_name : arch_param_.eltwise_engine().input_bank()) {
      set_access_white_list(bank_group_name, data_bank_group, "elew-in",
                            has_virtual_);
    }
    for (auto& bank_group_name : arch_param_.eltwise_engine().output_bank())
      set_access_white_list(bank_group_name, data_bank_group, "elew-out");
  }
  // init pool
  if (arch_param_.has_pool_engine()) {
    for (auto& bank_group_name : arch_param_.pool_engine().input_bank())
      set_access_white_list(bank_group_name, data_bank_group, "pool-in");
    for (auto& bank_group_name : arch_param_.pool_engine().output_bank())
      set_access_white_list(bank_group_name, data_bank_group, "pool-out");
  }
  // init dwconv
  if (arch_param_.has_dwconv_engine()) {
    for (auto& bank_group_name : arch_param_.dwconv_engine().input_bank())
      set_access_white_list(bank_group_name, data_bank_group, "dwconv-in");
    for (auto& bank_group_name : arch_param_.dwconv_engine().output_bank())
      set_access_white_list(bank_group_name, data_bank_group, "dwconv-out");
  }
  // init move
  // init threshold
  if (arch_param_.has_threshold_engine()) {
    for (auto& bank_group_name : arch_param_.threshold_engine().input_bank())
      set_access_white_list(bank_group_name, data_bank_group, "thd-in");
    for (auto& bank_group_name : arch_param_.threshold_engine().output_bank())
      set_access_white_list(bank_group_name, data_bank_group, "thd-out");
  }
  // init alu
  if (arch_param_.has_alu_engine()) {
    for (auto& bank_group_name : arch_param_.alu_engine().input_bank())
      set_access_white_list(bank_group_name, data_bank_group, "alu-in");
    for (auto& bank_group_name : arch_param_.alu_engine().output_bank())
      set_access_white_list(bank_group_name, data_bank_group, "alu-out");
  }
}

void ArchCfg::init_instr_limit_lists() {
  // kernel size, stride
  /// conv-limit
  if (arch_param_.has_conv_engine()) {
    if (arch_param_.conv_engine().has_conv_limit() &&
        arch_param_.conv_engine().conv_limit().kernel_size() != "") {
      instr_limit_["conv-kernel"] =
          parse_range(arch_param_.conv_engine().conv_limit().kernel_size());
    } else {
      instr_limit_["conv-kernel"] = parse_range("1-16");
    }

    if (arch_param_.conv_engine().has_conv_limit() &&
        arch_param_.conv_engine().conv_limit().stride() != "") {
      instr_limit_["conv-stride"] =
          parse_range(arch_param_.conv_engine().conv_limit().stride());
    } else {
      instr_limit_["conv-stride"] = parse_range("1-8");
    }
  }
  if (arch_param_.has_pool_engine()) {
    /// avg-limit
    if (arch_param_.pool_engine().has_avg_limit() &&
        arch_param_.pool_engine().avg_limit().kernel_size() != "") {
      instr_limit_["pool-avg-kernel"] =
          parse_range(arch_param_.pool_engine().avg_limit().kernel_size());
    } else {
      instr_limit_["pool-avg-kernel"] = parse_range("1-16");
    }
    if (arch_param_.pool_engine().has_avg_limit() &&
        arch_param_.pool_engine().avg_limit().stride() != "") {
      instr_limit_["pool-avg-stride"] =
          parse_range(arch_param_.pool_engine().avg_limit().stride());
    } else {
      instr_limit_["pool-avg-stride"] = parse_range("1-8");
    }
    /// max-limit
    if (arch_param_.pool_engine().has_max_limit() &&
        arch_param_.pool_engine().max_limit().kernel_size() != "") {
      instr_limit_["pool-max-kernel"] =
          parse_range(arch_param_.pool_engine().max_limit().kernel_size());
    } else {
      instr_limit_["pool-max-kernel"] = parse_range("1-16");
    }
    if (arch_param_.pool_engine().has_max_limit() &&
        arch_param_.pool_engine().max_limit().stride() != "") {
      instr_limit_["pool-max-stride"] =
          parse_range(arch_param_.pool_engine().max_limit().stride());
    } else {
      instr_limit_["pool-max-stride"] = parse_range("1-8");
    }
  }
  /// dwconv-limit
  if (arch_param_.has_dwconv_engine()) {
    if (arch_param_.dwconv_engine().has_dwconv_limit() &&
        arch_param_.dwconv_engine().dwconv_limit().kernel_size() != "") {
      instr_limit_["dwconv-kernel"] =
          parse_range(arch_param_.dwconv_engine().dwconv_limit().kernel_size());
    } else {
      instr_limit_["dwconv-kernel"] = parse_range("1-16");
    }

    if (arch_param_.dwconv_engine().has_dwconv_limit() &&
        arch_param_.dwconv_engine().dwconv_limit().stride() != "") {
      instr_limit_["dwconv-stride"] =
          parse_range(arch_param_.dwconv_engine().dwconv_limit().stride());
    } else {
      instr_limit_["dwconv-stride"] = parse_range("1-16");
    }
  }
  /// alu-limit
  if (arch_param_.has_alu_engine()) {
    if (arch_param_.alu_engine().has_alu_limit() &&
        arch_param_.alu_engine().alu_limit().kernel_size() != "") {
      instr_limit_["alu-kernel"] =
          parse_range(arch_param_.alu_engine().alu_limit().kernel_size());
    } else {
      instr_limit_["alu-kernel"] = parse_range("1-16");
    }

    if (arch_param_.alu_engine().has_alu_limit() &&
        arch_param_.alu_engine().alu_limit().stride() != "") {
      instr_limit_["alu-stride"] =
          parse_range(arch_param_.alu_engine().alu_limit().stride());
    } else {
      instr_limit_["alu-stride"] = parse_range("1-16");
    }
  }
}

void ArchCfg::init_alu_support_lists() {
  if (arch_param_.has_alu_engine()) {
    for (int i = 0; i < arch_param_.alu_engine().alu_type_size(); i++) {
      alu_support_list_.insert(arch_param_.alu_engine().alu_type(i));
    }
  }
}

std::set<std::int32_t> ArchCfg::get_bank_access_white_list(
    const std::string& access_type) {
  return bank_access_.at(access_type);
}

std::set<std::int32_t> ArchCfg::get_instr_limit_white_list(
    const std::string& limit_type) {
  return instr_limit_.at(limit_type);
}
std::unordered_set<std::int32_t> ArchCfg::get_alu_support_list() {
  return alu_support_list_;
}

// parse range
std::vector<std::string> ArchCfg::tokenize(const std::string& source,
                                           const std::string& delimiter) {
  std::vector<std::string> ret;
  size_t prev = 0, next = 0;
  while ((next = source.find_first_of(delimiter, prev)) != std::string::npos) {
    if (next - prev != 0) {
      ret.push_back(source.substr(prev, next - prev));
    }
    prev = next + 1;
  }
  if (prev < source.size()) {
    ret.push_back(source.substr(prev));
  }
  return ret;
}

std::vector<std::string> ArchCfg::match(const std::string& str,
                                        const std::string& regex) {
  const std::regex pieces_regex(regex);
  std::smatch pieces_match;
  std::vector<std::string> ret;
  if (std::regex_match(str, pieces_match, pieces_regex)) {
    for (auto& sub_match : pieces_match) {
      ret.push_back(sub_match.str());
    }
  }
  return ret;
}

std::set<std::int32_t> ArchCfg::parse_range(const std::string& range) {
  std::set<std::int32_t> ret;
  auto trim = range;
  trim.erase(std::remove_if(trim.begin(), trim.end(),
                            [](auto x) { return std::isspace(x); }),
             trim.end());
  auto token_list = tokenize(trim, ",");
  for (auto& token : token_list) {
    auto match_number = match(token, "^(\\d+)$");
    if (match_number.size() == 2) {
      ret.insert(std::stoi(match_number[1]));
      continue;
    }
    auto match_pair = match(token, "^(\\d+)-(\\d+)$");
    if (match_pair.size() == 3) {
      auto lower = std::stoi(match_pair[1]);
      auto upper = std::stoi(match_pair[2]);
      for (; lower <= upper; lower++) {
        ret.insert(lower);
      }
      continue;
    }
    UNI_LOG_FATAL(SIM_PARAMETER_FAILED)
        << "Cannot recognize range '" << range << "'";
  }
  return ret;
}

void ArchCfg::init_white_lists() {
  init_bank_access_lists();
  init_instr_limit_lists();
  init_alu_support_lists();
}

void ArchCfg::init() {
  bank_info_.clear();
  for (auto idx_bank_group = 0; idx_bank_group < arch_param_.bank_group_size();
       idx_bank_group++) {
    auto bank_group = arch_param_.bank_group(idx_bank_group);
    auto base_id = bank_group.base_id();
    for (auto idx_bank = 0U; idx_bank < bank_group.bank_num(); idx_bank++) {
      auto bank_id = base_id + idx_bank;
      bank_info_.emplace(
          bank_id,
          std::make_tuple(bank_group.name(), bank_group.type(),
                          bank_group.bank_width(), bank_group.bank_depth(),
                          base_id, bank_group.bank_num()));
    }
  }
}

uint32_t ArchCfg::get_base_bank_id(uint32_t bank_id) {
  UNI_LOG_CHECK(bank_info_.find(bank_id) != bank_info_.end(), SIM_OUT_OF_RANGE)
      << "invalid bank id!";
  return std::get<4>(bank_info_[bank_id]);
}
