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
#include <glog/logging.h>
#include <iomanip>
#include <iostream>

#include "vitis/ai/parse_value.hpp"
#include "xir/xrt_device_handle.hpp"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
using namespace std;

void xdpu_set_reg(const xir::XrtDeviceHandle* handle, const std::string& cu_name,
                  size_t cu_index) {
  struct reg {
    uint32_t addr;
    uint32_t value;
    std::string name;
  };
  std::vector<reg> regs;
  std::ifstream stream("set_reg.conf");
  std::string name;
  std::string offset_add;
  std::string offset_val;
  while ((stream >> name >> offset_add >> offset_val).good()) {
    uint64_t offset_add2;
    uint64_t offset_val2;
    // LOG(INFO) << "name=" << name << " offset_add=" << offset_add << "
    // offset_val=" << offset_val;
    vitis::ai::parse_value(offset_add, offset_add2);
    vitis::ai::parse_value(offset_val, offset_val2);
    regs.emplace_back(reg{(uint32_t)offset_add2, (uint32_t)offset_val2, name});
  }
  stream.close();
  for (const auto& reg : regs) {
    handle->write_register(cu_name, cu_index, reg.addr, reg.value);
    LOG_IF(INFO, true) << "0x" << std::hex << reg.addr << "\t"  //
                       << std::setfill(' ') << std::hex << "0x" << std::setw(16)
                       << std::left << reg.value << " "  //
                       << std::dec << std::setw(16) << std::right << reg.value
                       << "\t"  //
                       << reg.name << "\t" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  auto h = xir::XrtDeviceHandle::get_instance();
  auto cu_name = std::string(argv[1]);
  auto cu_index = std::stoi(std::string(argv[2]));
  xdpu_set_reg(h.get(), cu_name, cu_index);
  return 0;
}
