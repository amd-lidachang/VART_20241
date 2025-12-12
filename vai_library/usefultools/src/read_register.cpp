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

#include <vitis/ai/profiling.hpp>

#include "tools_extra_ops.hpp"

#ifdef ENABLE_XRT
#include <iomanip>
#include <iostream>
#include <xir/xrt_device_handle.hpp>

#include "parse_value.hpp"

std::vector<uint32_t> read_register(const xir::XrtDeviceHandle* handle,
                                    const std::string& cu_name, size_t cu_index,
                                    const std::vector<uint32_t>& addrs) {
  __TIC__(READ_REGISTER)

  std::vector<uint32_t> values;
  for (auto addr : addrs) {
    values.push_back(handle->read_register(cu_name, cu_index, addr));
  }
  __TOC__(READ_REGISTER)
  return values;
}
#else
std::vector<uint32_t> read_register(const xir::XrtDeviceHandle* handle,
                                    const std::string& cu_name, size_t cu_index,
                                    const std::vector<uint32_t>& addrs) {
  LOG(INFO) << "xrt not found ";
  return std::vector<uint32_t>();
}
#endif
